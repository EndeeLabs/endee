#pragma once

// System includes
#include <string>
#include <memory>
#include <stdexcept>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "json/nlohmann_json.hpp"
#include "../utils/settings.hpp"
#include "mdbx/mdbx.h"
#include "../utils/log.hpp"
#include "../core/types.hpp"

#include "numeric_index.hpp"
#include "bitmap_index.hpp"

enum class FieldType : uint8_t {
    Unknown = 0,
    String = 1,
    Number = 2,  // Unified Integer and Float
    Bool = 4
};

class Filter {
private:
    MDBX_env* env_;
    MDBX_dbi dbi_;  // Used for schema storage
    std::string path_;
    std::unique_ptr<ndd::numeric::NumericIndex> numeric_index_;
    std::unique_ptr<ndd::filter::BitmapIndex> bitmap_index_;

    static constexpr const char* SCHEMA_KEY = "__ndd_schema_v1__";
    std::unordered_map<std::string, FieldType> schema_cache_;
    mutable std::mutex schema_mutex_;

    void load_schema() {
        MDBX_txn* txn;
        int rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_RDONLY, &txn);
        if(rc != MDBX_SUCCESS) {
            return;
        }

        MDBX_val key{const_cast<char*>(SCHEMA_KEY), strlen(SCHEMA_KEY)};
        MDBX_val data;
        rc = mdbx_get(txn, dbi_, &key, &data);

        if(rc == MDBX_SUCCESS && data.iov_len > 0) {
            try {
                std::string json_str(static_cast<const char*>(data.iov_base), data.iov_len);
                auto j = nlohmann::json::parse(json_str);
                std::lock_guard<std::mutex> lock(schema_mutex_);
                for(auto& [k, v] : j.items()) {
                    schema_cache_[k] = static_cast<FieldType>(v.get<int>());
                }
            } catch(...) {
                LOG_ERROR("Failed to load schema");
            }
        }
        mdbx_txn_abort(txn);
    }

    void save_schema_internal() {
        nlohmann::json j;
        for(const auto& [k, v] : schema_cache_) {
            j[k] = static_cast<int>(v);
        }
        std::string json_str = j.dump();

        MDBX_txn* txn;
        int rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_READWRITE, &txn);
        if(rc != MDBX_SUCCESS) {
            return;
        }

        MDBX_val key{const_cast<char*>(SCHEMA_KEY), strlen(SCHEMA_KEY)};
        MDBX_val data{const_cast<char*>(json_str.c_str()), json_str.size()};

        rc = mdbx_put(txn, dbi_, &key, &data, MDBX_UPSERT);
        if(rc == MDBX_SUCCESS) {
            mdbx_txn_commit(txn);
        } else {
            mdbx_txn_abort(txn);
        }
    }

    bool register_field_type(const std::string& field, FieldType type) {
        std::lock_guard<std::mutex> lock(schema_mutex_);
        auto it = schema_cache_.find(field);
        if(it != schema_cache_.end()) {
            return it->second == type;
        }

        schema_cache_[field] = type;
        save_schema_internal();
        return true;
    }

    void init_environment() {
        int rc = mdbx_env_create(&env_);
        if(rc != 0) {
            throw std::runtime_error("Failed to create LMDB env for filters");
        }
        // max DBs to allow multiple databases (main + schema + numeric_forward + numeric_inverted)
        mdbx_env_set_maxdbs(env_, 10);

        // Set
        // Set geometry for auto-grow: initial per settings, growth=256MB, max=32GB
        rc = mdbx_env_set_geometry(
                env_,
                -1,                                          // lower size bound (use default)
                1ULL << settings::FILTER_MAP_SIZE_BITS,      // current/now size
                1ULL << settings::FILTER_MAP_SIZE_MAX_BITS,  // upper size bound (32GB)
                1ULL << settings::FILTER_MAP_SIZE_BITS,      // growth step (256MB)
                -1,                                          // shrink threshold (use default)
                -1);                                         // pagesize (use default)
        if(rc != MDBX_SUCCESS) {
            throw std::runtime_error("Failed to set geometry for filters");
        }

        rc = mdbx_env_open(
                env_, path_.c_str(), MDBX_WRITEMAP | MDBX_MAPASYNC | MDBX_NORDAHEAD, 0664);
        if(rc != 0) {
            throw std::runtime_error("Failed to open filter environment");
        }

        MDBX_txn* txn;
        rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_READWRITE, &txn);
        if(rc != 0) {
            throw std::runtime_error("Failed to begin filter transaction");
        }

        rc = mdbx_dbi_open(txn, nullptr, MDBX_CREATE, &dbi_);
        if(rc != 0) {
            mdbx_txn_abort(txn);
            throw std::runtime_error("Failed to open filter database");
        }
        rc = mdbx_txn_commit(txn);
        if(rc != 0) {
            throw std::runtime_error("Failed to commit filter transaction");
        }

        // Initialize Indices
        numeric_index_ = std::make_unique<ndd::numeric::NumericIndex>(env_);
        bitmap_index_ = std::make_unique<ndd::filter::BitmapIndex>(env_);

        load_schema();
    }

    static std::string format_filter_key(const std::string& field, const std::string& value) {
        return field + ":" + value;
    }

public:
    Filter(const std::string& path) :
        path_(path) {
        std::filesystem::create_directories(path);
        init_environment();
    }

    ~Filter() {
        mdbx_dbi_close(env_, dbi_);
        mdbx_env_close(env_);
    }

    // Compute the filter bitmap based on the provided JSON filter array
    ndd::RoaringBitmap computeFilterBitmap(const nlohmann::json& filter_array) const {
        if(!filter_array.is_array()) {
            throw std::runtime_error("Filter must be an array");
        }

        if(filter_array.empty()) {
            LOG_DEBUG("Empty filter array, returning empty bitmap");
            return ndd::RoaringBitmap();
        }

        ndd::RoaringBitmap final_result;
        bool first = true;

        for(const auto& condition : filter_array) {
            if(!condition.is_object() || condition.size() != 1) {
                throw std::runtime_error("Each condition must be a single-field object");
            }

            const auto& field = condition.begin().key();
            const auto& expr = condition.begin().value();

            if(field.empty()) {
                throw std::runtime_error("Filter field name cannot be empty");
            }

            // Check schema for field type
            FieldType type = FieldType::Unknown;
            {
                std::lock_guard<std::mutex> lock(schema_mutex_);
                auto it = schema_cache_.find(field);
                if(it != schema_cache_.end()) {
                    type = it->second;
                }
            }

            ndd::RoaringBitmap or_result;

            if(!expr.is_object() || expr.size() != 1) {
                throw std::runtime_error("Operator must be a single-field object");
            }

            const std::string op = expr.begin().key();
            const auto& val = expr.begin().value();

            if(op == "$eq") {
                if(type == FieldType::Number) {
                    uint32_t sortable_val;
                    if(val.is_number_integer()) {
                        sortable_val = ndd::numeric::int_to_sortable(val.get<int>());
                    } else if(val.is_number()) {
                        sortable_val = ndd::numeric::float_to_sortable(val.get<float>());
                    } else {
                        throw std::runtime_error("$eq value for numeric field must be a number");
                    }
                    or_result = numeric_index_->range(field, sortable_val, sortable_val);
                } else {
                    if(!val.is_string() && !val.is_number_integer() && !val.is_boolean()) {
                        throw std::runtime_error("$eq value must be string, integer or boolean");
                    }
                    std::string str_val;
                    if(val.is_string()) {
                        str_val = val.get<std::string>();
                    } else if(val.is_boolean()) {
                        str_val = val.get<bool>() ? "true" : "false";
                    } else {
                        // Integers for non-numeric fields are treated as strings without padding
                        str_val = std::to_string(val.get<int>());
                    }
                    std::string key = format_filter_key(field, str_val);
                    or_result = bitmap_index_->get_bitmap_by_key(key);
                }
            } else if(op == "$in") {
                if(!val.is_array()) {
                    throw std::runtime_error("$in must be array");
                }
                if(val.empty()) {
                    LOG_DEBUG("Empty $in array for field: " << field);
                } else {
                    for(const auto& v : val) {
                        if(type == FieldType::Number) {
                            uint32_t sortable_val;
                            if(v.is_number_integer()) {
                                sortable_val = ndd::numeric::int_to_sortable(v.get<int>());
                            } else if(v.is_number()) {
                                sortable_val = ndd::numeric::float_to_sortable(v.get<float>());
                            } else {
                                throw std::runtime_error(
                                        "$in value for numeric field must be a number");
                            }
                            or_result |= numeric_index_->range(field, sortable_val, sortable_val);
                        } else {
                            if(!v.is_string() && !v.is_number_integer() && !v.is_boolean()) {
                                throw std::runtime_error(
                                        "$in values must be string, integer or boolean");
                            }
                            std::string str_val;
                            if(v.is_string()) {
                                str_val = v.get<std::string>();
                            } else if(v.is_boolean()) {
                                str_val = v.get<bool>() ? "true" : "false";
                            } else {
                                str_val = std::to_string(v.get<int>());
                            }
                            if(!str_val.empty()) {
                                std::string key = format_filter_key(field, str_val);
                                or_result |= bitmap_index_->get_bitmap_by_key(key);
                            }
                        }
                    }
                }
            } else if(op == "$range") {
                if(!val.is_array() || val.size() != 2) {
                    throw std::runtime_error(
                            "$range must be [start, end] array with exactly 2 elements");
                }

                if(type == FieldType::Number) {
                    uint32_t start_val, end_val;

                    if(val[0].is_number_integer()) {
                        start_val = ndd::numeric::int_to_sortable(val[0].get<int>());
                    } else if(val[0].is_number()) {
                        start_val = ndd::numeric::float_to_sortable(val[0].get<float>());
                    } else {
                        throw std::runtime_error("Range start must be a number");
                    }

                    if(val[1].is_number_integer()) {
                        end_val = ndd::numeric::int_to_sortable(val[1].get<int>());
                    } else if(val[1].is_number()) {
                        end_val = ndd::numeric::float_to_sortable(val[1].get<float>());
                    } else {
                        throw std::runtime_error("Range end must be a number");
                    }

                    if(start_val > end_val) {
                        throw std::runtime_error("Invalid range: start > end");
                    }

                    or_result = numeric_index_->range(field, start_val, end_val);
                } else {
                    throw std::runtime_error(
                            "$range operator is only supported for numeric fields");
                }
            } else {
                throw std::runtime_error("Unsupported operator: " + op);
            }

            // Combine with final result
            if(first) {
                final_result = std::move(or_result);
                first = false;
            } else {
                final_result &= or_result;
            }
        }

        return final_result;
    }

    // Get IDs matching the filter using the provided JSON filter array
    std::vector<ndd::idInt> getIdsMatchingFilter(const nlohmann::json& filter_array) const {
        auto result = computeFilterBitmap(filter_array);
        std::vector<ndd::idInt> ids;
        ids.reserve(result.cardinality());
        result.iterate(
                [](ndd::idInt val, void* ptr) {
                    static_cast<std::vector<ndd::idInt>*>(ptr)->push_back(val);
                    return true;
                },
                &ids);
        return ids;
    }

    // Count the number of IDs matching the filter using the provided JSON filter array
    size_t countIdsMatchingFilter(const nlohmann::json& filter_array) const {
        return computeFilterBitmap(filter_array).cardinality();
    }

    void add_to_filter(const std::string& field, const std::string& value, ndd::idInt numeric_id) {
        bitmap_index_->add(field, value, numeric_id);
    }

    // Batch add operation for filters
    void add_to_filter_batch(const std::string& filter_key,
                             const std::vector<ndd::idInt>& numeric_ids) {
        if(numeric_ids.empty()) {
            return;
        }
        bitmap_index_->add_batch_by_key(filter_key, numeric_ids);
    }

    // Optimized version to process filter JSON in batch
    void add_filters_from_json_batch(
            const std::vector<std::pair<ndd::idInt, std::string>>& id_filter_pairs) {
        if(id_filter_pairs.empty()) {
            return;
        }

        // Create a map to collect IDs for each filter
        std::unordered_map<std::string, std::vector<ndd::idInt>> filter_to_ids;

        // Group IDs by filter
        for(const auto& [numeric_id, filter_json] : id_filter_pairs) {
            try {
                auto j = nlohmann::json::parse(filter_json);
                for(const auto& [field, value] : j.items()) {
                    FieldType type = FieldType::Unknown;
                    if(value.is_boolean()) {
                        type = FieldType::Bool;
                    } else if(value.is_number()) {
                        type = FieldType::Number;  // Unified check
                    } else if(value.is_string()) {
                        type = FieldType::String;
                    }

                    if(type == FieldType::Unknown) {
                        LOG_DEBUG("Unsupported filter type for field '" << field << "'");
                        continue;
                    }

                    if(!register_field_type(field, type)) {
                        LOG_ERROR("Type mismatch for field '" << field << "'");
                        continue;
                    }

                    if(value.is_string()) {
                        std::string filter_key = format_filter_key(field, value.get<std::string>());
                        filter_to_ids[filter_key].push_back(numeric_id);
                    } else if(value.is_number()) {
                        // Use Numeric Index for numbers
                        uint32_t sortable_val;
                        if(value.is_number_integer()) {
                            sortable_val = ndd::numeric::int_to_sortable(value.get<int>());
                        } else {
                            sortable_val = ndd::numeric::float_to_sortable(value.get<float>());
                        }
                        numeric_index_->put(field, numeric_id, sortable_val);
                    } else if(value.is_boolean()) {
                        std::string filter_key =
                                format_filter_key(field, value.get<bool>() ? "true" : "false");
                        filter_to_ids[filter_key].push_back(numeric_id);
                    } else {
                        // Optional: catch bad types (bool, float, object, array, etc.)
                        std::cerr << "Unsupported filter type for field '" << field
                                  << "' in filter: " << value.dump() << std::endl;
                    }
                }
            } catch(const std::exception& e) {
                std::cerr << "Error parsing filter JSON: " << e.what() << std::endl;
            }
        }

        // Process each filter with its batch of IDs
        for(const auto& [filter_key, ids] : filter_to_ids) {
            add_to_filter_batch(filter_key, ids);
        }
    }

    void
    remove_from_filter(const std::string& field, const std::string& value, ndd::idInt numeric_id) {
        bitmap_index_->remove(field, value, numeric_id);
    }

    bool contains(const std::string& field, const std::string& value, ndd::idInt numeric_id) const {
        return bitmap_index_->contains(field, value, numeric_id);
    }

    void add_filters_from_json(ndd::idInt numeric_id, const std::string& filter_json) {
        try {
            auto j = nlohmann::json::parse(filter_json);
            for(const auto& [field, value] : j.items()) {
                FieldType type = FieldType::Unknown;
                if(value.is_boolean()) {
                    type = FieldType::Bool;
                } else if(value.is_number()) {
                    type = FieldType::Number;
                } else if(value.is_string()) {
                    type = FieldType::String;
                }

                if(type == FieldType::Unknown) {
                    LOG_DEBUG("Unsupported filter type for field '" << field << "'");
                    continue;
                }

                if(!register_field_type(field, type)) {
                    LOG_ERROR("Type mismatch for field '" << field << "'");
                    continue;
                }

                if(value.is_string()) {
                    add_to_filter(field, value.get<std::string>(), numeric_id);
                } else if(value.is_number()) {
                    uint32_t sortable_val;
                    if(value.is_number_integer()) {
                        sortable_val = ndd::numeric::int_to_sortable(value.get<int>());
                    } else {
                        sortable_val = ndd::numeric::float_to_sortable(value.get<float>());
                    }
                    numeric_index_->put(field, numeric_id, sortable_val);
                } else if(value.is_boolean()) {
                    add_to_filter(field, value.get<bool>() ? "true" : "false", numeric_id);
                }
            }
        } catch(const std::exception& e) {
            std::cerr << "Error adding filters: " << e.what() << std::endl;
        }
    }

    void remove_filters_from_json(ndd::idInt numeric_id, const std::string& filter_json) {
        try {
            auto j = nlohmann::json::parse(filter_json);
            for(const auto& [field, value] : j.items()) {
                if(value.is_string()) {
                    remove_from_filter(field, value.get<std::string>(), numeric_id);
                } else if(value.is_number()) {
                    // Remove from Numeric Index
                    numeric_index_->remove(field, numeric_id);
                } else if(value.is_boolean()) {
                    remove_from_filter(field, value.get<bool>() ? "true" : "false", numeric_id);
                }
            }
        } catch(const std::exception& e) {
            std::cerr << "Error removing filters: " << e.what() << std::endl;
        }
    }

    // Combine multiple filters using AND operation
    ndd::RoaringBitmap
    combine_filters_and(const std::vector<std::pair<std::string, std::string>>& filters) const {
        ndd::RoaringBitmap result;
        bool first = true;
        for(const auto& [field, value] : filters) {
            if(first) {
                result = bitmap_index_->get_bitmap(field, value);
                first = false;
            } else {
                result &= bitmap_index_->get_bitmap(field, value);
            }
        }
        return result;
    }

    // Combine multiple filters using OR operation
    ndd::RoaringBitmap
    combine_filters_or(const std::vector<std::pair<std::string, std::string>>& filters) const {
        ndd::RoaringBitmap result;
        for(const auto& [field, value] : filters) {
            result |= bitmap_index_->get_bitmap(field, value);
        }
        return result;
    }

    // Check if ID satisfies a numeric condition using Forward Index
    bool check_numeric(const std::string& field,
                       ndd::idInt id,
                       const std::string& op,
                       const nlohmann::json& val) const {
        if(op == "$eq") {
            uint32_t sortable_val;
            if(val.is_number_integer()) {
                sortable_val = ndd::numeric::int_to_sortable(val.get<int>());
            } else if(val.is_number()) {
                sortable_val = ndd::numeric::float_to_sortable(val.get<float>());
            } else {
                return false;
            }
            return numeric_index_->check_range(field, id, sortable_val, sortable_val);
        } else if(op == "$in") {
            if(!val.is_array()) {
                return false;
            }
            for(const auto& v : val) {
                uint32_t sortable_val;
                if(v.is_number_integer()) {
                    sortable_val = ndd::numeric::int_to_sortable(v.get<int>());
                } else if(v.is_number()) {
                    sortable_val = ndd::numeric::float_to_sortable(v.get<float>());
                } else {
                    continue;
                }

                if(numeric_index_->check_range(field, id, sortable_val, sortable_val)) {
                    return true;
                }
            }
            return false;
        } else if(op == "$range") {
            if(!val.is_array() || val.size() != 2) {
                return false;
            }
            uint32_t start_val, end_val;

            if(val[0].is_number_integer()) {
                start_val = ndd::numeric::int_to_sortable(val[0].get<int>());
            } else if(val[0].is_number()) {
                start_val = ndd::numeric::float_to_sortable(val[0].get<float>());
            } else {
                return false;
            }

            if(val[1].is_number_integer()) {
                end_val = ndd::numeric::int_to_sortable(val[1].get<int>());
            } else if(val[1].is_number()) {
                end_val = ndd::numeric::float_to_sortable(val[1].get<float>());
            } else {
                return false;
            }

            return numeric_index_->check_range(field, id, start_val, end_val);
        }
        return false;
    }
};