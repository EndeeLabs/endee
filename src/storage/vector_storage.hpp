#pragma once

#include "mdbx/mdbx.h"
#include "log.hpp"
#include "../quant/dispatch.hpp"
#include "../filter/filter.hpp"
#include "../core/types.hpp"
#include "json/nlohmann_json.hpp"
#include "msgpack_ndd.hpp"
#include "quant_vector.hpp"
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <filesystem>

// Handles vector storage
class VectorStore {
private:
    MDBX_env* env_;
    MDBX_dbi dbi_;
    std::string path_;
    size_t vector_dim_;
    ndd::quant::QuantizationLevel quant_level_;
    size_t bytes_per_vector_;

    void init_environment() {
        int rc = mdbx_env_create(&env_);
        if(rc != MDBX_SUCCESS) {
            throw std::runtime_error("Failed to create LMDB env");
        }

        // Set geometry for auto-grow: initial=8GB, growth=1GB, max=128GB
        rc = mdbx_env_set_geometry(env_,
                                   -1,  // lower size bound (use default)
                                   1ULL << settings::VECTOR_MAP_SIZE_BITS,      // current/now size
                                   1ULL << settings::VECTOR_MAP_SIZE_MAX_BITS,  // upper size bound
                                   1ULL << settings::VECTOR_MAP_SIZE_BITS,      // growth step
                                   -1,   // shrink threshold (use default)
                                   -1);  // pagesize (use default)
        if(rc != MDBX_SUCCESS) {
            throw std::runtime_error("Failed to set geometry");
        }

        rc = mdbx_env_open(
                env_, path_.c_str(), MDBX_WRITEMAP | MDBX_MAPASYNC | MDBX_NORDAHEAD, 0664);
        if(rc != MDBX_SUCCESS) {
            throw std::runtime_error("Failed to open environment");
        }

        MDBX_txn* txn;
        rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_READWRITE, &txn);
        if(rc != MDBX_SUCCESS) {
            throw std::runtime_error("Failed to begin transaction");
        }

        rc = mdbx_dbi_open(txn, nullptr, MDBX_CREATE | MDBX_INTEGERKEY, &dbi_);
        if(rc != MDBX_SUCCESS) {
            mdbx_txn_abort(txn);
            throw std::runtime_error("Failed to open database");
        }

        rc = mdbx_txn_commit(txn);
        if(rc != MDBX_SUCCESS) {
            throw std::runtime_error("Failed to commit transaction: "
                                     + std::string(mdbx_strerror(rc)));
        }
    }

public:
    VectorStore(const std::string& path,
                size_t vector_dim,
                ndd::quant::QuantizationLevel quant_level) :
        path_(path),
        vector_dim_(vector_dim),
        quant_level_(quant_level) {
        bytes_per_vector_ =
                ndd::quant::get_quantizer_dispatch(quant_level_).get_storage_size(vector_dim);
        std::filesystem::create_directories(path);
        init_environment();
    }

    ~VectorStore() {
        mdbx_dbi_close(env_, dbi_);
        mdbx_env_close(env_);
    }
    // Nested Cursor struct

    struct Cursor {
        MDBX_txn* txn = nullptr;
        MDBX_cursor* cursor = nullptr;
        bool done = false;

        Cursor(MDBX_env* env, MDBX_dbi dbi) {
            if(mdbx_txn_begin(env, nullptr, MDBX_TXN_RDONLY, &txn) != MDBX_SUCCESS) {
                throw std::runtime_error("LMDB txn begin failed");
            }

            if(mdbx_cursor_open(txn, dbi, &cursor) != MDBX_SUCCESS) {
                throw std::runtime_error("LMDB cursor open failed");
            }
        }

        // prevent copying
        Cursor(const Cursor&) = delete;
        Cursor& operator=(const Cursor&) = delete;

        bool hasNext() { return !done; }

        std::pair<ndd::idInt, std::vector<uint8_t>> next() {
            MDBX_val key, val;
            int rc = mdbx_cursor_get(cursor, &key, &val, MDBX_NEXT);
            if(rc != MDBX_SUCCESS) {
                done = true;
                return {};
            }

            if(key.iov_len != sizeof(ndd::idInt)) {
                printf("Invalid key size: %zu, expected: %zu\n", key.iov_len, sizeof(ndd::idInt));
                throw std::runtime_error("Invalid key size in LMDB entry");
            }

            ndd::idInt label;
            std::memcpy(&label, key.iov_base, sizeof(label));

            std::vector<uint8_t> vec((uint8_t*)val.iov_base, (uint8_t*)val.iov_base + val.iov_len);
            return {label, std::move(vec)};
        }

        ~Cursor() {
            if(cursor) {
                mdbx_cursor_close(cursor);
            }
            if(txn) {
                mdbx_txn_abort(txn);
            }
        }
    };

    Cursor getCursor() { return Cursor(env_, dbi_); }

    void store_vector_bytes(ndd::idInt id, const std::vector<uint8_t>& vec) {
        store_vectors_batch({{id, vec}});
    }

    std::vector<uint8_t> get_vector_bytes(ndd::idInt numeric_id) const {
        MDBX_txn* txn;
        int rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_RDONLY, &txn);
        if(rc != MDBX_SUCCESS) {
            throw std::runtime_error("Failed to begin transaction");
        }

        try {
            MDBX_val key{const_cast<ndd::idInt*>(&numeric_id), sizeof(ndd::idInt)};
            MDBX_val data;

            rc = mdbx_get(txn, dbi_, &key, &data);
            if(rc == MDBX_NOTFOUND) {
                mdbx_txn_abort(txn);
                return std::vector<uint8_t>();
            }

            std::vector<uint8_t> result(static_cast<uint8_t*>(data.iov_base),
                                        static_cast<uint8_t*>(data.iov_base) + data.iov_len);

            mdbx_txn_abort(txn);
            return result;
        } catch(...) {
            mdbx_txn_abort(txn);
            throw;
        }
    }

    bool get_vector_bytes(ndd::idInt numeric_id, uint8_t* buffer) const {
        MDBX_txn* txn;
        int rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_RDONLY, &txn);
        if(rc != MDBX_SUCCESS) {
            return false;
        }

        try {
            MDBX_val key{const_cast<ndd::idInt*>(&numeric_id), sizeof(ndd::idInt)};
            MDBX_val data;

            rc = mdbx_get(txn, dbi_, &key, &data);
            if(rc == MDBX_NOTFOUND) {
                mdbx_txn_abort(txn);
                return false;
            }

            if(data.iov_len != bytes_per_vector_) {
                mdbx_txn_abort(txn);
                // Warning: data size mismatch.
                // We could log this but for now just fail or copy what is there if smaller?
                // Safer to fail or copy min to avoid overflow if buffer is assumed to be
                // bytes_per_vector_
                return false;
            }

            std::memcpy(buffer, data.iov_base, data.iov_len);

            mdbx_txn_abort(txn);
            return true;
        } catch(...) {
            mdbx_txn_abort(txn);
            return false;
        }
    }

    // Batch operations with raw bytes
    void
    store_vectors_batch(const std::vector<std::pair<ndd::idInt, std::vector<uint8_t>>>& batch) {
        if(batch.empty()) {
            return;
        }

        auto try_commit = [&](MDBX_txn* txn) {
            int rc = mdbx_txn_commit(txn);
            if(rc != MDBX_SUCCESS) {
                throw std::runtime_error("Failed to commit transaction: "
                                         + std::string(mdbx_strerror(rc)));
            }
        };

        auto write_batch = [&](MDBX_txn* txn) -> int {
            for(const auto& [numeric_id, vector_bytes] : batch) {
                if(vector_bytes.size() != bytes_per_vector_) {
                    throw std::runtime_error("Vector byte size mismatch");
                }

                MDBX_val key{const_cast<ndd::idInt*>(&numeric_id), sizeof(ndd::idInt)};
                MDBX_val data{const_cast<uint8_t*>(vector_bytes.data()), vector_bytes.size()};

                int rc = mdbx_put(txn, dbi_, &key, &data, MDBX_UPSERT);
            }
            return MDBX_SUCCESS;
        };

        MDBX_txn* txn;
        int rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_READWRITE, &txn);
        if(rc != MDBX_SUCCESS) {
            throw std::runtime_error("Failed to begin transaction");
        }

        rc = write_batch(txn);
        // MDBX auto-grows, no manual resize needed
        if(rc != MDBX_SUCCESS) {
            mdbx_txn_abort(txn);
            throw std::runtime_error("Failed to store vector");
        }

        try_commit(txn);
    }

    std::vector<std::pair<ndd::idInt, std::vector<uint8_t>>>
    get_vectors_batch(const std::vector<ndd::idInt>& numeric_ids) const {
        std::vector<std::pair<ndd::idInt, std::vector<uint8_t>>> result;
        if(numeric_ids.empty()) {
            return result;
        }

        result.reserve(numeric_ids.size());

        MDBX_txn* txn;
        int rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_RDONLY, &txn);
        if(rc != MDBX_SUCCESS) {
            throw std::runtime_error("Failed to begin transaction");
        }

        try {
            for(const auto& numeric_id : numeric_ids) {
                MDBX_val key{const_cast<ndd::idInt*>(&numeric_id), sizeof(ndd::idInt)};
                MDBX_val data;

                rc = mdbx_get(txn, dbi_, &key, &data);
                if(rc == MDBX_SUCCESS) {  // Found the vector
                    std::vector<uint8_t> bytes(static_cast<uint8_t*>(data.iov_base),
                                               static_cast<uint8_t*>(data.iov_base) + data.iov_len);
                    result.emplace_back(numeric_id, std::move(bytes));
                }
            }

            mdbx_txn_abort(txn);
            return result;
        } catch(...) {
            mdbx_txn_abort(txn);
            throw;
        }
    }

    void remove(ndd::idInt numeric_id) {
        MDBX_txn* txn;
        int rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_READWRITE, &txn);
        if(rc != MDBX_SUCCESS) {
            throw std::runtime_error("Failed to begin transaction");
        }

        try {
            MDBX_val key{const_cast<ndd::idInt*>(&numeric_id), sizeof(ndd::idInt)};

            rc = mdbx_del(txn, dbi_, &key, nullptr);
            if(rc != MDBX_SUCCESS && rc != MDBX_NOTFOUND) {
                throw std::runtime_error("Failed to delete vector data");
            }

            rc = mdbx_txn_commit(txn);
            if(rc != MDBX_SUCCESS) {
                throw std::runtime_error("Failed to commit vector deletion");
            }
        } catch(...) {
            mdbx_txn_abort(txn);
            throw;
        }
    }

    ndd::quant::QuantizationLevel getQuantLevel() const { return quant_level_; }
    size_t dimension() const { return vector_dim_; }
    size_t get_vector_size() const { return bytes_per_vector_; }

    // Allow access to LMDB environment for other operations
    MDBX_env* get_env() const { return env_; }
    MDBX_dbi get_dbi() const { return dbi_; }
};

// Handles meta storage
class MetaStore {
private:
    MDBX_env* env_;
    MDBX_dbi dbi_;
    std::string path_;

    void init_environment() {
        int rc = mdbx_env_create(&env_);
        if(rc != MDBX_SUCCESS) {
            throw std::runtime_error("Failed to create LMDB env");
        }

        // Set geometry for auto-grow
        rc = mdbx_env_set_geometry(
                env_,
                -1,                                            // lower size bound (use default)
                1ULL << settings::METADATA_MAP_SIZE_BITS,      // current/now size
                1ULL << settings::METADATA_MAP_SIZE_MAX_BITS,  // upper size bound
                1ULL << settings::METADATA_MAP_SIZE_BITS,      // growth step
                -1,                                            // shrink threshold (use default)
                -1);                                           // pagesize (use default)
        if(rc != MDBX_SUCCESS) {
            throw std::runtime_error("Failed to set geometry");
        }

        rc = mdbx_env_open(env_,
                           path_.c_str(),
                           MDBX_NOSUBDIR | MDBX_WRITEMAP | MDBX_MAPASYNC | MDBX_NORDAHEAD,
                           0664);
        if(rc != MDBX_SUCCESS) {
            throw std::runtime_error("Failed to open environment");
        }

        MDBX_txn* txn;
        rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_READWRITE, &txn);
        if(rc != MDBX_SUCCESS) {
            throw std::runtime_error("Failed to begin transaction");
        }

        rc = mdbx_dbi_open(txn, nullptr, MDBX_CREATE | MDBX_INTEGERKEY, &dbi_);
        if(rc != MDBX_SUCCESS) {
            mdbx_txn_abort(txn);
            throw std::runtime_error("Failed to open database");
        }

        rc = mdbx_txn_commit(txn);
        if(rc != MDBX_SUCCESS) {
            throw std::runtime_error("Failed to commit transaction: "
                                     + std::string(mdbx_strerror(rc)));
        }
    }

public:
    MetaStore(const std::string& path) :
        path_(path) {
        std::filesystem::create_directories(path);
        init_environment();
    }

    ~MetaStore() {
        mdbx_dbi_close(env_, dbi_);
        mdbx_env_close(env_);
    }

    void store_meta_batch(const std::vector<std::pair<ndd::idInt, ndd::VectorMeta>>& batch) {
        if(batch.empty()) {
            return;
        }

        auto try_commit = [&](MDBX_txn* txn) {
            int rc = mdbx_txn_commit(txn);
            if(rc != MDBX_SUCCESS) {
                throw std::runtime_error("Failed to commit transaction: "
                                         + std::string(mdbx_strerror(rc)));
            }
        };

        auto write_batch = [&](MDBX_txn* txn) {
            for(const auto& [numeric_id, meta] : batch) {
                msgpack::sbuffer sbuf;
                msgpack::pack(sbuf, meta);

                MDBX_val key{const_cast<ndd::idInt*>(&numeric_id), sizeof(ndd::idInt)};
                MDBX_val data{const_cast<char*>(sbuf.data()), sbuf.size()};

                int rc = mdbx_put(txn, dbi_, &key, &data, MDBX_UPSERT);
                if(rc != MDBX_SUCCESS) {
                }
            }
            return MDBX_SUCCESS;
        };

        MDBX_txn* txn;
        int rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_READWRITE, &txn);
        if(rc != MDBX_SUCCESS) {
            throw std::runtime_error("Failed to begin transaction");
        }

        rc = write_batch(txn);
        // MDBX auto-grows, no manual resize needed
        if(rc != MDBX_SUCCESS) {
            mdbx_txn_abort(txn);
            throw std::runtime_error("Failed to store meta");
        }

        try_commit(txn);
    }

    void store_meta(ndd::idInt id, const ndd::VectorMeta& meta) { store_meta_batch({{id, meta}}); }
    ndd::VectorMeta get_meta(ndd::idInt numeric_id) const {
        MDBX_txn* txn;
        int rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_RDONLY, &txn);
        if(rc != MDBX_SUCCESS) {
            throw std::runtime_error("Failed to begin transaction");
        }

        try {
            MDBX_val key{const_cast<ndd::idInt*>(&numeric_id), sizeof(ndd::idInt)};
            MDBX_val data;

            rc = mdbx_get(txn, dbi_, &key, &data);
            if(rc == MDBX_NOTFOUND) {
                mdbx_txn_abort(txn);
                throw std::runtime_error("Meta not found");
            }
            auto oh = msgpack::unpack(reinterpret_cast<const char*>(data.iov_base), data.iov_len);
            auto meta = oh.get().as<ndd::VectorMeta>();
            mdbx_txn_abort(txn);
            return meta;
        } catch(...) {
            mdbx_txn_abort(txn);
            throw;
        }
    }

    void remove(ndd::idInt numeric_id) {
        MDBX_txn* txn;
        int rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_READWRITE, &txn);
        if(rc != MDBX_SUCCESS) {
            throw std::runtime_error("Failed to begin transaction");
        }

        try {
            MDBX_val key{const_cast<ndd::idInt*>(&numeric_id), sizeof(ndd::idInt)};

            rc = mdbx_del(txn, dbi_, &key, nullptr);
            if(rc != MDBX_SUCCESS && rc != MDBX_NOTFOUND) {
                throw std::runtime_error("Failed to delete metadata");
            }

            rc = mdbx_txn_commit(txn);
            if(rc != MDBX_SUCCESS) {
                throw std::runtime_error("Failed to commit metadata deletion");
            }
        } catch(...) {
            mdbx_txn_abort(txn);
            throw;
        }
    }
};

// Main storage interface combining vector and meta stores
class VectorStorage {
private:
    std::unique_ptr<VectorStore> vector_store_;
    std::unique_ptr<MetaStore> meta_store_;

public:
    std::unique_ptr<Filter> filter_store_;

    VectorStorage(const std::string& base_path,
                  size_t vector_dim,
                  ndd::quant::QuantizationLevel quant_level) {
        vector_store_ =
                std::make_unique<VectorStore>(base_path + "/vectors", vector_dim, quant_level);
        meta_store_ = std::make_unique<MetaStore>(base_path + "/meta");
        filter_store_ = std::make_unique<Filter>(base_path + "/filters");
    }
    VectorStore::Cursor getCursor() { return vector_store_->getCursor(); }
    // Get numeric ids of matching filters
    std::vector<ndd::idInt> getIdsMatchingFilters(
            const std::vector<std::pair<std::string, std::string>>& filter_pairs) const {
        auto bitmap = filter_store_->combine_filters_and(filter_pairs);
        std::vector<ndd::idInt> numeric_ids;
        bitmap.iterate(
                [](ndd::idInt value, void* ptr) -> bool {
                    auto* ids = static_cast<std::vector<ndd::idInt>*>(ptr);
                    ids->push_back(value);
                    return true;
                },
                &numeric_ids);
        return numeric_ids;
    }

    bool matches_filter(ndd::idInt numeric_id,
                        const ndd::VectorMeta& meta,
                        const nlohmann::json& filter_query) {
        if(filter_query.empty()) {
            return true;
        }

        // 1. Fast Pass: Check Numeric Filters using Index
        bool has_non_numeric = false;

        for(const auto& condition : filter_query) {
            if(!condition.is_object() || condition.size() != 1) {
                continue;
            }
            const auto& field = condition.begin().key();
            const auto& expr = condition.begin().value();
            if(!expr.is_object() || expr.size() != 1) {
                continue;
            }

            const std::string op = expr.begin().key();
            const auto& val = expr.begin().value();

            bool is_numeric_query = false;
            if(op == "$range") {
                is_numeric_query = true;
            } else if(op == "$eq" && (val.is_number())) {
                is_numeric_query = true;
            } else if(op == "$in" && val.is_array() && !val.empty() && val[0].is_number()) {
                is_numeric_query = true;
            }

            if(is_numeric_query) {
                if(!filter_store_->check_numeric(field, numeric_id, op, val)) {
                    return false;
                }
            } else {
                has_non_numeric = true;
            }
        }

        if(!has_non_numeric) {
            return true;
        }

        try {
            // Parse the metadata associated with the vector
            nlohmann::json meta_filter = nlohmann::json::parse(meta.filter);

            // Each filter clause is ANDed
            for(const auto& condition : filter_query) {
                if(!condition.is_object() || condition.size() != 1) {
                    continue;  // Skip malformed conditions
                }

                const auto& field = condition.begin().key();
                const auto& expr = condition.begin().value();

                if(!expr.is_object() || expr.size() != 1) {
                    continue;
                }

                const std::string op = expr.begin().key();
                const auto& val = expr.begin().value();

                // Skip numeric queries as they are already checked
                bool is_numeric_query = false;
                if(op == "$range") {
                    is_numeric_query = true;
                } else if(op == "$eq" && (val.is_number())) {
                    is_numeric_query = true;
                } else if(op == "$in" && val.is_array() && !val.empty() && val[0].is_number()) {
                    is_numeric_query = true;
                }

                if(is_numeric_query) {
                    continue;
                }

                // If field is not present in the vector's metadata
                if(!meta_filter.contains(field)) {
                    return false;
                }

                const auto& actual_value = meta_filter[field];

                if(op == "$eq") {
                    if(actual_value != val) {
                        return false;
                    }
                } else if(op == "$in") {
                    if(!val.is_array()
                       || std::find(val.begin(), val.end(), actual_value) == val.end()) {
                        return false;
                    }
                } else {
                    continue;
                }
            }

            return true;

        } catch(const std::exception& e) {
            // std::cerr << "Error matching filter: " << e.what() << std::endl;
            return false;
        }
    }

    // Optimized batch operation using pre-quantized QuantVectorObject
    // This avoids double quantization by using already quantized data
    void store_vectors_batch(const std::vector<std::pair<ndd::idInt, QuantVectorObject>>& vectors) {
        if(vectors.empty()) {
            return;
        }

        // Prepare vector and meta batches
        std::vector<std::pair<ndd::idInt, std::vector<uint8_t>>> vector_batch;
        std::vector<std::pair<ndd::idInt, ndd::VectorMeta>> meta_batch;
        std::vector<std::pair<ndd::idInt, std::string>> filter_batch;

        vector_batch.reserve(vectors.size());
        meta_batch.reserve(vectors.size());
        filter_batch.reserve(vectors.size());

        for(const auto& [numeric_id, quant_obj] : vectors) {
            // Use pre-quantized data directly - no conversion needed!
            std::vector<uint8_t> vector_bytes = quant_obj.quant_vector;

            // Create metadata from QuantVectorObject
            ndd::VectorMeta meta;
            meta.id = quant_obj.id;
            meta.filter = quant_obj.filter;
            meta.meta = quant_obj.meta;
            meta.norm = quant_obj.norm;

            vector_batch.emplace_back(numeric_id, std::move(vector_bytes));
            meta_batch.emplace_back(numeric_id, std::move(meta));

            // Collect filter data for batch processing
            if(!quant_obj.filter.empty()) {
                filter_batch.emplace_back(numeric_id, quant_obj.filter);
            }
        }

        // Store vectors and metadata in single transactions
        vector_store_->store_vectors_batch(vector_batch);
        meta_store_->store_meta_batch(meta_batch);

        // Process filter data in batch if any
        if(!filter_batch.empty()) {
            filter_store_->add_filters_from_json_batch(filter_batch);
        }
    }

    std::vector<uint8_t> get_vector(ndd::idInt numeric_id) const {
        return vector_store_->get_vector_bytes(numeric_id);
    }

    bool get_vector(ndd::idInt numeric_id, uint8_t* buffer) const {
        return vector_store_->get_vector_bytes(numeric_id, buffer);
    }

    std::vector<std::pair<ndd::idInt, std::vector<uint8_t>>>
    get_vectors_batch(const std::vector<ndd::idInt>& numeric_ids) const {
        return vector_store_->get_vectors_batch(numeric_ids);
    }
    ndd::VectorMeta get_meta(ndd::idInt numeric_id) const {
        return meta_store_->get_meta(numeric_id);
    }

    // NOT used anymore. Deletes filter, meta and vector data.
    void deletePoint(ndd::idInt numeric_id) {
        try {
            // Get metadata first to get filter info
            auto meta = meta_store_->get_meta(numeric_id);

            // Remove filter entries if they exist
            if(!meta.filter.empty()) {
                filter_store_->remove_filters_from_json(numeric_id, meta.filter);
            }
            // Try to remove both vector and meta data
            vector_store_->remove(numeric_id);
            meta_store_->remove(numeric_id);
        } catch(const std::exception& e) {
            throw std::runtime_error(std::string("Failed to remove vector and metadata: ")
                                     + e.what());
        }
    }
    // Deletes filter only.
    void deleteFilter(ndd::idInt numeric_id, std::string filter) {
        filter_store_->remove_filters_from_json(numeric_id, filter);
    }

    // Update filter for a vector
    void updateFilter(ndd::idInt numeric_id, const std::string& new_filter_json) {
        // Get existing meta
        auto meta = meta_store_->get_meta(numeric_id);

        // Remove old filters
        if(!meta.filter.empty()) {
            filter_store_->remove_filters_from_json(numeric_id, meta.filter);
        }

        // Update meta
        meta.filter = new_filter_json;
        meta_store_->store_meta(numeric_id, meta);

        // Add new filters
        if(!new_filter_json.empty()) {
            filter_store_->add_filters_from_json(numeric_id, new_filter_json);
        }
    }

    ndd::quant::QuantizationLevel getQuantLevel() const { return vector_store_->getQuantLevel(); }
    size_t dimension() const { return vector_store_->dimension(); }
    size_t get_vector_size() const { return vector_store_->get_vector_size(); }
};