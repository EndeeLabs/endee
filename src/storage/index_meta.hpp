#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <optional>
#include <filesystem>
#include <mutex>
#include <memory>
#include <unordered_map>
#include "settings.hpp"
#include "mdbx/mdbx.h"
#include "quant/common.hpp"

struct IndexMetadata {
    std::string name;  // Just the index name, not the full path
    size_t dimension;
    size_t sparse_dim = 0;  // Added sparse dimension
    std::string space_type_str;
    ndd::quant::QuantizationLevel quant_level =
            ndd::quant::QuantizationLevel::INT8;  // Quantization level (8, 15, 16, 32)
    int32_t checksum;
    size_t total_elements;
    size_t M;
    size_t ef_con;
    std::chrono::system_clock::time_point created_at;

    nlohmann::json to_json() const {
        return {{"name", name},
                {"dimension", dimension},
                {"sparse_dim", sparse_dim},
                {"space_type_str", space_type_str},
                {"quant_level", static_cast<uint8_t>(quant_level)},
                {"checksum", checksum},
                {"total_elements", total_elements},
                {"M", M},
                {"ef_con", ef_con},
                {"created_at", std::chrono::system_clock::to_time_t(created_at)}};
    }

    static IndexMetadata from_json(const nlohmann::json& j) {
        IndexMetadata meta;
        meta.name = j["name"].get<std::string>();
        meta.dimension = j["dimension"].get<size_t>();
        if(j.contains("sparse_dim")) {
            meta.sparse_dim = j["sparse_dim"].get<size_t>();
        } else {
            meta.sparse_dim = 0;
        }
        meta.space_type_str = j["space_type_str"].get<std::string>();
        meta.quant_level =
                static_cast<ndd::quant::QuantizationLevel>(j["quant_level"].get<uint8_t>());
        meta.checksum = j["checksum"].get<int32_t>();
        meta.total_elements = j["total_elements"].get<size_t>();
        meta.M = j["M"].get<size_t>();
        meta.ef_con = j["ef_con"].get<size_t>();
        meta.created_at = std::chrono::system_clock::from_time_t(j["created_at"].get<time_t>());
        return meta;
    }
};

class MetadataManager {
private:
    MDBX_env* metadata_env_;
    MDBX_dbi metadata_dbi_;
    std::string metadata_dir_;

public:
    MetadataManager(const std::string& base_dir) :
        metadata_dir_(base_dir + "/meta") {
        std::filesystem::create_directories(metadata_dir_);
        initEnvironment();
    }

    ~MetadataManager() {
        mdbx_dbi_close(metadata_env_, metadata_dbi_);
        mdbx_env_close(metadata_env_);
    }

    // Store metadata for an index
    bool storeMetadata(const std::string& index_id, const IndexMetadata& metadata) {
        std::string key = index_id;
        MDBX_txn* txn;
        int rc = mdbx_txn_begin(metadata_env_, nullptr, MDBX_TXN_READWRITE, &txn);
        if(rc != 0) {
            std::cerr << "Failed to begin transaction: " << mdbx_strerror(rc) << std::endl;
            return false;
        }

        try {
            auto json_str = metadata.to_json().dump();
            MDBX_val db_key{(void*)key.c_str(), key.size()};
            MDBX_val data{(void*)json_str.c_str(), json_str.size()};

            rc = mdbx_put(txn, metadata_dbi_, &db_key, &data, MDBX_UPSERT);
            if(rc != 0) {
                mdbx_txn_abort(txn);
                std::cerr << "Failed to store metadata: " << mdbx_strerror(rc) << std::endl;
                return false;
            }

            rc = mdbx_txn_commit(txn);
            if(rc != 0) {
                std::cerr << "Failed to commit transaction: " << mdbx_strerror(rc) << std::endl;
                return false;
            }

            return true;
        } catch(const std::exception& e) {
            mdbx_txn_abort(txn);
            std::cerr << "Exception while storing metadata: " << e.what() << std::endl;
            return false;
        }
    }

    // Update element count in index metadata
    bool updateElementCount(const std::string& index_id, size_t count) {
        auto metadata = getMetadata(index_id);
        if(!metadata) {
            std::cerr << "Cannot update element count: metadata not found for " << index_id
                      << std::endl;
            return false;
        }
        metadata->total_elements = count;
        return storeMetadata(index_id, *metadata);
    }

    // Retrieve metadata for an index
    std::optional<IndexMetadata> getMetadata(const std::string& index_id) {
        std::string key = index_id;

        MDBX_txn* txn;
        int rc = mdbx_txn_begin(metadata_env_, nullptr, MDBX_TXN_RDONLY, &txn);
        if(rc != 0) {
            std::cerr << "Failed to begin transaction: " << mdbx_strerror(rc) << std::endl;
            return std::nullopt;
        }

        try {
            MDBX_val db_key{(void*)key.c_str(), key.size()};
            MDBX_val data;

            rc = mdbx_get(txn, metadata_dbi_, &db_key, &data);
            if(rc != 0) {
                mdbx_txn_abort(txn);
                if(rc != MDBX_NOTFOUND) {
                    std::cerr << "Failed to retrieve metadata: " << mdbx_strerror(rc) << std::endl;
                }
                return std::nullopt;
            }

            std::string json_str(static_cast<char*>(data.iov_base), data.iov_len);
            mdbx_txn_abort(txn);

            return IndexMetadata::from_json(nlohmann::json::parse(json_str));
        } catch(const std::exception& e) {
            mdbx_txn_abort(txn);
            std::cerr << "Exception while retrieving metadata: " << e.what() << std::endl;
            return std::nullopt;
        }
    }

    // Delete metadata for an index
    bool deleteMetadata(const std::string& index_id) {
        std::string key = index_id;
        MDBX_txn* txn;
        int rc = mdbx_txn_begin(metadata_env_, nullptr, MDBX_TXN_READWRITE, &txn);
        if(rc != MDBX_SUCCESS) {
            std::cerr << "Failed to begin transaction: " << mdbx_strerror(rc) << std::endl;
            return false;
        }

        try {
            MDBX_val db_key{(void*)key.c_str(), key.size()};

            rc = mdbx_del(txn, metadata_dbi_, &db_key, nullptr);
            if(rc != MDBX_SUCCESS && rc != MDBX_NOTFOUND) {
                mdbx_txn_abort(txn);
                std::cerr << "Failed to delete metadata: " << mdbx_strerror(rc) << std::endl;
                return false;
            }

            rc = mdbx_txn_commit(txn);
            if(rc != MDBX_SUCCESS) {
                std::cerr << "Failed to commit transaction: " << mdbx_strerror(rc) << std::endl;
                return false;
            }

            return true;
        } catch(const std::exception& e) {
            mdbx_txn_abort(txn);
            std::cerr << "Exception while deleting metadata: " << e.what() << std::endl;
            return false;
        }
    }

    // List all indexes metadata
    std::vector<std::pair<std::string, IndexMetadata>> listAllMetadata() {
        std::vector<std::pair<std::string, IndexMetadata>> result;

        MDBX_txn* txn;
        int rc = mdbx_txn_begin(metadata_env_, nullptr, MDBX_TXN_RDONLY, &txn);
        if(rc != 0) {
            std::cerr << "Failed to begin transaction: " << mdbx_strerror(rc) << std::endl;
            return result;
        }

        MDBX_cursor* cursor;
        rc = mdbx_cursor_open(txn, metadata_dbi_, &cursor);
        if(rc != 0) {
            mdbx_txn_abort(txn);
            std::cerr << "Failed to open cursor: " << mdbx_strerror(rc) << std::endl;
            return result;
        }

        MDBX_val key, data;
        while(mdbx_cursor_get(cursor, &key, &data, MDBX_NEXT) == 0) {
            try {
                std::string key_str(static_cast<char*>(key.iov_base), key.iov_len);
                std::string json_str(static_cast<char*>(data.iov_base), data.iov_len);
                result.push_back(
                        {key_str, IndexMetadata::from_json(nlohmann::json::parse(json_str))});
            } catch(const std::exception& e) {
                std::cerr << "Failed to parse metadata: " << e.what() << std::endl;
                // Skip invalid entries
            }
        }

        mdbx_cursor_close(cursor);
        mdbx_txn_abort(txn);
        return result;
    }

    // List indexes with metadata for a specific user
    std::vector<std::pair<std::string, IndexMetadata>>
    listUserIndexes(const std::string& username) {
        std::vector<std::pair<std::string, IndexMetadata>> indexes;
        std::string prefix = username + "/";

        MDBX_txn* txn;
        int rc = mdbx_txn_begin(metadata_env_, nullptr, MDBX_TXN_RDONLY, &txn);
        if(rc != 0) {
            std::cerr << "Failed to begin transaction: " << mdbx_strerror(rc) << std::endl;
            return indexes;
        }

        MDBX_cursor* cursor;
        rc = mdbx_cursor_open(txn, metadata_dbi_, &cursor);
        if(rc != 0) {
            mdbx_txn_abort(txn);
            std::cerr << "Failed to open cursor: " << mdbx_strerror(rc) << std::endl;
            return indexes;
        }

        MDBX_val key, data;
        while(mdbx_cursor_get(cursor, &key, &data, MDBX_NEXT) == 0) {
            std::string key_str(static_cast<char*>(key.iov_base), key.iov_len);

            // Check if key starts with the username prefix
            if(key_str.substr(0, prefix.length()) == prefix) {
                try {
                    // Parse the metadata
                    std::string json_str(static_cast<char*>(data.iov_base), data.iov_len);
                    IndexMetadata metadata =
                            IndexMetadata::from_json(nlohmann::json::parse(json_str));

                    // Extract just the index name part (without username prefix)
                    std::string index_name = key_str.substr(prefix.length());

                    // Add to result
                    indexes.emplace_back(index_name, std::move(metadata));
                } catch(const std::exception& e) {
                    std::cerr << "Failed to parse metadata for index " << key_str << ": "
                              << e.what() << std::endl;
                    // Skip invalid entries
                }
            }
        }

        mdbx_cursor_close(cursor);
        mdbx_txn_abort(txn);
        return indexes;
    }

    std::vector<std::pair<std::string, IndexMetadata>> listAllIndexes() {
        std::vector<std::pair<std::string, IndexMetadata>> result;

        MDBX_txn* txn;
        int rc = mdbx_txn_begin(metadata_env_, nullptr, MDBX_TXN_RDONLY, &txn);
        if(rc != 0) {
            std::cerr << "Failed to begin transaction: " << mdbx_strerror(rc) << std::endl;
            return result;
        }

        MDBX_cursor* cursor;
        rc = mdbx_cursor_open(txn, metadata_dbi_, &cursor);
        if(rc != 0) {
            mdbx_txn_abort(txn);
            std::cerr << "Failed to open cursor: " << mdbx_strerror(rc) << std::endl;
            return result;
        }

        MDBX_val key, data;
        while(mdbx_cursor_get(cursor, &key, &data, MDBX_NEXT) == 0) {
            try {
                std::string key_str(static_cast<char*>(key.iov_base), key.iov_len);
                std::string json_str(static_cast<char*>(data.iov_base), data.iov_len);
                IndexMetadata metadata = IndexMetadata::from_json(nlohmann::json::parse(json_str));
                result.emplace_back(key_str, std::move(metadata));
            } catch(const std::exception& e) {
                std::cerr << "Failed to parse metadata: " << e.what() << std::endl;
                // skip bad record
            }
        }

        mdbx_cursor_close(cursor);
        mdbx_txn_abort(txn);
        return result;
    }

private:
    void initEnvironment() {
        int rc = mdbx_env_create(&metadata_env_);
        if(rc != 0) {
            throw std::runtime_error(std::string("Failed to create metadata LMDB env: ")
                                     + mdbx_strerror(rc));
        }

        // Set geometry for auto-grow
        rc = mdbx_env_set_geometry(
                metadata_env_,
                -1,                                              // lower size bound (use default)
                1ULL << settings::INDEX_META_MAP_SIZE_BITS,      // current/now size
                1ULL << settings::INDEX_META_MAP_SIZE_MAX_BITS,  // upper size bound
                1ULL << settings::INDEX_META_MAP_SIZE_BITS,      // growth step
                -1,                                              // shrink threshold (use default)
                -1);                                             // pagesize (use default)
        if(rc != MDBX_SUCCESS) {
            throw std::runtime_error(std::string("Failed to set geometry: ") + mdbx_strerror(rc));
        }

        rc = mdbx_env_open(metadata_env_,
                           metadata_dir_.c_str(),
                           MDBX_WRITEMAP | MDBX_MAPASYNC | MDBX_NORDAHEAD,
                           0664);
        if(rc != 0) {
            throw std::runtime_error(std::string("Failed to open metadata environment: ")
                                     + mdbx_strerror(rc));
        }

        MDBX_txn* txn;
        rc = mdbx_txn_begin(metadata_env_, nullptr, MDBX_TXN_READWRITE, &txn);
        if(rc != 0) {
            throw std::runtime_error(std::string("Failed to begin transaction: ")
                                     + mdbx_strerror(rc));
        }

        rc = mdbx_dbi_open(txn, nullptr, MDBX_CREATE, &metadata_dbi_);
        if(rc != 0) {
            mdbx_txn_abort(txn);
            throw std::runtime_error(std::string("Failed to open metadata database: ")
                                     + mdbx_strerror(rc));
        }

        rc = mdbx_txn_commit(txn);
        if(rc != 0) {
            throw std::runtime_error(std::string("Failed to commit transaction: ")
                                     + mdbx_strerror(rc));
        }
    }
};