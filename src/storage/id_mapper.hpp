#pragma once

#include "mdbx/mdbx.h"
#include "log.hpp"
#include "auth.hpp"
#include "wal.hpp"
#include <string>
#include <stdexcept>
#include <memory>
#include <mutex>
#include <vector>
#include <numeric>
#include <filesystem>
#include <set>
#include "../core/types.hpp"
#include "../utils/settings.hpp"

using ndd::idInt;
class IDMapper {
public:
    IDMapper(const std::string& path, bool is_new = false, UserType user_type = UserType::Admin) :
        path_(path),
        user_type_(user_type) {
        if(is_new) {
            std::filesystem::create_directories(path);
        }
        int rc = mdbx_env_create(&env_);
        if(rc != MDBX_SUCCESS) {
            throw std::runtime_error(std::string("Failed to create MDBX environment: ")
                                     + mdbx_strerror(rc));
        }

        // Set geometry for auto-grow
        rc = mdbx_env_set_geometry(
                env_,
                -1,                                             // lower size bound (use default)
                1ULL << settings::ID_MAPPER_MAP_SIZE_BITS,      // current/now size (512MB)
                1ULL << settings::ID_MAPPER_MAP_SIZE_MAX_BITS,  // upper size bound (16GB)
                1ULL << settings::ID_MAPPER_MAP_SIZE_BITS,      // growth step (256MB)
                -1,                                             // shrink threshold (use default)
                -1);                                            // pagesize (use default)
        if(rc != MDBX_SUCCESS) {
            throw std::runtime_error(std::string("Failed to set geometry: ") + mdbx_strerror(rc));
        }

        rc = mdbx_env_open(
                env_, path.c_str(), MDBX_WRITEMAP | MDBX_MAPASYNC | MDBX_NORDAHEAD, 0664);
        if(rc != MDBX_SUCCESS) {
            throw std::runtime_error(std::string("Failed to open environment: ")
                                     + mdbx_strerror(rc));
        }

        MDBX_txn* txn;
        rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_READWRITE, &txn);
        if(rc != MDBX_SUCCESS) {
            throw std::runtime_error(std::string("Failed to begin transaction: ")
                                     + mdbx_strerror(rc));
        }

        rc = mdbx_dbi_open(txn, nullptr, MDBX_CREATE, &dbi_);
        if(rc != MDBX_SUCCESS) {
            mdbx_txn_abort(txn);
            throw std::runtime_error(std::string("Failed to open database: ") + mdbx_strerror(rc));
        }

        rc = mdbx_txn_commit(txn);
        if(rc != MDBX_SUCCESS) {
            throw std::runtime_error(std::string("Failed to commit transaction: ")
                                     + mdbx_strerror(rc));
        }

        if(is_new) {
            init_next_id();
        }
    }

    ~IDMapper() {
        mdbx_dbi_close(env_, dbi_);
        mdbx_env_close(env_);
    }

    // Create string ID to numeric ID mapping. If string ids exists in the database, it will return
    // the existing numeric ID along with flag It will also use old numeric IDs of deleted points
    template <bool use_deleted_ids>
    std::vector<std::pair<idInt, bool>> create_ids_batch(const std::vector<std::string>& str_ids,
                                                         void* wal_ptr = nullptr) {
        if(str_ids.empty()) {
            return {};
        }
        LOG_DEBUG("=== create_ids_batch START ===");
        LOG_DEBUG("Processing " << str_ids.size() << " string IDs");
        // for (size_t i = 0; i < str_ids.size(); i++) {
        //     LOG_DEBUG("Input[" << i << "]: [" << str_ids[i] << "] length=" <<
        //     str_ids[i].length());
        // }

        LOG_TIME("create_ids_batch");
        constexpr idInt INVALID_LABEL = static_cast<idInt>(-1);
        // Tuple: <str_id, numeric_id, is_new_to_db, is_reused>
        std::vector<std::tuple<std::string, idInt, bool, bool>> id_tuples;

        id_tuples.reserve(str_ids.size());
        for(const auto& str_id : str_ids) {
            // true means that the ID is new and false means that the ID already exists
            // is_reused defaults to false
            id_tuples.emplace_back(str_id, INVALID_LABEL, true, false);
        }

        //Read-only LMDB check
        LOG_DEBUG("--- STEP 2: LMDB database check ---");
        {
            MDBX_txn* txn;
            int rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_RDONLY, &txn);
            if(rc != MDBX_SUCCESS) {
                LOG_DEBUG("ERROR: Failed to begin read-only transaction: " << mdbx_strerror(rc));
                throw std::runtime_error("Failed to begin read-only transaction: "
                                         + std::string(mdbx_strerror(rc)));
            }
            LOG_DEBUG("LMDB read-only transaction started successfully");

            try {
                int keys_checked = 0;
                for(auto& tup : id_tuples) {
                    if(std::get<1>(tup) == INVALID_LABEL) {
                        const std::string& str_id = std::get<0>(tup);
                        MDBX_val key{(void*)str_id.c_str(), str_id.size()};
                        MDBX_val data;

                        // Add debug logging
                        LOG_DEBUG("LMDB: Checking key[" << keys_checked << "]: [" << str_id
                                                        << "] size: " << str_id.size());
                        keys_checked++;

                        rc = mdbx_get(txn, dbi_, &key, &data);
                        if(rc == MDBX_SUCCESS) {
                            idInt existing_id = *(idInt*)data.iov_base;
                            LOG_DEBUG("LMDB: ✓ FOUND existing ID: " << existing_id << " for key: ["
                                                                    << str_id << "]");
                            std::get<1>(tup) = existing_id;
                            std::get<2>(tup) = false;  // ID already exists
                        } else if(rc == MDBX_NOTFOUND) {
                            LOG_DEBUG("LMDB: ✗ NOT FOUND: [" << str_id << "]");
                            std::get<1>(tup) = 0;
                        } else {
                            LOG_DEBUG("LMDB: ERROR for key: [" << str_id
                                                               << "] error: " << mdbx_strerror(rc));
                            mdbx_txn_abort(txn);
                            throw std::runtime_error("Database error checking ID: "
                                                     + std::string(mdbx_strerror(rc)));
                        }
                    }
                }
                LOG_DEBUG("LMDB: Checked " << keys_checked << " keys in database");
                mdbx_txn_abort(txn);
                LOG_DEBUG("LMDB check done");
            } catch(...) {
                mdbx_txn_abort(txn);
                throw;
            }
        }

        //Count and generate new IDs
        LOG_DEBUG("--- STEP 3: Count and generate new IDs ---");
        size_t total_new_ids_needed =
                std::count_if(id_tuples.begin(), id_tuples.end(), [](const auto& t) {
                    return std::get<1>(t) == 0;
                });
        LOG_DEBUG("Total new IDs needed: " << total_new_ids_needed);

        size_t fresh_ids_count = total_new_ids_needed;
        size_t deleted_index = 0;

        if(use_deleted_ids) {
            // Use deleted IDs first, but ONLY for entries that are actually new (not found in DB)
            std::vector<idInt> deletedIds = getDeletedIds(fresh_ids_count);

            for(auto& tup : id_tuples) {
                // Only assign deleted IDs to entries that are new (id=0 and is_new=true)
                if(std::get<1>(tup) == 0 && std::get<2>(tup) == true
                   && deleted_index < deletedIds.size()) {
                    std::get<1>(tup) = deletedIds[deleted_index++];
                    std::get<3>(tup) = true;  // Mark as reused
                    // Keep std::get<2>(tup) as true because this still needs to be written to DB
                }
            }
            fresh_ids_count -= deleted_index;  // Reduce by actual number of deleted IDs used
        }

        if(total_new_ids_needed > 0) {
            LOG_DEBUG("Generating " << fresh_ids_count << " fresh IDs");

            std::vector<idInt> new_ids;
            if(fresh_ids_count > 0) {
                new_ids = get_next_ids(fresh_ids_count);
            }

            // CRITICAL FIX: Log to WAL AFTER generating IDs (minimal risk window)
            if(wal_ptr) {
                WriteAheadLog* wal = static_cast<WriteAheadLog*>(wal_ptr);
                std::vector<WriteAheadLog::WALEntry> wal_entries;

                // Log reused IDs
                for(const auto& tup : id_tuples) {
                    if(std::get<2>(tup) && std::get<1>(tup) != 0) {
                        wal_entries.push_back({WALOperationType::VECTOR_ADD, std::get<1>(tup)});
                    }
                }

                // Log fresh IDs
                for(idInt id : new_ids) {
                    wal_entries.push_back({WALOperationType::VECTOR_ADD, id});
                }

                if(!wal_entries.empty()) {
                    wal->log(wal_entries);
                }
            }

            if(fresh_ids_count > 0 && new_ids.size() != fresh_ids_count) {
                throw std::runtime_error("Mismatch: get_next_ids returned "
                                         + std::to_string(new_ids.size()) + " but expected "
                                         + std::to_string(fresh_ids_count));
            }

            size_t new_id_index = 0;

            // Step 4: Write txn with auto-resize retry
            LOG_DEBUG("--- STEP 4: Writing to database ---");
            auto try_write = [&](MDBX_txn* txn) -> int {
                int writes_attempted = 0;
                for(auto& tup : id_tuples) {
                    // Write entries that need to be written to DB (is_new=true) but don't have ID=0
                    if(std::get<2>(tup) == true && std::get<1>(tup) != 0) {
                        const std::string& str_id = std::get<0>(tup);
                        idInt id = std::get<1>(tup);

                        MDBX_val key{(void*)str_id.c_str(), str_id.size()};
                        MDBX_val data{&id, sizeof(idInt)};

                        // Add debug logging for write operations
                        LOG_DEBUG("WRITE[" << writes_attempted << "]: key=[" << str_id
                                           << "] size=" << str_id.size() << " ID=" << id);
                        writes_attempted++;

                        int rc = mdbx_put(txn, dbi_, &key, &data, MDBX_UPSERT);
                        if(rc == MDBX_MAP_FULL) {
                            LOG_DEBUG("WRITE ERROR: MDBX_MAP_FULL for key=[" << str_id << "]");
                            return MDBX_MAP_FULL;
                        }
                        if(rc != MDBX_SUCCESS) {
                            LOG_DEBUG("WRITE ERROR: [" << str_id
                                                       << "] error: " << mdbx_strerror(rc));
                            return rc;
                        }

                        LOG_DEBUG("WRITE SUCCESS: [" << str_id << "] with ID: " << id);

                    } else if(std::get<1>(tup) == 0) {
                        // Handle remaining entries that still need new IDs
                        if(new_id_index >= new_ids.size()) {
                            LOG_DEBUG("ERROR: new_id_index ("
                                      << new_id_index << ") >= new_ids.size() (" << new_ids.size()
                                      << ")");
                            return MDBX_PROBLEM;  // Internal error
                        }
                        idInt new_id = new_ids[new_id_index++];
                        const std::string& str_id = std::get<0>(tup);

                        MDBX_val key{(void*)str_id.c_str(), str_id.size()};
                        MDBX_val data{&new_id, sizeof(idInt)};

                        writes_attempted++;

                        int rc = mdbx_put(txn, dbi_, &key, &data, MDBX_UPSERT);
                        if(rc == MDBX_MAP_FULL) {
                            LOG_DEBUG("WRITE_NEW ERROR: MDBX_MAP_FULL for key=[" << str_id << "]");
                            return MDBX_MAP_FULL;
                        }
                        if(rc != MDBX_SUCCESS) {
                            LOG_DEBUG("WRITE_NEW ERROR: [" << str_id
                                                           << "] error: " << mdbx_strerror(rc));
                            return rc;
                        }

                        std::get<1>(tup) = new_id;
                    }
                }
                return MDBX_SUCCESS;
            };

            MDBX_txn* txn;
            int rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_READWRITE, &txn);
            if(rc != MDBX_SUCCESS) {
                throw std::runtime_error("Failed to begin write transaction: "
                                         + std::string(mdbx_strerror(rc)));
            }

            rc = try_write(txn);
            // MDBX auto-grows, no manual resize needed
            if(rc != MDBX_SUCCESS) {
                mdbx_txn_abort(txn);
                throw std::runtime_error("Failed to insert new IDs: "
                                         + std::string(mdbx_strerror(rc)));
            }

            rc = mdbx_txn_commit(txn);
            if(rc != MDBX_SUCCESS) {
                throw std::runtime_error("Failed to commit transaction: "
                                         + std::string(mdbx_strerror(rc)));
            }
            LOG_DEBUG("Write transaction committed successfully");
        } else {
            LOG_DEBUG("No new IDs needed, skipping write transaction");
        }

        // Final state logging
        LOG_DEBUG("--- FINAL RESULTS ---");
        std::vector<std::pair<idInt, bool>> result;
        result.reserve(id_tuples.size());
        for(size_t i = 0; i < id_tuples.size(); i++) {
            const auto& tup = id_tuples[i];
            bool is_new_to_hnsw = std::get<2>(tup);
            // If the ID was reused from deleted list, treat it as an update (not new to HNSW)
            if(std::get<3>(tup)) {
                is_new_to_hnsw = false;
            }
            result.emplace_back(std::get<1>(tup), is_new_to_hnsw);
        }
        LOG_DEBUG("=== create_ids_batch END ===");
        return result;
    }

    // Get the number of key-value pairs in the database
    size_t get_count() const {
        MDBX_txn* txn;
        MDBX_stat stat;
        int rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_RDONLY, &txn);
        if(rc != MDBX_SUCCESS) {
            throw std::runtime_error(std::string("Failed to begin transaction: ")
                                     + mdbx_strerror(rc));
        }

        rc = mdbx_dbi_stat(txn, dbi_, &stat, sizeof(stat));
        if(rc != MDBX_SUCCESS) {
            mdbx_txn_abort(txn);
            throw std::runtime_error(std::string("Failed to get database statistics: ")
                                     + mdbx_strerror(rc));
        }

        mdbx_txn_abort(txn);
        return stat.ms_entries - 1;  // Subtract 1 for NEXT_ID_KEY
    }

    // Get ID for a string (returns 0 if not found)
    idInt get_id(const std::string& str_id) const {
        LOG_DEBUG("=== get_id START for: [" << str_id << "] size: " << str_id.size() << " ===");

        MDBX_txn* txn;
        int rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_RDONLY, &txn);
        if(rc != MDBX_SUCCESS) {
            LOG_DEBUG("get_id: ERROR - Failed to begin transaction: " << mdbx_strerror(rc));
            throw std::runtime_error(std::string("Failed to begin transaction: ")
                                     + mdbx_strerror(rc));
        }
        LOG_DEBUG("get_id: LMDB read transaction started");

        MDBX_val key, data;
        key.iov_len = str_id.size();
        key.iov_base = (void*)str_id.c_str();

        LOG_DEBUG("get_id: LMDB lookup for key: [" << str_id << "] size: " << str_id.size());

        rc = mdbx_get(txn, dbi_, &key, &data);
        if(rc == MDBX_SUCCESS) {
            idInt id = *(idInt*)data.iov_base;
            LOG_DEBUG("get_id: ✓ FOUND ID: " << id << " for key: [" << str_id << "]");
            mdbx_txn_abort(txn);
            LOG_DEBUG("=== get_id END - FOUND ===");
            return id;
        } else if(rc == MDBX_NOTFOUND) {
            LOG_DEBUG("get_id: ✗ NOT FOUND: [" << str_id << "]");
        } else {
            LOG_DEBUG("get_id: ERROR: [" << str_id << "] error: " << mdbx_strerror(rc));
        }

        mdbx_txn_abort(txn);
        LOG_DEBUG("=== get_id END - NOT FOUND ===");
        return 0;  // Not found
    }

    // Deletes mapping from string_id to numeric_id, append to DELETED_IDS_KEY
    // Returns the deleted numeric_ids, if strings is not found, returns 0
    std::vector<idInt> deletePoints(const std::vector<std::string>& external_ids) {
        std::vector<idInt> deleted_ids;
        MDBX_txn* txn;
        mdbx_txn_begin(env_, nullptr, MDBX_TXN_READWRITE, &txn);

        MDBX_val key, data;
        for(const auto& ext_id : external_ids) {
            key.iov_len = ext_id.size();
            key.iov_base = const_cast<char*>(ext_id.data());

            if(mdbx_get(txn, dbi_, &key, &data) == MDBX_SUCCESS) {
                idInt label = *reinterpret_cast<idInt*>(data.iov_base);
                deleted_ids.push_back(label);
                mdbx_del(txn, dbi_, &key, nullptr);
            } else {
                deleted_ids.push_back(0);
            }
        }

        // Now append deleted_ids to DELETED_IDS_KEY
        if(!deleted_ids.empty()) {
            std::string del_key = DELETED_IDS_KEY;
            MDBX_val del_mdb_key, del_mdb_val;

            del_mdb_key.iov_len = del_key.size();
            del_mdb_key.iov_base = const_cast<char*>(del_key.data());

            // Fetch existing
            std::vector<idInt> existing;
            if(mdbx_get(txn, dbi_, &del_mdb_key, &del_mdb_val) == MDBX_SUCCESS) {
                size_t count = del_mdb_val.iov_len / sizeof(idInt);
                idInt* raw = reinterpret_cast<idInt*>(del_mdb_val.iov_base);
                existing.insert(existing.end(), raw, raw + count);
            }

            // Append new ones
            for(idInt l : deleted_ids) {
                if(l != 0) {
                    existing.push_back(l);
                }
            }

            del_mdb_val.iov_len = existing.size() * sizeof(idInt);
            del_mdb_val.iov_base = existing.data();
            mdbx_put(txn, dbi_, &del_mdb_key, &del_mdb_val, MDBX_UPSERT);
        }

        mdbx_txn_commit(txn);
        return deleted_ids;
    }

    // returns a vector of deleted ids after removing them from DELETED_IDS_KEY
    std::vector<idInt> getDeletedIds(size_t max_count) {
        std::vector<idInt> result;
        MDBX_txn* txn;
        mdbx_txn_begin(env_, nullptr, MDBX_TXN_READWRITE, &txn);

        std::string del_key = DELETED_IDS_KEY;
        MDBX_val key, val;
        key.iov_len = del_key.size();
        key.iov_base = const_cast<char*>(del_key.data());

        if(mdbx_get(txn, dbi_, &key, &val) != MDBX_SUCCESS) {
            mdbx_txn_abort(txn);
            return result;
        }

        size_t total = val.iov_len / sizeof(idInt);
        idInt* raw = reinterpret_cast<idInt*>(val.iov_base);

        size_t count = std::min(max_count, total);
        result.insert(result.end(), raw, raw + count);

        // Write back the remaining
        if(count < total) {
            MDBX_val new_val;
            new_val.iov_len = (total - count) * sizeof(idInt);
            new_val.iov_base = raw + count;
            mdbx_put(txn, dbi_, &key, &new_val, MDBX_UPSERT);
        } else {
            // Delete the key entirely
            mdbx_del(txn, dbi_, &key, nullptr);
        }

        mdbx_txn_commit(txn);
        return result;
    }

    // Public method to add failed IDs back to deleted_ids for reuse
    void reclaim_failed_ids(const std::vector<idInt>& failed_ids) {
        add_to_deleted_ids(failed_ids);
    }

    // Public method to update user type
    void update_user_type(UserType new_user_type) {
        user_type_ = new_user_type;
        // It will grow automatically as needed via compact operations
    }

private:
    MDBX_env* env_;
    MDBX_dbi dbi_;
    std::string path_;
    UserType user_type_;
    mutable std::mutex mutex_;  // Only used for next_id management
    // Along with string:number pairs, the database also stores a key for next_id. They key for next
    // id also has random alphanumeric characters to avoid collision with other keys. The key is
    // stored as a string.
    static const std::string NEXT_ID_KEY;
    static const std::string DELETED_IDS_KEY;

    // Atomic operation to get and increment next_ids
    std::vector<idInt> get_next_ids(size_t size = 1) {
        std::lock_guard<std::mutex> lock(mutex_);
        MDBX_txn* txn;
        int rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_READWRITE, &txn);
        if(rc != 0) {
            throw std::runtime_error(std::string("Failed to begin transaction: ")
                                     + mdbx_strerror(rc));
        }

        try {
            MDBX_val key{(void*)NEXT_ID_KEY.c_str(), NEXT_ID_KEY.size()};
            MDBX_val data;
            idInt current_id = 0;

            rc = mdbx_get(txn, dbi_, &key, &data);
            if(rc == 0) {
                current_id = *(idInt*)data.iov_base;
            } else if(rc != MDBX_NOTFOUND) {
                throw std::runtime_error(std::string("Failed to get next_id: ")
                                         + mdbx_strerror(rc));
            }

            // CRITICAL FIX: Log VECTOR_ADD to WAL BEFORE incrementing next_id
            // This is now handled in create_ids_batch after ID generation

            idInt next_id = current_id + size;
            data.iov_len = sizeof(idInt);
            data.iov_base = &next_id;

            rc = mdbx_put(txn, dbi_, &key, &data, MDBX_UPSERT);
            if(rc != 0) {
                throw std::runtime_error(std::string("Failed to store next_id: ")
                                         + mdbx_strerror(rc));
            }

            rc = mdbx_txn_commit(txn);
            if(rc != 0) {
                throw std::runtime_error(std::string("Failed to commit transaction: ")
                                         + mdbx_strerror(rc));
            }
            // Return a vector of ids starting from current_id
            std::vector<idInt> ids(size);
            std::iota(ids.begin(), ids.end(), current_id);
            return ids;
        } catch(...) {
            mdbx_txn_abort(txn);
            throw;
        }
    }

    // Helper method to add IDs to deleted_ids list
    void add_to_deleted_ids(const std::vector<idInt>& ids) {
        if(ids.empty()) {
            return;
        }

        MDBX_txn* txn;
        int rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_READWRITE, &txn);
        if(rc != 0) {
            return;  // Silently fail for recovery
        }

        try {
            std::string del_key = DELETED_IDS_KEY;
            MDBX_val del_mdb_key, del_mdb_val;

            del_mdb_key.iov_len = del_key.size();
            del_mdb_key.iov_base = const_cast<char*>(del_key.data());

            // Fetch existing deleted IDs
            std::vector<idInt> existing;
            if(mdbx_get(txn, dbi_, &del_mdb_key, &del_mdb_val) == MDBX_SUCCESS) {
                size_t count = del_mdb_val.iov_len / sizeof(idInt);
                idInt* raw = reinterpret_cast<idInt*>(del_mdb_val.iov_base);
                existing.insert(existing.end(), raw, raw + count);
            }

            // Add new IDs
            for(idInt id : ids) {
                existing.push_back(id);
            }

            // Write back to DB
            del_mdb_val.iov_len = existing.size() * sizeof(idInt);
            del_mdb_val.iov_base = existing.data();
            mdbx_put(txn, dbi_, &del_mdb_key, &del_mdb_val, MDBX_UPSERT);

            mdbx_txn_commit(txn);
        } catch(...) {
            mdbx_txn_abort(txn);
        }
    }

    // Initialize next_id .. called only once during construction
    void init_next_id() {
        MDBX_txn* txn;
        int rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_READWRITE, &txn);
        if(rc != 0) {
            throw std::runtime_error(std::string("Failed to begin transaction: ")
                                     + mdbx_strerror(rc));
        }

        try {
            MDBX_val key{(void*)NEXT_ID_KEY.c_str(), NEXT_ID_KEY.size()};
            MDBX_val data;
            idInt next_id = 1;  // Default starting value

            // Store the next_id (whether new or existing)
            data.iov_len = sizeof(idInt);
            data.iov_base = &next_id;
            rc = mdbx_put(txn, dbi_, &key, &data, MDBX_UPSERT);
            if(rc != 0) {
                throw std::runtime_error(std::string("Failed to store next_id: ")
                                         + mdbx_strerror(rc));
            }

            rc = mdbx_txn_commit(txn);
            if(rc != 0) {
                throw std::runtime_error(std::string("Failed to commit transaction: ")
                                         + mdbx_strerror(rc));
            }

        } catch(...) {
            mdbx_txn_abort(txn);
            throw;
        }
    }
};

inline const std::string IDMapper::NEXT_ID_KEY = "__next_id_px7b39lw__";
inline const std::string IDMapper::DELETED_IDS_KEY = "__deleted_ids_px7b39lw__";