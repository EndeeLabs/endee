#pragma once

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <unordered_set>
#include <filesystem>
#include "mdbx/mdbx.h"
#include "bmw.hpp"
#include "sparse_vector.hpp"
#include "../utils/log.hpp"

namespace ndd {

    class SparseVectorStorage {
    public:
        explicit SparseVectorStorage(const std::string& db_path) :
            db_path_(db_path),
            env_(nullptr) {
            bmw_index_ = nullptr;
        }

        ~SparseVectorStorage() { closeMDBX(); }

        // Initialize storage
        bool initialize() {
            if(!initializeMDBX()) {
                return false;
            }

            bmw_index_ = std::make_unique<BMWIndex>(env_, 0);  // Vocab size unknown/dynamic
            if(!bmw_index_->initialize()) {
                return false;
            }

            updateVectorCount();
            return true;
        }

        // Transaction support
        class Transaction {
        public:
            Transaction(SparseVectorStorage* storage, bool read_only = false) :
                storage_(storage),
                read_only_(read_only),
                committed_(false) {
                int flags = read_only ? MDBX_TXN_RDONLY : MDBX_TXN_READWRITE;
                int rc = mdbx_txn_begin(
                        storage_->env_, nullptr, static_cast<MDBX_txn_flags_t>(flags), &txn_);
                if(rc != 0) {
                    throw std::runtime_error("Failed to begin transaction: "
                                             + std::string(mdbx_strerror(rc)));
                }
            }

            ~Transaction() {
                if(!committed_) {
                    abort();
                }
            }

            bool commit() {
                if(committed_) {
                    return true;
                }
                int rc = mdbx_txn_commit(txn_);
                if(rc == 0) {
                    committed_ = true;
                    return true;
                }
                return false;
            }

            void abort() {
                if(!committed_) {
                    mdbx_txn_abort(txn_);
                    committed_ = true;  // effectively closed
                }
            }

            MDBX_txn* getTxn() { return txn_; }

            bool store_vector(ndd::idInt doc_id, const SparseVector& vec) {
                if(read_only_) {
                    return false;
                }

                // 1. Store in docs DB
                if(!storage_->storeVectorInternal(txn_, doc_id, vec)) {
                    return false;
                }

                // 2. Update Index
                if(!storage_->bmw_index_->addDocumentsBatch(txn_, {{doc_id, vec}})) {
                    return false;
                }

                // 3. Save Metadata (Handled internally by BMWIndex per term)
                // if (!storage_->bmw_index_->saveMetadata(txn_)) return false;

                storage_->vector_count_++;
                return true;
            }

            std::optional<SparseVector> get_vector(ndd::idInt doc_id) const {
                return storage_->getVectorInternal(txn_, doc_id);
            }

            bool delete_vector(ndd::idInt doc_id) {
                if(read_only_) {
                    return false;
                }

                // 1. Get vector to remove from index
                auto vec = get_vector(doc_id);
                if(!vec) {
                    return false;  // Not found
                }

                // 2. Remove from Index
                if(!storage_->bmw_index_->removeDocument(txn_, doc_id, *vec)) {
                    return false;
                }

                // 3. Remove from docs DB
                if(!storage_->deleteVectorInternal(txn_, doc_id)) {
                    return false;
                }

                // 4. Save Metadata (Handled internally)
                // if (!storage_->bmw_index_->saveMetadata(txn_)) return false;

                storage_->vector_count_--;
                return true;
            }

        private:
            SparseVectorStorage* storage_;
            MDBX_txn* txn_;
            bool committed_;
            bool read_only_;
        };

        std::unique_ptr<Transaction> begin_transaction(bool read_only = false) {
            return std::make_unique<Transaction>(this, read_only);
        }

        // Vector management
        bool store_vector(ndd::idInt doc_id, const SparseVector& vec) {
            std::unique_lock<std::shared_mutex> lock(mutex_);
            auto txn = begin_transaction(false);
            if(!txn->store_vector(doc_id, vec)) {
                txn->abort();
                return false;
            }
            return txn->commit();
        }

        std::optional<SparseVector> get_vector(ndd::idInt doc_id) const {
            std::shared_lock<std::shared_mutex> lock(mutex_);
            // Const cast to create read-only transaction
            auto* non_const_this = const_cast<SparseVectorStorage*>(this);
            auto txn = non_const_this->begin_transaction(true);
            return txn->get_vector(doc_id);
        }

        bool delete_vector(ndd::idInt doc_id) {
            std::unique_lock<std::shared_mutex> lock(mutex_);
            auto txn = begin_transaction(false);
            if(!txn->delete_vector(doc_id)) {
                txn->abort();
                return false;
            }
            return txn->commit();
        }

        bool update_vector(ndd::idInt doc_id, const SparseVector& vec) {
            std::unique_lock<std::shared_mutex> lock(mutex_);
            auto txn = begin_transaction(false);

            // Get old vector to remove from index
            auto old_vec = txn->get_vector(doc_id);
            if(old_vec) {
                if(!bmw_index_->removeDocument(txn->getTxn(), doc_id, *old_vec)) {
                    txn->abort();
                    return false;
                }
            }

            // Store new vector (overwrites in docs_dbi)
            if(!storeVectorInternal(txn->getTxn(), doc_id, vec)) {
                txn->abort();
                return false;
            }

            // Add to index
            if(!bmw_index_->addDocumentsBatch(txn->getTxn(), {{doc_id, vec}})) {
                txn->abort();
                return false;
            }

            // Save metadata (Handled internally)
            // if (!bmw_index_->saveMetadata(txn->getTxn())) {
            //    txn->abort();
            //    return false;
            // }

            return txn->commit();
        }

        // Batch operations
        bool store_vectors_batch(const std::vector<std::pair<ndd::idInt, SparseVector>>& batch) {
            std::unique_lock<std::shared_mutex> lock(mutex_);
            auto txn = begin_transaction(false);

            for(const auto& [doc_id, vec] : batch) {
                if(!storeVectorInternal(txn->getTxn(), doc_id, vec)) {
                    txn->abort();
                    return false;
                }
            }

            if(!bmw_index_->addDocumentsBatch(txn->getTxn(), batch)) {
                txn->abort();
                return false;
            }

            // Metadata handled internally
            // if (!bmw_index_->saveMetadata(txn->getTxn())) {
            //    txn->abort();
            //    return false;
            // }

            if(txn->commit()) {
                vector_count_ += batch.size();
                return true;
            }
            return false;
        }

        bool delete_vectors_batch(const std::vector<ndd::idInt>& doc_ids) {
            std::unique_lock<std::shared_mutex> lock(mutex_);
            auto txn = begin_transaction(false);

            for(ndd::idInt doc_id : doc_ids) {
                if(!txn->delete_vector(doc_id)) {
                    // Continue or abort? Usually continue for batch delete
                }
            }
            return txn->commit();
        }

        // Search (delegates to BMW)
        std::vector<std::pair<ndd::idInt, float>> search(const SparseVector& query, size_t k) {
            return bmw_index_->search(query, k);
        }

        // Statistics
        size_t get_vector_count() const { return vector_count_; }
        size_t get_term_count() const { return bmw_index_ ? bmw_index_->getTermCount() : 0; }
        size_t get_block_count() const { return bmw_index_ ? bmw_index_->getBlockCount() : 0; }

        // Maintenance
        bool compact() {
            // MDBX compaction usually involves copying to a new file
            return true;
        }

        bool backup(const std::string& backup_path) {
            // MDBX backup
            return true;
        }

    private:
        std::string db_path_;
        MDBX_env* env_;
        MDBX_dbi docs_dbi_;

        std::unique_ptr<BMWIndex> bmw_index_;
        mutable std::shared_mutex mutex_;

        std::atomic<size_t> vector_count_{0};
        std::unordered_set<ndd::idInt> deleted_docs_;

        // Helper methods
        bool initializeMDBX() {
            int rc = mdbx_env_create(&env_);
            if(rc != 0) {
                LOG_ERROR("mdbx_env_create failed: " << rc);
                return false;
            }

            // Set geometry (max 1TB for now, can be configured)
            rc = mdbx_env_set_geometry(env_, -1, -1, TB, -1, -1, -1);
            if(rc != 0) {
                LOG_ERROR("mdbx_env_set_geometry failed: " << rc);
                return false;
            }

            // Set maxdbs to allow named databases
            rc = mdbx_env_set_maxdbs(env_, 10);
            if(rc != 0) {
                LOG_ERROR("mdbx_env_set_maxdbs failed: " << rc);
                return false;
            }

            std::error_code ec;
            std::filesystem::create_directories(db_path_, ec);
            if(ec) {
                LOG_ERROR("create_directories failed: " << ec.message());
                return false;
            }

            rc = mdbx_env_open(env_,
                               db_path_.c_str(),
                               MDBX_NOSTICKYTHREADS | MDBX_NORDAHEAD | MDBX_LIFORECLAIM,
                               0664);
            if(rc != 0) {
                LOG_ERROR("mdbx_env_open failed: " << rc << " path: " << db_path_);
                return false;
            }

            MDBX_txn* txn;
            rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_READWRITE, &txn);
            if(rc != 0) {
                LOG_ERROR("mdbx_txn_begin failed: " << rc);
                return false;
            }

            rc = mdbx_dbi_open(txn, "sparse_docs", MDBX_CREATE | MDBX_INTEGERKEY, &docs_dbi_);
            if(rc != 0) {
                LOG_ERROR("mdbx_dbi_open failed: " << rc);
                mdbx_txn_abort(txn);
                return false;
            }

            rc = mdbx_txn_commit(txn);
            if(rc != 0) {
                LOG_ERROR("mdbx_txn_commit failed: " << rc);
                return false;
            }
            return true;
        }

        void closeMDBX() {
            if(env_) {
                mdbx_env_close(env_);
                env_ = nullptr;
            }
        }

        bool storeVectorInternal(MDBX_txn* txn, ndd::idInt doc_id, const SparseVector& vec) {
            auto packed = vec.pack();
            MDBX_val key, data;
            key.iov_base = &doc_id;
            key.iov_len = sizeof(ndd::idInt);
            data.iov_base = packed.data();
            data.iov_len = packed.size();

            return mdbx_put(txn, docs_dbi_, &key, &data, MDBX_UPSERT) == 0;
        }

        std::optional<SparseVector> getVectorInternal(MDBX_txn* txn, ndd::idInt doc_id) const {
            MDBX_val key, data;
            key.iov_base = const_cast<ndd::idInt*>(&doc_id);
            key.iov_len = sizeof(ndd::idInt);

            int rc = mdbx_get(txn, docs_dbi_, &key, &data);
            if(rc == MDBX_SUCCESS) {
                return SparseVector(static_cast<const uint8_t*>(data.iov_base), data.iov_len);
            }
            return std::nullopt;
        }

        bool deleteVectorInternal(MDBX_txn* txn, ndd::idInt doc_id) {
            MDBX_val key;
            key.iov_base = &doc_id;
            key.iov_len = sizeof(ndd::idInt);
            return mdbx_del(txn, docs_dbi_, &key, nullptr) == 0;
        }

        void updateVectorCount() {
            MDBX_txn* txn;
            if(mdbx_txn_begin(env_, nullptr, MDBX_TXN_RDONLY, &txn) == 0) {
                MDBX_stat stat;
                if(mdbx_dbi_stat(txn, docs_dbi_, &stat, sizeof(stat)) == 0) {
                    vector_count_ = stat.ms_entries;
                }
                mdbx_txn_abort(txn);
            }
        }
    };

    // MDBX Transaction RAII wrapper
    class MDBXTransaction {
    public:
        MDBXTransaction(MDBX_env* env, bool read_only = false);
        ~MDBXTransaction();

        bool commit();
        void abort();

        MDBX_txn* txn = nullptr;

    private:
        bool committed_ = false;
        bool read_only_;
    };

}  // namespace ndd