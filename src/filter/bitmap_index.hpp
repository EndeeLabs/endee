#pragma once

#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include "mdbx/mdbx.h"
#include "../utils/log.hpp"
#include "../core/types.hpp"

namespace ndd {
    namespace filter {

        class BitmapIndex {
        private:
            MDBX_env* env_;
            MDBX_dbi dbi_;

            static std::string format_filter_key(const std::string& field,
                                                 const std::string& value) {
                return field + ":" + value;
            }

            // Load bitmap from LMDB
            ndd::RoaringBitmap get_bitmap_internal(const std::string& filter_key) const {
                MDBX_txn* txn;
                int rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_RDONLY, &txn);
                if(rc != MDBX_SUCCESS) {
                    throw std::runtime_error("Failed to begin read transaction: "
                                             + std::string(mdbx_strerror(rc)));
                }

                try {
                    MDBX_val key{const_cast<char*>(filter_key.c_str()), filter_key.size()};
                    MDBX_val data;

                    rc = mdbx_get(txn, dbi_, &key, &data);
                    if(rc == MDBX_NOTFOUND) {
                        mdbx_txn_abort(txn);
                        // LOG_DEBUG("Filter key not found: " << filter_key);
                        return ndd::RoaringBitmap();  // Return empty bitmap
                    }
                    if(rc != MDBX_SUCCESS) {
                        mdbx_txn_abort(txn);
                        throw std::runtime_error("Failed to read filter key '" + filter_key
                                                 + "': " + std::string(mdbx_strerror(rc)));
                    }

                    if(data.iov_len == 0) {
                        mdbx_txn_abort(txn);
                        // LOG_DEBUG("Empty data for filter key: " << filter_key);
                        return ndd::RoaringBitmap();
                    }

                    ndd::RoaringBitmap bitmap =
                            ndd::RoaringBitmap::read(static_cast<const char*>(data.iov_base));
                    mdbx_txn_abort(txn);
                    return bitmap;
                } catch(...) {
                    mdbx_txn_abort(txn);
                    throw;
                }
            }

            void store_bitmap_internal(const std::string& filter_key,
                                       const ndd::RoaringBitmap& bitmap) {
                if(bitmap.cardinality() == 0) {
                    // LOG_DEBUG("Storing empty bitmap for key: " << filter_key);
                }

                size_t required_size = bitmap.getSizeInBytes();
                if(required_size == 0) {
                    throw std::runtime_error("Invalid bitmap serialization: size is 0");
                }

                std::vector<char> buffer(required_size);
                bitmap.write(buffer.data(), true);

                MDBX_val key{const_cast<char*>(filter_key.c_str()), filter_key.size()};
                MDBX_val data{const_cast<char*>(buffer.data()), buffer.size()};

                MDBX_txn* txn;
                int rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_READWRITE, &txn);
                if(rc != MDBX_SUCCESS) {
                    throw std::runtime_error("Failed to begin write transaction: "
                                             + std::string(mdbx_strerror(rc)));
                }

                rc = mdbx_put(txn, dbi_, &key, &data, MDBX_UPSERT);
                if(rc != MDBX_SUCCESS) {
                    mdbx_txn_abort(txn);
                    throw std::runtime_error("Failed to store bitmap: "
                                             + std::string(mdbx_strerror(rc)));
                }

                rc = mdbx_txn_commit(txn);
                if(rc != MDBX_SUCCESS) {
                    throw std::runtime_error("Failed to commit transaction: "
                                             + std::string(mdbx_strerror(rc)));
                }
            }

        public:
            BitmapIndex(MDBX_env* env) :
                env_(env) {
                MDBX_txn* txn;
                int rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_READWRITE, &txn);
                if(rc != MDBX_SUCCESS) {
                    throw std::runtime_error("Failed to begin txn for BitmapIndex init");
                }

                // Open default DB or named DB? The original code used nullptr (default DB)
                rc = mdbx_dbi_open(txn, nullptr, MDBX_CREATE, &dbi_);
                if(rc != MDBX_SUCCESS) {
                    mdbx_txn_abort(txn);
                    throw std::runtime_error("Failed to open BitmapIndex dbi");
                }

                mdbx_txn_commit(txn);
            }

            ndd::RoaringBitmap get_bitmap(const std::string& field,
                                          const std::string& value) const {
                return get_bitmap_internal(format_filter_key(field, value));
            }

            // Direct key access for internal use if needed, or expose format_filter_key
            ndd::RoaringBitmap get_bitmap_by_key(const std::string& key) const {
                return get_bitmap_internal(key);
            }

            void add(const std::string& field, const std::string& value, ndd::idInt id) {
                std::string filter_key = format_filter_key(field, value);
                ndd::RoaringBitmap bitmap = get_bitmap_internal(filter_key);
                bitmap.add(id);
                store_bitmap_internal(filter_key, bitmap);
            }

            void remove(const std::string& field, const std::string& value, ndd::idInt id) {
                std::string filter_key = format_filter_key(field, value);
                ndd::RoaringBitmap bitmap = get_bitmap_internal(filter_key);
                bitmap.remove(id);
                store_bitmap_internal(filter_key, bitmap);
            }

            bool contains(const std::string& field, const std::string& value, ndd::idInt id) const {
                std::string filter_key = format_filter_key(field, value);
                ndd::RoaringBitmap bitmap = get_bitmap_internal(filter_key);
                return bitmap.contains(id);
            }

            void add_batch(const std::string& field,
                           const std::string& value,
                           const std::vector<ndd::idInt>& ids) {
                if(ids.empty()) {
                    return;
                }
                std::string filter_key = format_filter_key(field, value);
                ndd::RoaringBitmap bitmap = get_bitmap_internal(filter_key);
                for(const auto& id : ids) {
                    bitmap.add(id);
                }
                store_bitmap_internal(filter_key, bitmap);
            }

            // Helper for batch operations where key is already formatted
            void add_batch_by_key(const std::string& key, const std::vector<ndd::idInt>& ids) {
                if(ids.empty()) {
                    return;
                }
                ndd::RoaringBitmap bitmap = get_bitmap_internal(key);
                for(const auto& id : ids) {
                    bitmap.add(id);
                }
                store_bitmap_internal(key, bitmap);
            }

            // Expose key formatting for external batching logic
            static std::string make_key(const std::string& field, const std::string& value) {
                return format_filter_key(field, value);
            }

            MDBX_dbi get_dbi() const { return dbi_; }
        };

    }  // namespace filter
}  // namespace ndd
