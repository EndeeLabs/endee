#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include "mdbx/mdbx.h"
#include "../utils/log.hpp"
#include "../core/types.hpp"

namespace ndd {
    namespace numeric {

        // Sortable Key Utilities
        inline uint32_t float_to_sortable(float f) {
            uint32_t i;
            std::memcpy(&i, &f, sizeof(float));
            // IEEE 754 floats:
            // If f >= 0 (sign bit 0): map to [0x80000000, 0xFFFFFFFF]
            // If f < 0 (sign bit 1): map to [0x00000000, 0x7FFFFFFF]
            uint32_t mask = (int32_t(i) >> 31) | 0x80000000;
            return i ^ mask;
        }

        inline float sortable_to_float(uint32_t i) {
            uint32_t mask = ((i >> 31) - 1) | 0x80000000;
            uint32_t result = i ^ mask;
            float f;
            std::memcpy(&f, &result, sizeof(float));
            return f;
        }

        inline uint32_t int_to_sortable(int32_t i) {
            return static_cast<uint32_t>(i) ^ 0x80000000;
        }

        inline int32_t sortable_to_int(uint32_t i) {
            return static_cast<int32_t>(i ^ 0x80000000);
        }

        // Bucket Structure
        struct Bucket {
            static constexpr size_t MAX_SIZE = 512;                // Increased bucket size
            std::vector<std::pair<uint32_t, ndd::idInt>> entries;  // value, doc_id

            // Serialize to byte buffer
            std::vector<uint8_t> serialize() const {
                // Format: Count(4) + [Value(4) + ID(sizeof(idInt))] * N
                std::vector<uint8_t> buffer;
                buffer.reserve(4 + entries.size() * (4 + sizeof(ndd::idInt)));

                uint32_t count = static_cast<uint32_t>(entries.size());
                buffer.insert(buffer.end(), (uint8_t*)&count, (uint8_t*)&count + 4);

                for(const auto& entry : entries) {
                    buffer.insert(buffer.end(), (uint8_t*)&entry.first, (uint8_t*)&entry.first + 4);
                    buffer.insert(buffer.end(),
                                  (uint8_t*)&entry.second,
                                  (uint8_t*)&entry.second + sizeof(ndd::idInt));
                }
                return buffer;
            }

            static Bucket deserialize(const void* data, size_t len) {
                Bucket b;
                if(len < 4) {
                    return b;
                }

                const uint8_t* ptr = static_cast<const uint8_t*>(data);
                uint32_t count;
                std::memcpy(&count, ptr, 4);
                ptr += 4;

                size_t entry_size = 4 + sizeof(ndd::idInt);
                if(len < 4 + count * entry_size) {
                    // Corrupt data or partial read
                    return b;
                }

                b.entries.reserve(count);
                for(uint32_t i = 0; i < count; ++i) {
                    uint32_t val;
                    ndd::idInt id;
                    std::memcpy(&val, ptr, 4);
                    ptr += 4;
                    std::memcpy(&id, ptr, sizeof(ndd::idInt));
                    ptr += sizeof(ndd::idInt);
                    b.entries.emplace_back(val, id);
                }
                return b;
            }

            void add(uint32_t val, ndd::idInt id) {
                entries.emplace_back(val, id);
                // Keep sorted by value
                std::sort(entries.begin(), entries.end());
            }

            bool remove(ndd::idInt id) {
                auto it = std::remove_if(entries.begin(), entries.end(), [id](const auto& p) {
                    return p.second == id;
                });
                if(it != entries.end()) {
                    entries.erase(it, entries.end());
                    return true;
                }
                return false;
            }

            bool is_full() const { return entries.size() >= MAX_SIZE; }
            bool is_empty() const { return entries.empty(); }

            // Split bucket into two, returning the new bucket (upper half)
            Bucket split() {
                Bucket new_bucket;
                size_t mid = entries.size() / 2;

                new_bucket.entries.assign(entries.begin() + mid, entries.end());
                entries.resize(mid);

                return new_bucket;
            }

            uint32_t min_val() const { return entries.empty() ? 0 : entries.front().first; }
            uint32_t max_val() const { return entries.empty() ? 0 : entries.back().first; }
        };

        class NumericIndex {
        private:
            MDBX_env* env_;
            MDBX_dbi forward_dbi_;   // ID -> Value (Field:ID -> Value)
            MDBX_dbi inverted_dbi_;  // BucketKey -> Bucket (Field:StartVal -> BucketBlob)

            // Helper to format keys
            std::string make_forward_key(const std::string& field, ndd::idInt id) {
                return field + ":" + std::to_string(id);
            }

            std::string make_bucket_key(const std::string& field, uint32_t start_val) {
                // Big-endian for sorting
                uint32_t be_val = 0;
#if defined(__GNUC__) || defined(__clang__)
                be_val = __builtin_bswap32(start_val);
#else
                // Fallback for MSVC or others
                be_val = ((start_val >> 24) & 0xff) | ((start_val << 8) & 0xff0000)
                         | ((start_val >> 8) & 0xff00) | ((start_val << 24) & 0xff000000);
#endif

                std::string key = field + ":";
                key.append((char*)&be_val, 4);
                return key;
            }

            // Helper to parse bucket key
            uint32_t parse_bucket_key_val(const std::string& key) {
                if(key.size() < 4) {
                    return 0;
                }
                uint32_t be_val;
                std::memcpy(&be_val, key.data() + key.size() - 4, 4);
#if defined(__GNUC__) || defined(__clang__)
                return __builtin_bswap32(be_val);
#else
                return ((be_val >> 24) & 0xff) | ((be_val << 8) & 0xff0000)
                       | ((be_val >> 8) & 0xff00) | ((be_val << 24) & 0xff000000);
#endif
            }

        public:
            NumericIndex(MDBX_env* env) :
                env_(env) {
                MDBX_txn* txn;
                int rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_READWRITE, &txn);
                if(rc != MDBX_SUCCESS) {
                    throw std::runtime_error("Failed to begin txn for NumericIndex init");
                }

                rc = mdbx_dbi_open(txn, "numeric_forward", MDBX_CREATE, &forward_dbi_);
                if(rc != MDBX_SUCCESS) {
                    throw std::runtime_error("Failed to open numeric_forward dbi");
                }

                rc = mdbx_dbi_open(txn, "numeric_inverted", MDBX_CREATE, &inverted_dbi_);
                if(rc != MDBX_SUCCESS) {
                    throw std::runtime_error("Failed to open numeric_inverted dbi");
                }

                mdbx_txn_commit(txn);
            }

            void put(const std::string& field, ndd::idInt id, uint32_t value) {
                MDBX_txn* txn;
                mdbx_txn_begin(env_, nullptr, MDBX_TXN_READWRITE, &txn);
                try {
                    put_internal(txn, field, id, value);
                    mdbx_txn_commit(txn);
                } catch(...) {
                    mdbx_txn_abort(txn);
                    throw;
                }
            }

            void
            put_internal(MDBX_txn* txn, const std::string& field, ndd::idInt id, uint32_t value) {
                // 1. Check Forward Index for existing value (Update case)
                std::string fwd_key_str = make_forward_key(field, id);
                MDBX_val fwd_key{const_cast<char*>(fwd_key_str.data()), fwd_key_str.size()};
                MDBX_val fwd_val;

                int rc = mdbx_get(txn, forward_dbi_, &fwd_key, &fwd_val);
                if(rc == MDBX_SUCCESS) {
                    uint32_t old_val;
                    std::memcpy(&old_val, fwd_val.iov_base, 4);
                    if(old_val == value) {
                        return;  // No change
                    }

                    // Remove from old bucket
                    remove_from_bucket(txn, field, old_val, id);
                }

                // 2. Update Forward Index
                MDBX_val new_val_data{&value, sizeof(uint32_t)};
                mdbx_put(txn, forward_dbi_, &fwd_key, &new_val_data, MDBX_UPSERT);

                // 3. Add to Inverted Index (Buckets)
                add_to_bucket(txn, field, value, id);
            }

            void remove(const std::string& field, ndd::idInt id) {
                MDBX_txn* txn;
                mdbx_txn_begin(env_, nullptr, MDBX_TXN_READWRITE, &txn);
                try {
                    std::string fwd_key_str = make_forward_key(field, id);
                    MDBX_val fwd_key{const_cast<char*>(fwd_key_str.data()), fwd_key_str.size()};
                    MDBX_val fwd_val;

                    int rc = mdbx_get(txn, forward_dbi_, &fwd_key, &fwd_val);
                    if(rc == MDBX_SUCCESS) {
                        uint32_t old_val;
                        std::memcpy(&old_val, fwd_val.iov_base, 4);

                        // Remove from bucket
                        remove_from_bucket(txn, field, old_val, id);

                        // Remove from forward index
                        mdbx_del(txn, forward_dbi_, &fwd_key, nullptr);
                    }
                    mdbx_txn_commit(txn);
                } catch(...) {
                    mdbx_txn_abort(txn);
                    throw;
                }
            }

            ndd::RoaringBitmap range(const std::string& field, uint32_t min_val, uint32_t max_val) {
                ndd::RoaringBitmap result;
                MDBX_txn* txn;
                int rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_RDONLY, &txn);
                if(rc != MDBX_SUCCESS) {
                    return result;
                }

                try {
                    MDBX_cursor* cursor;
                    mdbx_cursor_open(txn, inverted_dbi_, &cursor);

                    // 1. Find start bucket (bucket with start_val <= min_val)
                    std::string start_key_str = make_bucket_key(field, min_val);
                    MDBX_val key{const_cast<char*>(start_key_str.data()), start_key_str.size()};
                    MDBX_val data;

                    rc = mdbx_cursor_get(cursor, &key, &data, MDBX_SET_RANGE);

                    bool valid_start = false;

                    if(rc == MDBX_SUCCESS) {
                        std::string found_key((char*)key.iov_base, key.iov_len);
                        if(found_key.rfind(field + ":", 0) == 0) {
                            // Found a bucket in the same field
                            if(found_key > start_key_str) {
                                // We landed on a bucket starting AFTER min_val.
                                // Check previous bucket to see if it covers min_val.
                                MDBX_val p_key = key;
                                MDBX_val p_data;
                                int p_rc = mdbx_cursor_get(cursor, &p_key, &p_data, MDBX_PREV);

                                if(p_rc == MDBX_SUCCESS) {
                                    std::string prev_key((char*)p_key.iov_base, p_key.iov_len);
                                    if(prev_key.rfind(field + ":", 0) == 0) {
                                        // Previous bucket is in same field, start there
                                        valid_start = true;
                                        // cursor is already at prev
                                        key = p_key;
                                        data = p_data;
                                    } else {
                                        // Previous bucket is different field.
                                        // This means min_val is before the first bucket of this
                                        // field. So we start at the found_key (first bucket). Reset
                                        // cursor to found_key
                                        mdbx_cursor_get(cursor, &key, &data, MDBX_SET_RANGE);
                                        valid_start = true;
                                    }
                                } else {
                                    // No prev, start at found_key
                                    mdbx_cursor_get(cursor, &key, &data, MDBX_SET_RANGE);
                                    valid_start = true;
                                }
                            } else {
                                // Exact match on start key
                                valid_start = true;
                            }
                        } else {
                            // Found key is next field. Go back to see if we have buckets for this
                            // field.
                            rc = mdbx_cursor_get(cursor, &key, &data, MDBX_PREV);
                            if(rc == MDBX_SUCCESS) {
                                std::string prev_key((char*)key.iov_base, key.iov_len);
                                if(prev_key.rfind(field + ":", 0) == 0) {
                                    valid_start = true;
                                }
                            }
                        }
                    } else if(rc == MDBX_NOTFOUND) {
                        // Try last bucket
                        rc = mdbx_cursor_get(cursor, &key, &data, MDBX_LAST);
                        if(rc == MDBX_SUCCESS) {
                            std::string last_key((char*)key.iov_base, key.iov_len);
                            if(last_key.rfind(field + ":", 0) == 0) {
                                valid_start = true;
                            }
                        }
                    }

                    if(valid_start) {
                        // Iterate buckets
                        while(true) {
                            std::string curr_key((char*)key.iov_base, key.iov_len);
                            if(curr_key.rfind(field + ":", 0) != 0) {
                                break;  // End of field
                            }

                            uint32_t bucket_start = parse_bucket_key_val(curr_key);
                            if(bucket_start > max_val) {
                                break;  // Bucket starts after range
                            }

                            // Deserialize and scan
                            Bucket bucket = Bucket::deserialize(data.iov_base, data.iov_len);
                            for(const auto& entry : bucket.entries) {
                                if(entry.first >= min_val && entry.first <= max_val) {
                                    result.add(entry.second);
                                }
                            }

                            rc = mdbx_cursor_get(cursor, &key, &data, MDBX_NEXT);
                            if(rc != MDBX_SUCCESS) {
                                break;
                            }
                        }
                    }

                    mdbx_cursor_close(cursor);
                    mdbx_txn_abort(txn);  // Read-only
                } catch(...) {
                    mdbx_txn_abort(txn);
                    throw;
                }
                return result;
            }

            // Check if ID has value in range [min_val, max_val] using Forward Index
            bool check_range(const std::string& field,
                             ndd::idInt id,
                             uint32_t min_val,
                             uint32_t max_val) {
                MDBX_txn* txn;
                int rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_RDONLY, &txn);
                if(rc != MDBX_SUCCESS) {
                    return false;
                }

                try {
                    std::string fwd_key_str = make_forward_key(field, id);
                    MDBX_val fwd_key{const_cast<char*>(fwd_key_str.data()), fwd_key_str.size()};
                    MDBX_val fwd_val;

                    rc = mdbx_get(txn, forward_dbi_, &fwd_key, &fwd_val);
                    bool match = false;
                    if(rc == MDBX_SUCCESS) {
                        uint32_t val;
                        std::memcpy(&val, fwd_val.iov_base, 4);
                        if(val >= min_val && val <= max_val) {
                            match = true;
                        }
                    }
                    mdbx_txn_abort(txn);
                    return match;
                } catch(...) {
                    mdbx_txn_abort(txn);
                    throw;
                }
            }

        private:
            void
            add_to_bucket(MDBX_txn* txn, const std::string& field, uint32_t value, ndd::idInt id) {
                // Find the bucket that starts <= value
                // We search for key = field:value. If exact match, good.
                // If not, we go to the previous key (MDBX_SET_RANGE returns >=, so we might need
                // prev)

                std::string target_key = make_bucket_key(field, value);
                MDBX_val key{const_cast<char*>(target_key.data()), target_key.size()};
                MDBX_val data;
                MDBX_cursor* cursor;
                mdbx_cursor_open(txn, inverted_dbi_, &cursor);

                int rc = mdbx_cursor_get(cursor, &key, &data, MDBX_SET_RANGE);

                bool found_bucket = false;
                std::string bucket_key_str;

                if(rc == MDBX_SUCCESS) {
                    // We found a key >= target.
                    // Check if it belongs to the same field
                    std::string found_key((char*)key.iov_base, key.iov_len);
                    if(found_key.rfind(field + ":", 0) == 0) {
                        // Same field.
                        // If found_key > target_key, we might need the PREVIOUS bucket
                        // unless this is the very first bucket and value < found_key's start
                        // (shouldn't happen if we maintain logic) Actually, we want the bucket
                        // where start_val <= value.

                        if(found_key > target_key) {
                            // Go back one
                            rc = mdbx_cursor_get(cursor, &key, &data, MDBX_PREV);
                            if(rc == MDBX_SUCCESS) {
                                std::string prev_key((char*)key.iov_base, key.iov_len);
                                if(prev_key.rfind(field + ":", 0) == 0) {
                                    // Found valid previous bucket
                                    bucket_key_str = prev_key;
                                    found_bucket = true;
                                }
                            }
                        } else {
                            // Exact match
                            bucket_key_str = found_key;
                            found_bucket = true;
                        }
                    } else {
                        // Found key is for next field, go back
                        rc = mdbx_cursor_get(cursor, &key, &data, MDBX_PREV);
                        if(rc == MDBX_SUCCESS) {
                            std::string prev_key((char*)key.iov_base, key.iov_len);
                            if(prev_key.rfind(field + ":", 0) == 0) {
                                bucket_key_str = prev_key;
                                found_bucket = true;
                            }
                        }
                    }
                } else if(rc == MDBX_NOTFOUND) {
                    // No key >= target. Try last key.
                    rc = mdbx_cursor_get(cursor, &key, &data, MDBX_LAST);
                    if(rc == MDBX_SUCCESS) {
                        std::string last_key((char*)key.iov_base, key.iov_len);
                        if(last_key.rfind(field + ":", 0) == 0) {
                            bucket_key_str = last_key;
                            found_bucket = true;
                        }
                    }
                }

                Bucket bucket;
                if(found_bucket) {
                    // Load existing bucket
                    // Note: cursor is already at the key if we didn't move it?
                    // Actually we moved it around. Let's just get by key to be safe/simple
                    MDBX_val b_key{const_cast<char*>(bucket_key_str.data()), bucket_key_str.size()};
                    mdbx_get(txn, inverted_dbi_, &b_key, &data);
                    bucket = Bucket::deserialize(data.iov_base, data.iov_len);
                } else {
                    // No bucket exists for this field yet. Create new one starting at 'value'
                    // Actually, let's start at 0 or min possible?
                    // Better: Start at 'value' for the first bucket.
                    bucket_key_str = make_bucket_key(field, value);
                }

                bucket.add(value, id);

                if(bucket.is_full()) {
                    // Split!
                    Bucket new_bucket = bucket.split();
                    uint32_t new_start = new_bucket.min_val();

                    // Save old bucket
                    auto bytes = bucket.serialize();
                    MDBX_val b_key{const_cast<char*>(bucket_key_str.data()), bucket_key_str.size()};
                    MDBX_val b_val{bytes.data(), bytes.size()};
                    mdbx_put(txn, inverted_dbi_, &b_key, &b_val, MDBX_put_flags_t(0));

                    // Save new bucket
                    std::string new_key_str = make_bucket_key(field, new_start);
                    auto new_bytes = new_bucket.serialize();
                    MDBX_val nb_key{const_cast<char*>(new_key_str.data()), new_key_str.size()};
                    MDBX_val nb_val{new_bytes.data(), new_bytes.size()};
                    mdbx_put(txn, inverted_dbi_, &nb_key, &nb_val, MDBX_put_flags_t(0));

                } else {
                    // Just save
                    auto bytes = bucket.serialize();
                    MDBX_val b_key{const_cast<char*>(bucket_key_str.data()), bucket_key_str.size()};
                    MDBX_val b_val{bytes.data(), bytes.size()};
                    mdbx_put(txn, inverted_dbi_, &b_key, &b_val, MDBX_put_flags_t(0));
                }

                mdbx_cursor_close(cursor);
            }

            void remove_from_bucket(MDBX_txn* txn,
                                    const std::string& field,
                                    uint32_t value,
                                    ndd::idInt id) {
                // Find bucket
                std::string target_key = make_bucket_key(field, value);
                MDBX_val key{const_cast<char*>(target_key.data()), target_key.size()};
                MDBX_val data;
                MDBX_cursor* cursor;
                mdbx_cursor_open(txn, inverted_dbi_, &cursor);

                int rc = mdbx_cursor_get(cursor, &key, &data, MDBX_SET_RANGE);

                std::string bucket_key_str;
                bool found = false;

                // Same logic as add_to_bucket to find the correct bucket
                if(rc == MDBX_SUCCESS) {
                    std::string found_key((char*)key.iov_base, key.iov_len);
                    if(found_key.rfind(field + ":", 0) == 0) {
                        if(found_key > target_key) {
                            rc = mdbx_cursor_get(cursor, &key, &data, MDBX_PREV);
                            if(rc == MDBX_SUCCESS) {
                                std::string prev_key((char*)key.iov_base, key.iov_len);
                                if(prev_key.rfind(field + ":", 0) == 0) {
                                    bucket_key_str = prev_key;
                                    found = true;
                                }
                            }
                        } else {
                            bucket_key_str = found_key;
                            found = true;
                        }
                    } else {
                        rc = mdbx_cursor_get(cursor, &key, &data, MDBX_PREV);
                        if(rc == MDBX_SUCCESS) {
                            std::string prev_key((char*)key.iov_base, key.iov_len);
                            if(prev_key.rfind(field + ":", 0) == 0) {
                                bucket_key_str = prev_key;
                                found = true;
                            }
                        }
                    }
                } else if(rc == MDBX_NOTFOUND) {
                    rc = mdbx_cursor_get(cursor, &key, &data, MDBX_LAST);
                    if(rc == MDBX_SUCCESS) {
                        std::string last_key((char*)key.iov_base, key.iov_len);
                        if(last_key.rfind(field + ":", 0) == 0) {
                            bucket_key_str = last_key;
                            found = true;
                        }
                    }
                }

                if(found) {
                    // Reload data to be sure
                    MDBX_val b_key{const_cast<char*>(bucket_key_str.data()), bucket_key_str.size()};
                    mdbx_get(txn, inverted_dbi_, &b_key, &data);

                    Bucket bucket = Bucket::deserialize(data.iov_base, data.iov_len);
                    if(bucket.remove(id)) {
                        if(bucket.is_empty()) {
                            // Delete bucket
                            mdbx_del(txn, inverted_dbi_, &b_key, nullptr);
                        } else {
                            // Save updated bucket
                            auto bytes = bucket.serialize();
                            MDBX_val b_val{bytes.data(), bytes.size()};
                            mdbx_put(txn, inverted_dbi_, &b_key, &b_val, MDBX_put_flags_t(0));
                        }
                    }
                }

                mdbx_cursor_close(cursor);
            }
        };

    }  // namespace numeric
}  // namespace ndd