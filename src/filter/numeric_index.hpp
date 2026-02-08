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
    namespace filter {

        // --- Sortable Key Utilities ---
        inline uint32_t float_to_sortable(float f) {
            uint32_t i;
            std::memcpy(&i, &f, sizeof(float));
            // IEEE 754: if sign bit set, flip all bits. Else flip just sign.
            // This makes negatives < positives order correctly.
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

        // --- Bucket Structure (Hybrid) ---
        struct Bucket {
            static constexpr size_t MAX_SIZE = 1024;
            static constexpr uint32_t MAX_DELTA = 65535;

            // Runtime only, not serialized in the payload
            uint32_t base_value = 0;

            // Data
            std::vector<uint16_t> deltas;
            std::vector<ndd::idInt> ids;
            ndd::RoaringBitmap summary_bitmap;

            bool is_dirty = false;

            // Helper to get actual value
            uint32_t get_value(size_t index) const {
                return base_value + deltas[index];
            }

            void add(uint32_t val, ndd::idInt id) {
                if (val < base_value) {
                     // Should not happen if Key logic is correct
                     throw std::runtime_error("Insert value < Base Value"); 
                }
                uint32_t delta_32 = val - base_value;
                if (delta_32 > MAX_DELTA) {
                    throw std::runtime_error("Delta overflow");
                }
                
                // Maintain sorted order by Value (Delta)
                uint16_t delta = static_cast<uint16_t>(delta_32);
                
                // Find insertion point
                auto it = std::lower_bound(deltas.begin(), deltas.end(), delta);
                size_t index = std::distance(deltas.begin(), it);

                deltas.insert(it, delta);
                ids.insert(ids.begin() + index, id);
                
                summary_bitmap.add(id);
                is_dirty = true;
            }

            bool remove(ndd::idInt id) {
                // Find index by ID (linear scan needed as ids are not sorted)
                for (size_t i = 0; i < ids.size(); ++i) {
                    if (ids[i] == id) {
                        ids.erase(ids.begin() + i);
                        deltas.erase(deltas.begin() + i);
                        
                        // Rebuild or update bitmap? Roaring remove is fast
                        summary_bitmap.remove(id);
                        is_dirty = true;
                        return true;
                    }
                }
                return false;
            }

            // Serialization Format:
            // [BitmapSize (4)]
            // [Bitmap Bytes]
            // [Count (2)]
            // [Deltas (Count * 2)]
            // [IDs (Count * sizeof(idInt))]
            std::vector<uint8_t> serialize() const {
                // Optimize bitmap
                const_cast<ndd::RoaringBitmap&>(summary_bitmap).runOptimize();
                
                size_t bm_size = summary_bitmap.getSizeInBytes();
                uint16_t count = static_cast<uint16_t>(ids.size());
                
                size_t total_size = 4 + bm_size + 2 + (count * 2) + (count * sizeof(ndd::idInt));
                std::vector<uint8_t> buffer(total_size);
                uint8_t* ptr = buffer.data();

                // 1. Bitmap Header
                uint32_t bm_size_32 = static_cast<uint32_t>(bm_size);
                std::memcpy(ptr, &bm_size_32, 4); ptr += 4;

                // 2. Bitmap Data
                if (bm_size > 0) {
                    summary_bitmap.write(reinterpret_cast<char*>(ptr));
                    ptr += bm_size;
                }

                // 3. Count
                std::memcpy(ptr, &count, 2); ptr += 2;

                // 4. Deltas
                if (count > 0) {
                    std::memcpy(ptr, deltas.data(), count * 2); ptr += count * 2;
                }

                // 5. IDs
                if (count > 0) {
                    std::memcpy(ptr, ids.data(), count * sizeof(ndd::idInt)); 
                }
                
                return buffer;
            }

            static Bucket deserialize(const void* data, size_t len, uint32_t base_val) {
                Bucket b;
                b.base_value = base_val;
                
                if (len < 6) return b; // Min valid size

                const uint8_t* ptr = static_cast<const uint8_t*>(data);
                const uint8_t* end = ptr + len;
                
                // 1. Bitmap Size
                uint32_t bm_size;
                std::memcpy(&bm_size, ptr, 4); ptr += 4;

                if (ptr + bm_size > end) {
                    throw std::runtime_error("Bucket corrupt: invalid bitmap size");
                }

                // 2. Bitmap
                if (bm_size > 0) {
                   b.summary_bitmap = ndd::RoaringBitmap::read(reinterpret_cast<const char*>(ptr));
                   ptr += bm_size;
                }

                if (ptr + 2 > end) throw std::runtime_error("Bucket corrupt: truncated count");

                // 3. Count
                uint16_t count;
                std::memcpy(&count, ptr, 2); ptr += 2;

                // 4. Deltas & IDs
                if (count > 0) {
                    size_t delta_size = count * 2;
                    size_t id_size = count * sizeof(ndd::idInt);
                    
                    if (ptr + delta_size + id_size > end) {
                         throw std::runtime_error("Bucket corrupt: truncated Data");
                    }

                    b.deltas.resize(count);
                    std::memcpy(b.deltas.data(), ptr, delta_size); ptr += delta_size;

                    b.ids.resize(count);
                    std::memcpy(b.ids.data(), ptr, id_size); 
                }
                
                return b;
            }

            // Fast access to just the bitmap (for middle buckets)
            static ndd::RoaringBitmap read_summary_bitmap(const void* data, size_t len) {
               const uint8_t* ptr = static_cast<const uint8_t*>(data);
               uint32_t bm_size;
               std::memcpy(&bm_size, ptr, 4); ptr += 4;
               if(bm_size == 0) return ndd::RoaringBitmap();
               return ndd::RoaringBitmap::read(reinterpret_cast<const char*>(ptr));
            }

            bool is_full() const { return ids.size() >= MAX_SIZE; }
            bool is_empty() const { return ids.empty(); }
        };

        class NumericIndex {
        private:
            MDBX_env* env_;
            MDBX_dbi forward_dbi_;   // ID -> Value (Field:ID -> Value)
            MDBX_dbi inverted_dbi_;  // BucketKey -> BucketBlob

            std::string make_forward_key(const std::string& field, ndd::idInt id) {
                return field + ":" + std::to_string(id);
            }

            // Key Format: [Field]:[BigEndian_BaseValue]
            std::string make_bucket_key(const std::string& field, uint32_t start_val) {
                uint32_t be_val = 0;
#if defined(__GNUC__) || defined(__clang__)
                be_val = __builtin_bswap32(start_val);
#else
                be_val = ((start_val >> 24) & 0xff) | ((start_val << 8) & 0xff0000)
                         | ((start_val >> 8) & 0xff00) | ((start_val << 24) & 0xff000000);
#endif
                std::string key = field + ":";
                key.append((char*)&be_val, 4);
                return key;
            }

            uint32_t parse_bucket_key_val(const std::string& key) {
                if (key.size() < 4) return 0;
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
            NumericIndex(MDBX_env* env) : env_(env) {
                MDBX_txn* txn;
                if (mdbx_txn_begin(env_, nullptr, MDBX_TXN_READWRITE, &txn) == MDBX_SUCCESS) {
                    mdbx_dbi_open(txn, "numeric_forward", MDBX_CREATE, &forward_dbi_);
                    mdbx_dbi_open(txn, "numeric_inverted", MDBX_CREATE, &inverted_dbi_);
                    mdbx_txn_commit(txn);
                }
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

            void remove(const std::string& field, ndd::idInt id) {
                MDBX_txn* txn;
                mdbx_txn_begin(env_, nullptr, MDBX_TXN_READWRITE, &txn);
                try {
                    std::string fwd_key_str = make_forward_key(field, id);
                    MDBX_val fwd_key{const_cast<char*>(fwd_key_str.data()), fwd_key_str.size()};
                    MDBX_val fwd_val;

                    if(mdbx_get(txn, forward_dbi_, &fwd_key, &fwd_val) == MDBX_SUCCESS) {
                        uint32_t old_val;
                        std::memcpy(&old_val, fwd_val.iov_base, sizeof(uint32_t));
                        remove_from_buckets(txn, field, old_val, id);
                        mdbx_del(txn, forward_dbi_, &fwd_key, nullptr);
                    }

                    mdbx_txn_commit(txn);
                } catch(...) {
                    mdbx_txn_abort(txn);
                    throw;
                }
            }

        private:
            void put_internal(MDBX_txn* txn, const std::string& field, ndd::idInt id, uint32_t value) {
                // 1. Check Forward Index
                std::string fwd_key_str = make_forward_key(field, id);
                MDBX_val fwd_key{const_cast<char*>(fwd_key_str.data()), fwd_key_str.size()};
                MDBX_val fwd_val;

                if (mdbx_get(txn, forward_dbi_, &fwd_key, &fwd_val) == MDBX_SUCCESS) {
                    uint32_t old_val;
                    std::memcpy(&old_val, fwd_val.iov_base, 4);
                    if (old_val == value) return;
                    remove_from_buckets(txn, field, old_val, id);
                }

                // 2. Update Forward
                MDBX_val new_val_data{&value, sizeof(uint32_t)};
                mdbx_put(txn, forward_dbi_, &fwd_key, &new_val_data, MDBX_UPSERT);

                // 3. Add to Inverted Buckets
                add_to_buckets(txn, field, value, id);
            }

            void remove_from_buckets(MDBX_txn* txn, const std::string& field, uint32_t value, ndd::idInt id) {
                // Find bucket
                std::string bkey_str = make_bucket_key(field, value);
                MDBX_val key{const_cast<char*>(bkey_str.data()), bkey_str.size()};
                MDBX_val data;
                MDBX_cursor* cursor;
                mdbx_cursor_open(txn, inverted_dbi_, &cursor);

                // Scan backward to find bucket covering 'value'
                int rc = mdbx_cursor_get(cursor, &key, &data, MDBX_SET_RANGE);
                
                // Logic to find correct bucket:
                std::string found_key;

                if (rc == MDBX_SUCCESS) {
                    found_key = std::string((char*)key.iov_base, key.iov_len);
                    // Check if we are in right field & range
                    if (found_key.rfind(field + ":", 0) != 0 || parse_bucket_key_val(found_key) > value) {
                            rc = mdbx_cursor_get(cursor, &key, &data, MDBX_PREV);
                    }
                } else if (rc == MDBX_NOTFOUND) {
                   rc = mdbx_cursor_get(cursor, &key, &data, MDBX_LAST);
                }

                // Should be at correct bucket now
                if (rc == MDBX_SUCCESS) {
                     found_key = std::string((char*)key.iov_base, key.iov_len);
                     if (found_key.rfind(field + ":", 0) == 0) {
                         uint32_t bucket_base = parse_bucket_key_val(found_key);
                         if (value >= bucket_base) {
                             Bucket b = Bucket::deserialize(data.iov_base, data.iov_len, bucket_base);
                             if (b.remove(id)) {
                                 // Save back or Delete if empty
                                 if (b.is_empty()) {
                                     mdbx_cursor_del(cursor, static_cast<MDBX_put_flags_t>(0));
                                 } else {
                                     auto bytes = b.serialize();
                                     MDBX_val new_data{bytes.data(), bytes.size()};
                                     mdbx_cursor_put(cursor, &key, &new_data, MDBX_CURRENT);
                                 }
                             }
                         }
                     }
                }
                mdbx_cursor_close(cursor);
            }

            void add_to_buckets(MDBX_txn* txn, const std::string& field, uint32_t value, ndd::idInt id) {
                MDBX_cursor* cursor;
                mdbx_cursor_open(txn, inverted_dbi_, &cursor);

                // Find candidate bucket
                std::string search_key = make_bucket_key(field, value);
                MDBX_val key{const_cast<char*>(search_key.data()), search_key.size()};
                MDBX_val data;

                int rc = mdbx_cursor_get(cursor, &key, &data, MDBX_SET_RANGE);
                
                bool create_new = false;
                std::string target_key_str;
                uint32_t target_base = 0;

                // Move logic to find predecessor
                if (rc == MDBX_SUCCESS) {
                     std::string found_key((char*)key.iov_base, key.iov_len);
                     if (found_key.rfind(field + ":", 0) != 0 || parse_bucket_key_val(found_key) > value) {
                         rc = mdbx_cursor_get(cursor, &key, &data, MDBX_PREV);
                     }
                } else {
                    rc = mdbx_cursor_get(cursor, &key, &data, MDBX_LAST);
                }

                if (rc == MDBX_SUCCESS) {
                    std::string found_key((char*)key.iov_base, key.iov_len);
                    if (found_key.rfind(field + ":", 0) == 0) {
                        target_base = parse_bucket_key_val(found_key);
                        // Check range condition
                        if (value >= target_base && (static_cast<uint64_t>(value) - target_base) <= Bucket::MAX_DELTA) {
                             target_key_str = found_key;
                        } else {
                            create_new = true;
                        }
                    } else {
                        create_new = true;
                    }
                } else {
                    create_new = true;
                }

                if (create_new) {
                    // Create new bucket at exact value
                    Bucket b;
                    b.base_value = value;
                    b.add(value, id);
                    auto bytes = b.serialize();
                    
                    target_key_str = make_bucket_key(field, value);
                    MDBX_val k{const_cast<char*>(target_key_str.data()), target_key_str.size()};
                    MDBX_val v{bytes.data(), bytes.size()};
                    mdbx_put(txn, inverted_dbi_, &k, &v, MDBX_UPSERT);
                    
                } else {
                    // Update existing
                    // We must re-fetch current key/data because cursor move might have updated key/data
                     MDBX_val k{const_cast<char*>(target_key_str.data()), target_key_str.size()};
                     MDBX_val v;
                     if(mdbx_cursor_get(cursor, &k, &v, MDBX_SET) != MDBX_SUCCESS) {
                         // Should not happen if logic is correct
                         throw std::runtime_error("Cursor sync fail");
                     }

                    Bucket b = Bucket::deserialize(v.iov_base, v.iov_len, target_base);
                    
                    // Capacity Check
                    if (b.ids.size() >= Bucket::MAX_SIZE) {
                         // SPLIT LOGIC
                         // Sort is maintained by arrays. 
                         // "Slide Split": Scan right from median
                         size_t mid_idx = b.ids.size() / 2;
                         
                         // Ensure we don't split a group of identical values
                         size_t probe_right = mid_idx;
                         while (probe_right < b.deltas.size() && probe_right > 0 && b.deltas[probe_right] == b.deltas[probe_right - 1]) {
                             probe_right++;
                         }

                         if (probe_right < b.deltas.size()) {
                             mid_idx = probe_right;
                         } else {
                             // Fallback: Try scanning left
                             size_t probe_left = mid_idx;
                             while (probe_left > 0 && b.deltas[probe_left] == b.deltas[probe_left - 1]) {
                                 probe_left--;
                             }
                             
                             if (probe_left > 0) {
                                 mid_idx = probe_left;
                             } else {
                                 // All identical
                                 mid_idx = b.deltas.size();
                             }
                         }
                         
                         // If we hit end, we can't split by value uniqueness
                         if (mid_idx == b.deltas.size()) {
                             // Fallback: Just append (overfill) or implement logic to handle identicals.
                             // For now: Append
                             b.add(value, id);
                             auto bytes = b.serialize();
                             MDBX_val k2{const_cast<char*>(target_key_str.data()), target_key_str.size()};
                             MDBX_val v2{bytes.data(), bytes.size()};
                             mdbx_cursor_put(cursor, &k2, &v2, MDBX_CURRENT);
                             mdbx_cursor_close(cursor);
                             return;
                         }

                         // Standard Slide Split
                         Bucket right_b;
                         right_b.base_value = b.base_value + b.deltas[mid_idx]; // New base
                         
                         // Move entries
                         for(size_t i=mid_idx; i<b.deltas.size(); ++i) {
                             right_b.add(b.base_value + b.deltas[i], b.ids[i]);
                         }
                         
                         // Truncate left
                         b.deltas.resize(mid_idx);
                         b.ids.resize(mid_idx);
                         // Rebuild left bitmap
                         b.summary_bitmap = ndd::RoaringBitmap();
                         for(auto pid : b.ids) b.summary_bitmap.add(pid);

                         // Now add new value to correct bucket
                         if (value >= right_b.base_value) {
                             right_b.add(value, id);
                         } else {
                             // If value < right, goes to left. 
                             // But wait, split point was determined by existing items.
                             // If new value is >= base+split_delta, it goes right.
                             // BUT we just cleared right from b.
                             // Correct logic:
                             b.add(value, id); // Add to left if it fits range (logic handles delta)
                             // Oh wait, if we added to left, we might overflow again or break order? 
                             // Simply: Check which bucket covers it.
                             // Left covers [Base, RightBase-1]
                             // Right covers [RightBase, ...]
                         }

                         // Save Left
                         auto left_bytes = b.serialize();
                         MDBX_val left_v{left_bytes.data(), left_bytes.size()};
                         MDBX_val left_k{const_cast<char*>(target_key_str.data()), target_key_str.size()};
                         mdbx_cursor_put(cursor, &left_k, &left_v, MDBX_CURRENT);

                         // Save Right
                         auto right_bytes = right_b.serialize();
                         std::string right_k_str = make_bucket_key(field, right_b.base_value);
                         MDBX_val right_k{const_cast<char*>(right_k_str.data()), right_k_str.size()};
                         MDBX_val right_v{right_bytes.data(), right_bytes.size()};
                         
                         // Use put for new key
                         mdbx_put(txn, inverted_dbi_, &right_k, &right_v, MDBX_UPSERT);

                    } else {
                        // Normal Insert
                        b.add(value, id);
                        auto bytes = b.serialize();
                        MDBX_val new_data{bytes.data(), bytes.size()};
                        
                        // Use cursor put to update current
                         mdbx_cursor_put(cursor, &k, &new_data, MDBX_CURRENT);
                    }
                }
                mdbx_cursor_close(cursor);
            }

        public:
            ndd::RoaringBitmap range(const std::string& field, uint32_t min_val, uint32_t max_val) {
                ndd::RoaringBitmap result;
                MDBX_txn* txn;
                if (mdbx_txn_begin(env_, nullptr, MDBX_TXN_RDONLY, &txn) != MDBX_SUCCESS) return result;

                MDBX_cursor* cursor;
                mdbx_cursor_open(txn, inverted_dbi_, &cursor);

                // 1. Find Start Bucket
                std::string start_k = make_bucket_key(field, min_val);
                MDBX_val key{const_cast<char*>(start_k.data()), start_k.size()};
                MDBX_val data;

                int rc = mdbx_cursor_get(cursor, &key, &data, MDBX_SET_RANGE);
                if (rc == MDBX_SUCCESS) {
                    // Check if we need to back up
                     std::string fkey((char*)key.iov_base, key.iov_len);
                     if (fkey.rfind(field + ":", 0) != 0 || parse_bucket_key_val(fkey) > min_val) {
                         // Check prev
                         MDBX_val p_key = key; 
                         MDBX_val p_data;
                         if (mdbx_cursor_get(cursor, &p_key, &p_data, MDBX_PREV) == MDBX_SUCCESS) {
                              std::string pkey_str((char*)p_key.iov_base, p_key.iov_len);
                              if (pkey_str.rfind(field + ":", 0) == 0) {
                                  // Prev is valid start
                                  key = p_key; data = p_data;
                                  rc = MDBX_SUCCESS;
                              }
                         }
                     }
                } else if (rc == MDBX_NOTFOUND) {
                     rc = mdbx_cursor_get(cursor, &key, &data, MDBX_LAST);
                     if (rc == MDBX_SUCCESS && data.iov_len > 0) {
                         std::string fkey((char*)key.iov_base, key.iov_len);
                         if (fkey.rfind(field + ":", 0) == 0) {
                             rc = MDBX_SUCCESS;
                         } else {
                             rc = MDBX_NOTFOUND;
                         }
                     } else {
                         rc = MDBX_NOTFOUND;
                     }
                }

                // Iterate forward
                while (rc == MDBX_SUCCESS) {
                    std::string cur_key((char*)key.iov_base, key.iov_len);
                    if (cur_key.rfind(field + ":", 0) != 0) break; // End of field

                    uint32_t bucket_base = parse_bucket_key_val(cur_key);
                    
                    if (bucket_base > max_val) break; // Past the end

                    // Peek Strategy:
                    // If bucket_base >= min_val, we know the start is covered.
                    // If we could know NEXT bucket start, we'd know overlap.
                    // Since we iterate, we can be greedy on read.
                    
                    // For now, always deserialize. 
                    // Potential optimization: Read only bitmap if we are "deep" in the range. 
                    // e.g. min_val=10, max_val=100. Bucket=20.
                    // If bucket=20. Next Bucket=30.
                    // Then Bucket 20 covers [20..30).
                    // Range [10..100] covers [20..30] fully.
                    // So we need lookahead. 
                    
                    // Simple logic without lookahead:
                    // Just read full bucket. It's 8KB max (2 pages). 
                    // It's fast unless we have millions of buckets.
                    
                    Bucket b = Bucket::deserialize(data.iov_base, data.iov_len, bucket_base);
                    
                    if (b.ids.empty()) {
                        rc = mdbx_cursor_get(cursor, &key, &data, MDBX_NEXT);
                        continue;
                    }

                    uint32_t b_min = b.get_value(0);
                    uint32_t b_max = b.get_value(b.ids.size()-1);

                    if (b_min >= min_val && b_max <= max_val) {
                         // Full overlap
                         result |= b.summary_bitmap;
                    } else {
                        // Partial overlap
                         for(size_t i=0; i<b.ids.size(); ++i) {
                             uint32_t v = b.get_value(i);
                             if (v >= min_val && v <= max_val) {
                                 result.add(b.ids[i]);
                             }
                         }
                    }

                    rc = mdbx_cursor_get(cursor, &key, &data, MDBX_NEXT);
                }

                mdbx_cursor_close(cursor);
                mdbx_txn_abort(txn);
                return result;
            }

            bool check_range(const std::string& field, ndd::idInt id, uint32_t min_val, uint32_t max_val) {
                MDBX_txn* txn;
                if(mdbx_txn_begin(env_, nullptr, MDBX_TXN_RDONLY, &txn) != MDBX_SUCCESS) return false;
                
                std::string fwd_key_str = make_forward_key(field, id);
                MDBX_val fwd_key{const_cast<char*>(fwd_key_str.data()), fwd_key_str.size()};
                MDBX_val fwd_val;
                
                bool match = false;
                if(mdbx_get(txn, forward_dbi_, &fwd_key, &fwd_val) == MDBX_SUCCESS) {
                    uint32_t val;
                    std::memcpy(&val, fwd_val.iov_base, 4);
                    if(val >= min_val && val <= max_val) match = true;
                }
                
                mdbx_txn_abort(txn);
                return match;
            }
        };

    } // namespace filter
} // namespace ndd
