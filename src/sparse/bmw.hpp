#pragma once

/**
 * This code implements Block-Max WAND search index using MDBX.
 * This algorithm is an optimization of the WAND (Weak AND) algorithm
 * used to skip large portions of the index that cannot possibly rank in
 * the top-K results.
 *
 * It is designed for high performance retrieval of sparse vector spaces.
 */

#include <vector>
#include <unordered_map>
#include <memory>
#include <optional>
#include <algorithm>
#include <queue>
#include <cstring>
#include <atomic>
#include <thread>
#include <shared_mutex>
#include <limits>
#include <cmath>

#if defined(__x86_64__) || defined(_M_X64)
#    include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#    include <arm_neon.h>
#endif

#include "mdbx/mdbx.h"
#include "../utils/log.hpp"
#include "../core/types.hpp"

#include "sparse_vector.hpp"

namespace ndd {

#pragma pack(push, 1)
    struct BlockIdx {
        ndd::idInt start_doc_id;
        float block_max_value;

        BlockIdx() = default;
        BlockIdx(ndd::idInt start, float max_val) :
            start_doc_id(start),
            block_max_value(max_val) {}
    };

    // Block header for term_blocks data
    struct BlockHeader {
        uint8_t version = 3;      // Version 3: SoA layout, uint8 values (quantized)
        uint8_t diff_bits = 16;   // 16, 32, or 64 bit doc diffs. Default to 16 for compression.
        uint16_t n = 0;           // total stored (incl. tombstones)
        uint16_t live_count = 0;  // nonzero entries
        uint16_t padding = 0;     // explicit padding
        float block_max_value = 0.0f;  // max value in block (for WAND)
        uint32_t alignment_pad = 0;    // Ensure 16-byte alignment for payload

        static constexpr size_t HEADER_SIZE = 16;
    };

    // Entry in a block (In-memory representation)
    struct BlockEntry {
        ndd::idInt doc_diff;  // difference from block start_doc_id
        float value;          // stored as float in memory, quantized to uint8 on disk

        BlockEntry() = default;
        BlockEntry(ndd::idInt diff, float val) :
            doc_diff(diff),
            value(val) {}

        bool operator<(const BlockEntry& other) const { return doc_diff < other.doc_diff; }
    };

#pragma pack(pop)

    // BMW search candidate
    struct BMWCandidate {
        ndd::idInt doc_id;
        float score;

        BMWCandidate(ndd::idInt id, float s) :
            doc_id(id),
            score(s) {}

        bool operator<(const BMWCandidate& other) const {
            return score > other.score;  // Min-heap (lowest scores first)
        }
    };

    class BMWIndex {
    public:
        static constexpr uint8_t CURRENT_VERSION = 3;

        BMWIndex(MDBX_env* env, size_t vocab_size) :
            env_(env),
            vocab_size_(vocab_size) {}

        ~BMWIndex() = default;

        // Initialize databases
        bool initialize() {
            std::unique_lock<std::shared_mutex> lock(mutex_);

            // Open LMDB databases
            MDBX_txn* txn;
            int rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_READWRITE, &txn);
            if(rc != 0) {
                LOG_ERROR("Failed to begin transaction: " << mdbx_strerror(rc));
                return false;
            }

            // Create term_blocks database
            rc = mdbx_dbi_open(txn, "term_blocks", MDBX_CREATE, &term_blocks_dbi_);
            if(rc != 0) {
                LOG_ERROR("Failed to open term_blocks database: " << mdbx_strerror(rc));
                mdbx_txn_abort(txn);
                return false;
            }

            // Create term_blocks_index database
            rc = mdbx_dbi_open(txn, "term_blocks_index", MDBX_CREATE, &term_blocks_index_dbi_);
            if(rc != 0) {
                LOG_ERROR("Failed to open term_blocks_index database: " << mdbx_strerror(rc));
                mdbx_txn_abort(txn);
                return false;
            }

            rc = mdbx_txn_commit(txn);
            if(rc != 0) {
                LOG_ERROR("Failed to commit initialization transaction: " << mdbx_strerror(rc));
                return false;
            }

            // Load existing term blocks index
            return loadTermBlocksIndex();
        }

        // Document management
        bool addDocument(ndd::idInt doc_id, const SparseVector& vec) {
            return addDocumentsBatch({{doc_id, vec}});
        }

        bool addDocumentsBatch(const std::vector<std::pair<ndd::idInt, SparseVector>>& docs) {
            if(docs.empty()) {
                return true;
            }

            std::unique_lock<std::shared_mutex> lock(mutex_);

            MDBX_txn* txn;
            int rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_READWRITE, &txn);
            if(rc != 0) {
                LOG_ERROR("Failed to begin transaction: " << mdbx_strerror(rc));
                return false;
            }

            try {
                if(!addDocumentsBatchInternal(txn, docs)) {
                    mdbx_txn_abort(txn);
                    return false;
                }

                rc = mdbx_txn_commit(txn);
                if(rc != 0) {
                    LOG_ERROR("Failed to commit initialization transaction: " << mdbx_strerror(rc));
                    return false;
                }

                return true;
            } catch(const std::exception& e) {
                LOG_ERROR("Failed to add documents batch: " << e.what());
                mdbx_txn_abort(txn);
                return false;
            }
        }

        bool removeDocument(ndd::idInt doc_id, const SparseVector& vec) {
            std::unique_lock<std::shared_mutex> lock(mutex_);
            MDBX_txn* txn;
            int rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_READWRITE, &txn);
            if(rc != 0) {
                return false;
            }

            try {
                if(!removeDocumentInternal(txn, doc_id, vec)) {
                    mdbx_txn_abort(txn);
                    return false;
                }
                return mdbx_txn_commit(txn) == 0;
            } catch(const std::exception& e) {
                LOG_ERROR("Failed to remove document: " << e.what());
                mdbx_txn_abort(txn);
                return false;
            }
        }

        bool updateDocument(ndd::idInt doc_id,
                            const SparseVector& old_vec,
                            const SparseVector& new_vec) {
            std::unique_lock<std::shared_mutex> lock(mutex_);
            MDBX_txn* txn;
            int rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_READWRITE, &txn);
            if(rc != 0) {
                return false;
            }

            try {
                if(!removeDocumentInternal(txn, doc_id, old_vec)) {
                    mdbx_txn_abort(txn);
                    return false;
                }

                if(!addDocumentsBatchInternal(txn, {{doc_id, new_vec}})) {
                    mdbx_txn_abort(txn);
                    return false;
                }

                return mdbx_txn_commit(txn) == 0;
            } catch(const std::exception& e) {
                LOG_ERROR("Failed to update document: " << e.what());
                mdbx_txn_abort(txn);
                return false;
            }
        }

        // Batch operations - Removed empty implementation

        // Transaction-aware methods for external orchestration
        bool addDocumentsBatch(MDBX_txn* txn,
                               const std::vector<std::pair<ndd::idInt, SparseVector>>& docs) {
            std::unique_lock<std::shared_mutex> lock(mutex_);
            return addDocumentsBatchInternal(txn, docs);
        }

        bool removeDocument(MDBX_txn* txn, ndd::idInt doc_id, const SparseVector& vec) {
            std::unique_lock<std::shared_mutex> lock(mutex_);
            return removeDocumentInternal(txn, doc_id, vec);
        }

        // Search using BMW algorithm (DAAT)
        std::vector<std::pair<ndd::idInt, float>> search(const SparseVector& query, size_t k) {
            if(query.empty() || k == 0) {
                return {};
            }

            std::shared_lock<std::shared_mutex> lock(mutex_);

            // Start Read Transaction
            MDBX_txn* txn;
            int rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_RDONLY, &txn);
            if(rc != 0) {
                LOG_ERROR("Failed to begin search transaction: " << mdbx_strerror(rc));
                return {};
            }

            // Initialize iterators for all query terms
            // Use vector for storage to ensure pointer stability (reserve is key)
            std::vector<BlockIterator> iterators_storage;
            iterators_storage.reserve(query.indices.size());

            // Pointers for sorting
            std::vector<BlockIterator*> iterators;
            iterators.reserve(query.indices.size());

            for(size_t i = 0; i < query.indices.size(); ++i) {
                auto it = term_blocks_index_.find(query.indices[i]);
                if(it != term_blocks_index_.end()) {
                    iterators_storage.emplace_back(
                            query.indices[i], query.values[i], &it->second, this, txn);
                }
            }

            // Initialize pointers
            for(auto& it : iterators_storage) {
                iterators.push_back(&it);
            }

            if(iterators.empty()) {
                mdbx_txn_abort(txn);
                return {};
            }

            std::priority_queue<BMWCandidate> top_k;
            float threshold = 0.0f;

            // Helper to sort iterators by current doc ID
            auto sort_iterators = [&]() {
                std::sort(
                        iterators.begin(), iterators.end(), [](BlockIterator* a, BlockIterator* b) {
                            return a->current_doc_id < b->current_doc_id;
                        });
            };

            sort_iterators();

            while(true) {
                // Remove exhausted iterators
                while(!iterators.empty()
                      && iterators.back()->current_doc_id
                                 == std::numeric_limits<ndd::idInt>::max()) {
                    iterators.pop_back();
                }

                if(iterators.empty()) {
                    break;
                }

                // WAND/BMW logic
                float upper_bound_sum = 0.0f;
                size_t pivot_idx = 0;
                bool found_pivot = false;

                // Find pivot term
                for(size_t i = 0; i < iterators.size(); ++i) {
                    upper_bound_sum += iterators[i]->upperBound();
                    if(upper_bound_sum > threshold) {
                        pivot_idx = i;
                        found_pivot = true;
                        break;
                    }
                }

                if(!found_pivot) {
                    // No document can exceed threshold
                    break;
                }

                ndd::idInt pivot_doc_id = iterators[pivot_idx]->current_doc_id;

                if(iterators[0]->current_doc_id == pivot_doc_id) {
                    // Pivot is the first iterator, so we have a candidate
                    iterators[0]->advance(pivot_doc_id);  // Should be no-op
                    float score = iterators[0]->current_score * iterators[0]->term_weight;
                    iterators[0]->next();

                    // Check other terms
                    for(size_t i = 1; i < iterators.size(); ++i) {
                        iterators[i]->advance(pivot_doc_id);
                        if(iterators[i]->current_doc_id == pivot_doc_id) {
                            score += iterators[i]->current_score * iterators[i]->term_weight;
                            iterators[i]->next();
                        }
                    }

                    if(top_k.size() < k) {
                        top_k.emplace(pivot_doc_id, score);
                        if(top_k.size() == k) {
                            threshold = top_k.top().score;
                        }
                    } else if(score > threshold) {
                        top_k.pop();
                        top_k.emplace(pivot_doc_id, score);
                        threshold = top_k.top().score;
                    }
                } else {
                    // Pivot is further ahead, skip predecessors
                    for(size_t i = 0; i < pivot_idx; ++i) {
                        iterators[i]->advance(pivot_doc_id);
                    }
                }

                sort_iterators();
            }

            // Clean up
            mdbx_txn_abort(txn);

            // Extract results
            std::vector<std::pair<ndd::idInt, float>> results;
            results.reserve(top_k.size());

            while(!top_k.empty()) {
                const auto& candidate = top_k.top();
                results.emplace_back(candidate.doc_id, candidate.score);
                top_k.pop();
            }

            std::reverse(results.begin(), results.end());
            return results;
        }

        // Maintenance
        // Functions removed as they were empty/unused placeholders

        bool splitBlock(MDBX_txn* txn, uint32_t term_id, ndd::idInt start_doc_id) {
            auto& blocks = term_blocks_index_[term_id];
            auto block_it = findBlockIterator(blocks, start_doc_id);

            // Verify we found the correct block
            if(block_it == blocks.end() || block_it->start_doc_id != start_doc_id) {
                return false;
            }

            auto entries = loadBlock(txn, term_id, start_doc_id);
            if(entries.size() <= MAX_BLOCK_SIZE) {
                return true;
            }

            // Split point (middle)
            size_t split_idx = entries.size() / 2;
            ndd::idInt new_start_doc_id = start_doc_id + entries[split_idx].doc_diff;

            std::vector<BlockEntry> first_half(entries.begin(), entries.begin() + split_idx);
            std::vector<BlockEntry> second_half;
            second_half.reserve(entries.size() - split_idx);

            // Re-calculate diffs for second half
            ndd::idInt base_diff = entries[split_idx].doc_diff;
            for(size_t i = split_idx; i < entries.size(); ++i) {
                second_half.emplace_back(entries[i].doc_diff - base_diff, entries[i].value);
            }

            // Calculate max values for both new blocks
            float max1 = 0.0f, max2 = 0.0f;

            for(const auto& e : first_half) {
                if(e.value > 0) {
                    max1 = std::max(max1, e.value);
                }
            }
            for(const auto& e : second_half) {
                if(e.value > 0) {
                    max2 = std::max(max2, e.value);
                }
            }

            // Update first block metadata
            block_it->block_max_value = max1;

            // Save first block
            BlockHeader h1;
            h1.n = static_cast<uint16_t>(first_half.size());
            h1.live_count = 0;
            for(const auto& e : first_half) {
                if(e.value > 0) {
                    h1.live_count++;
                }
            }
            h1.block_max_value = max1;

            if(!saveBlock(txn, term_id, start_doc_id, first_half, h1)) {
                return false;
            }

            // Insert second block metadata
            // Note: block_it might be invalidated by insert, so calculate index first
            size_t idx = std::distance(blocks.begin(), block_it);
            blocks.insert(blocks.begin() + idx + 1, BlockIdx(new_start_doc_id, max2));

            // Save second block
            BlockHeader h2;
            h2.n = static_cast<uint16_t>(second_half.size());
            h2.live_count = 0;
            for(const auto& e : second_half) {
                if(e.value > 0) {
                    h2.live_count++;
                }
            }
            h2.block_max_value = max2;

            if(!saveBlock(txn, term_id, new_start_doc_id, second_half, h2)) {
                return false;
            }

            return true;
        }

        // Statistics
        size_t getTermCount() const {
            std::shared_lock<std::shared_mutex> lock(mutex_);
            return term_blocks_index_.size();
        }

        size_t getBlockCount() const {
            std::shared_lock<std::shared_mutex> lock(mutex_);
            size_t total = 0;
            for(const auto& [term_id, blocks] : term_blocks_index_) {
                total += blocks.size();
            }
            return total;
        }

        size_t getVocabSize() const { return vocab_size_; }

    private:
        std::vector<BlockIdx>::iterator findBlockIterator(std::vector<BlockIdx>& blocks,
                                                          ndd::idInt doc_id) {
            auto it = std::upper_bound(blocks.begin(),
                                       blocks.end(),
                                       doc_id,
                                       [](ndd::idInt doc_id, const BlockIdx& block) {
                                           return doc_id < block.start_doc_id;
                                       });

            if(it == blocks.begin()) {
                return it;
            }
            return it - 1;
        }

        std::vector<BlockIdx>::const_iterator findBlockIterator(const std::vector<BlockIdx>& blocks,
                                                                ndd::idInt doc_id) const {
            auto it = std::upper_bound(blocks.begin(),
                                       blocks.end(),
                                       doc_id,
                                       [](ndd::idInt doc_id, const BlockIdx& block) {
                                           return doc_id < block.start_doc_id;
                                       });

            if(it == blocks.begin()) {
                return it;
            }
            return it - 1;
        }

        // Helper to dequantize
        static inline float dequantize(uint8_t val, float max_val) {
            return (static_cast<float>(val) * (1.0f / 255.0f)) * max_val;
        }

        // Helper struct for getReadOnlyBlock return value
        struct BlockView {
            const void* doc_diffs;  // Can be uint16_t* or uint32_t*
            const uint8_t* values;
            size_t count;
            uint8_t diff_bits;  // 16 or 32
        };

        struct BlockIterator {
            uint32_t term_id;
            float term_weight;
            const std::vector<BlockIdx>* blocks;
            size_t current_block_idx;

            // SoA pointers
            const void* doc_diffs_ptr = nullptr;  // Can be u16 or u32
            const uint8_t* values_ptr = nullptr;
            size_t block_data_size = 0;
            uint8_t diff_bits = 32;

            size_t current_entry_idx;
            ndd::idInt current_doc_id;
            float current_score;
            BMWIndex* index;
            MDBX_txn* txn;

            BlockIterator(uint32_t tid,
                          float weight,
                          const std::vector<BlockIdx>* blks,
                          BMWIndex* idx,
                          MDBX_txn* t) :
                term_id(tid),
                term_weight(weight),
                blocks(blks),
                current_block_idx(0),
                current_entry_idx(0),
                current_doc_id(std::numeric_limits<ndd::idInt>::max()),
                current_score(0.0f),
                index(idx),
                txn(t) {
                if(blocks && !blocks->empty()) {
                    loadCurrentBlock();
                }
            }

            void loadCurrentBlock() {
                if(current_block_idx >= blocks->size()) {
                    current_doc_id = std::numeric_limits<ndd::idInt>::max();
                    return;
                }
                const auto& block_meta = (*blocks)[current_block_idx];
                auto view = index->getReadOnlyBlock(txn, term_id, block_meta.start_doc_id);
                doc_diffs_ptr = view.doc_diffs;
                values_ptr = view.values;
                block_data_size = view.count;
                diff_bits = view.diff_bits;
                current_entry_idx = 0;
                advanceToNextLive();
            }

            inline void advanceToNextLive() {
                // Branch prediction will handle diff_bits effectively (constant per block)
                if(diff_bits == 16) {
                    advanceToNextLive16();
                } else {
                    advanceToNextLive32();
                }
            }

            inline void advanceToNextLive16() {
                auto diff_ptr = static_cast<const uint16_t*>(doc_diffs_ptr);

                // Fast path: check if current entry is already live
                if(current_entry_idx < block_data_size && values_ptr[current_entry_idx] > 0) {
                    const auto& block_meta = (*blocks)[current_block_idx];
                    current_doc_id = block_meta.start_doc_id + diff_ptr[current_entry_idx];
                    current_score =
                            dequantize(values_ptr[current_entry_idx], block_meta.block_max_value);
                    return;
                }

                current_entry_idx =
                        index->findNextLiveSIMD(values_ptr, block_data_size, current_entry_idx);

                if(current_entry_idx < block_data_size) {
                    const auto& block_meta = (*blocks)[current_block_idx];
                    current_doc_id = block_meta.start_doc_id + diff_ptr[current_entry_idx];
                    current_score =
                            dequantize(values_ptr[current_entry_idx], block_meta.block_max_value);
                    return;
                }
                // Block exhausted
                current_block_idx++;
                loadCurrentBlock();
            }

            inline void advanceToNextLive32() {
                auto diff_ptr = static_cast<const ndd::idInt*>(doc_diffs_ptr);

                if(current_entry_idx < block_data_size && values_ptr[current_entry_idx] > 0) {
                    const auto& block_meta = (*blocks)[current_block_idx];
                    current_doc_id = block_meta.start_doc_id + diff_ptr[current_entry_idx];
                    current_score =
                            dequantize(values_ptr[current_entry_idx], block_meta.block_max_value);
                    return;
                }

                current_entry_idx =
                        index->findNextLiveSIMD(values_ptr, block_data_size, current_entry_idx);

                if(current_entry_idx < block_data_size) {
                    const auto& block_meta = (*blocks)[current_block_idx];
                    current_doc_id = block_meta.start_doc_id + diff_ptr[current_entry_idx];
                    current_score =
                            dequantize(values_ptr[current_entry_idx], block_meta.block_max_value);
                    return;
                }
                current_block_idx++;
                loadCurrentBlock();
            }

            inline void next() {
                current_entry_idx++;
                // Inline the check to avoid function call overhead in tight loops
                if(diff_bits == 16) {
                    advanceToNextLive16();
                } else {
                    advanceToNextLive32();
                }
            }

            void advance(ndd::idInt target_doc_id) {
                if(current_doc_id >= target_doc_id) {
                    return;
                }

                // Dispatch to specialized implementation
                if(diff_bits == 16) {
                    advance16(target_doc_id);
                } else {
                    advanceGeneric(target_doc_id);
                }
            }

            // Specialized advance for 16-bit
            void advance16(ndd::idInt target_doc_id) {
                // Optimize Block Skipping logic (Same as before)
                if(current_block_idx < blocks->size()) {
                    if(current_block_idx + 1 < blocks->size()
                       && (*blocks)[current_block_idx + 1].start_doc_id < target_doc_id) {
                        auto it = std::upper_bound(blocks->begin() + current_block_idx,
                                                   blocks->end(),
                                                   target_doc_id,
                                                   [](ndd::idInt id, const BlockIdx& b) {
                                                       return id < b.start_doc_id;
                                                   });
                        size_t next_idx = std::distance(blocks->begin(), it);
                        if(next_idx > 0) {
                            current_block_idx = next_idx - 1;
                            // Reset state for new block
                            current_entry_idx = 0;
                            doc_diffs_ptr = nullptr;
                            block_data_size = 0;
                            // DO NOT recursively call loadCurrentBlock -> advance(), just break to
                            // reload below
                        }
                    }
                }

                if(block_data_size == 0) {
                    loadCurrentBlock();
                    // If diff_bits changed (unlikely but possible), dispatch again
                    if(diff_bits != 16) {
                        advance(target_doc_id);
                        return;
                    }
                }

                if(current_block_idx >= blocks->size()) {
                    return;
                }

                const auto& block_meta = (*blocks)[current_block_idx];
                if(target_doc_id > block_meta.start_doc_id) {
                    ndd::idInt diff = target_doc_id - block_meta.start_doc_id;
                    // If diff > 65535, we know it's not in this 16-bit block
                    if(diff > 65535) {
                        current_entry_idx = block_data_size;
                    } else {
                        current_entry_idx = index->findEntryIndexSIMD16(
                                static_cast<const uint16_t*>(doc_diffs_ptr),
                                block_data_size,
                                current_entry_idx,
                                static_cast<uint16_t>(diff));
                    }
                    advanceToNextLive16();
                }
            }

            // Specialized advance for non-16-bit blocks (32-bit or Generic)
            void advanceGeneric(ndd::idInt target_doc_id) {
                // Optimize Block Skipping logic
                if(current_block_idx < blocks->size()) {
                    if(current_block_idx + 1 < blocks->size()
                       && (*blocks)[current_block_idx + 1].start_doc_id < target_doc_id) {
                        auto it = std::upper_bound(blocks->begin() + current_block_idx,
                                                   blocks->end(),
                                                   target_doc_id,
                                                   [](ndd::idInt id, const BlockIdx& b) {
                                                       return id < b.start_doc_id;
                                                   });
                        size_t next_idx = std::distance(blocks->begin(), it);
                        if(next_idx > 0) {
                            current_block_idx = next_idx - 1;
                            current_entry_idx = 0;
                            doc_diffs_ptr = nullptr;
                            block_data_size = 0;
                        }
                    }
                }

                if(block_data_size == 0) {
                    loadCurrentBlock();
                    if(diff_bits == 16) {
                        advance(target_doc_id);
                        return;
                    }
                }

                if(current_block_idx >= blocks->size()) {
                    return;
                }

                const auto& block_meta = (*blocks)[current_block_idx];
                if(target_doc_id > block_meta.start_doc_id) {
                    ndd::idInt diff = target_doc_id - block_meta.start_doc_id;

#ifdef NDD_USE_64BIT_IDS
                    // 64-bit ID build: Supports 32-bit compressed blocks and 64-bit full blocks.
                    // Note: 16-bit blocks are handled by advance16/SIMD16.
                    current_entry_idx = index->findEntryIndexGeneric(
                            doc_diffs_ptr, block_data_size, current_entry_idx, diff, diff_bits);
#else
                    // 32-bit ID build: Supports only 32-bit blocks here.
                    // diff fits in 32 bits.
                    current_entry_idx =
                            index->findEntryIndexSIMD32(static_cast<const uint32_t*>(doc_diffs_ptr),
                                                        block_data_size,
                                                        current_entry_idx,
                                                        static_cast<uint32_t>(diff));
#endif

                    advanceToNextLive32();
                }
            }

            float upperBound() const {
                if(current_block_idx >= blocks->size()) {
                    return 0.0f;
                }
                return term_weight * (*blocks)[current_block_idx].block_max_value;
            }
        };

        MDBX_env* env_;
        MDBX_dbi term_blocks_dbi_;
        MDBX_dbi term_blocks_index_dbi_;
        size_t vocab_size_;

        // In-memory cache of term block indices
        std::unordered_map<uint32_t, std::vector<BlockIdx>> term_blocks_index_;
        mutable std::shared_mutex mutex_;

        // Block management constants
        static constexpr size_t MAX_BLOCK_SIZE = 128;
        static constexpr size_t SPLIT_THRESHOLD = 160;

        // Optimized SIMD search for 16-bit diffs
        size_t findEntryIndexSIMD16(const uint16_t* doc_diffs,
                                    size_t size,
                                    size_t start_idx,
                                    uint16_t target_diff) {
            size_t idx = start_idx;

#if defined(USE_AVX512)
            const size_t simd_width = 32;
            __m512i target_vec = _mm512_set1_epi16(static_cast<short>(target_diff));

            while(idx + simd_width <= size) {
                __m512i data_vec = _mm512_loadu_si512(doc_diffs + idx);
                __mmask32 mask = _mm512_cmpge_epi16_mask(data_vec, target_vec);

                if(mask != 0) {
                    return idx + __builtin_ctz(mask);
                }
                idx += simd_width;
            }
#elif defined(USE_AVX2)
            const size_t simd_width = 16;
            __m256i target_vec = _mm256_set1_epi16(static_cast<short>(target_diff));

            while(idx + simd_width <= size) {
                if(doc_diffs[idx + simd_width - 1] < target_diff) {
                    idx += simd_width;
                    continue;
                }
                __m256i data_vec =
                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(doc_diffs + idx));
                __m256i offset = _mm256_set1_epi16(static_cast<short>(0x8000));
                __m256i data_biased = _mm256_add_epi16(data_vec, offset);
                __m256i target_biased = _mm256_add_epi16(target_vec, offset);

                __m256i lt = _mm256_cmpgt_epi16(target_biased, data_biased);
                int mask = _mm256_movemask_epi8(lt);

                if(mask != -1) {
                    return idx + (__builtin_ctz(~mask) / 2);
                }
                idx += simd_width;
            }
#elif defined(USE_SVE2)
            svbool_t pg = svwhilelt_b16(idx, size);
            svuint16_t target_vec = svdup_u16(target_diff);

            while(svptest_any(svptrue_b16(), pg)) {
                svuint16_t data_vec = svld1_u16(pg, doc_diffs + idx);
                svbool_t cmp = svcmpge_u16(pg, data_vec, target_vec);

                if(svptest_any(pg, cmp)) {
                    svbool_t before_match = svbrkb_z(pg, cmp);
                    uint64_t count = svcntp_b16(pg, before_match);
                    return idx + count;
                }
                idx += svcnth();
                pg = svwhilelt_b16(idx, size);
            }
            return idx;
#elif defined(USE_NEON)
            const size_t simd_width = 8;
            uint16x8_t target_vec = vdupq_n_u16(target_diff);

            while(idx + simd_width <= size) {
                uint16x8_t data_vec = vld1q_u16(doc_diffs + idx);
                uint16x8_t cmp = vcgeq_u16(data_vec, target_vec);

                // Check if any element is >= target (result of vcgeq is all 1s if true)
                if(vmaxvq_u16(cmp) != 0) {
                    for(size_t i = 0; i < simd_width; ++i) {
                        if(doc_diffs[idx + i] >= target_diff) {
                            return idx + i;
                        }
                    }
                }
                idx += simd_width;
            }
#endif

            // Scalar fallback
            while(idx < size && doc_diffs[idx] < target_diff) {
                idx++;
            }
            return idx;
        }

        // Optimized SIMD search for 32-bit diffs
        size_t findEntryIndexSIMD32(const uint32_t* doc_diffs,
                                    size_t size,
                                    size_t start_idx,
                                    uint32_t target_diff) {
            size_t idx = start_idx;

#if defined(USE_AVX512)
            const size_t simd_width = 16;
            __m512i target_vec = _mm512_set1_epi32(static_cast<int>(target_diff));

            while(idx + simd_width <= size) {
                __m512i data_vec = _mm512_loadu_si512(doc_diffs + idx);
                __mmask16 mask = _mm512_cmpge_epu32_mask(data_vec, target_vec);

                if(mask != 0) {
                    return idx + __builtin_ctz(mask);
                }
                idx += simd_width;
            }
#elif defined(USE_AVX2)
            const size_t simd_width = 8;
            __m256i target_vec = _mm256_set1_epi32(static_cast<int>(target_diff));

            while(idx + simd_width <= size) {
                __builtin_prefetch(doc_diffs + idx + 32);
                if(doc_diffs[idx + simd_width - 1] < target_diff) {
                    idx += simd_width;
                    continue;
                }

                __m256i data_vec =
                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(doc_diffs + idx));
                // unsigned comparison using max: a >= b iff max(a,b) == a
                __m256i max_vec = _mm256_max_epu32(data_vec, target_vec);
                __m256i cmp = _mm256_cmpeq_epi32(max_vec, data_vec);

                int mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));
                if(mask != 0) {
                    return idx + __builtin_ctz(mask);
                }
                idx += simd_width;
            }
#elif defined(USE_SVE2)
            svbool_t pg = svwhilelt_b32(idx, size);
            svuint32_t target_vec = svdup_u32(target_diff);

            while(svptest_any(svptrue_b32(), pg)) {
                svuint32_t data_vec = svld1_u32(pg, doc_diffs + idx);
                svbool_t cmp = svcmpge_u32(pg, data_vec, target_vec);

                if(svptest_any(pg, cmp)) {
                    svbool_t before_match = svbrkb_z(pg, cmp);
                    uint64_t count = svcntp_b32(pg, before_match);
                    return idx + count;
                }
                idx += svcntw();
                pg = svwhilelt_b32(idx, size);
            }
            return idx;
#elif defined(USE_NEON)
            const size_t simd_width = 4;
            uint32x4_t target_vec = vdupq_n_u32(target_diff);

            while(idx + simd_width <= size) {
                uint32x4_t data_vec = vld1q_u32(doc_diffs + idx);
                uint32x4_t cmp = vcgeq_u32(data_vec, target_vec);

                // Check if any bit is expected (vcgeq returns all 1s or 0s)
                if(vmaxvq_u32(cmp) != 0) {
                    for(size_t i = 0; i < simd_width; ++i) {
                        if(doc_diffs[idx + i] >= target_diff) {
                            return idx + i;
                        }
                    }
                }
                idx += simd_width;
            }
#endif

            // Scalar fallback
            while(idx < size && doc_diffs[idx] < target_diff) {
                idx++;
            }
            return idx;
        }

        size_t findEntryIndexGeneric(const void* doc_diffs,
                                     size_t size,
                                     size_t start_idx,
                                     ndd::idInt target_diff,
                                     uint8_t bits) {
// In 64-bit mode, we might encounter 32-bit compressed blocks or 64-bit blocks
#ifdef NDD_USE_64BIT_IDS
            if(bits == 32) {
                if(target_diff > 0xFFFFFFFFULL) {
                    return size;  // Optimization: target exceeds max possible value in 32-bit block
                }
                return findEntryIndexSIMD32(static_cast<const uint32_t*>(doc_diffs),
                                            size,
                                            start_idx,
                                            static_cast<uint32_t>(target_diff));
            } else {
                size_t idx = start_idx;
                const uint64_t* ptr = static_cast<const uint64_t*>(doc_diffs);
                while(idx < size && ptr[idx] < target_diff) {
                    idx++;
                }
                return idx;
            }
#else
            // In 32-bit mode, we only expect 32-bit blocks here (16-bit handled by SIMD16)
            return findEntryIndexSIMD32(static_cast<const uint32_t*>(doc_diffs),
                                        size,
                                        start_idx,
                                        static_cast<uint32_t>(target_diff));
#endif
        }

        // Find next non-zero value (live entry)
        size_t findNextLiveSIMD(const uint8_t* values, size_t size, size_t start_idx) {
            size_t idx = start_idx;

#if defined(USE_AVX512)
            const size_t simd_width = 64;
            __m512i zero_vec = _mm512_setzero_si512();

            while(idx + simd_width <= size) {
                __m512i data_vec = _mm512_loadu_si512(values + idx);
                __mmask64 mask = _mm512_cmpneq_epu8_mask(data_vec, zero_vec);

                if(mask != 0) {
                    return idx + __builtin_ctzll(mask);
                }
                idx += simd_width;
            }
#elif defined(USE_AVX2)
            const size_t simd_width = 32;
            __m256i zero_vec = _mm256_setzero_si256();

            while(idx + simd_width <= size) {
                __m256i data_vec =
                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values + idx));
                __m256i cmp = _mm256_cmpeq_epi8(data_vec, zero_vec);
                int mask = _mm256_movemask_epi8(cmp);  // 1 = zero, 0 = non-zero

                // If all 1s (mask 0xFFFFFFFF), then all zeros -> continue
                if(static_cast<uint32_t>(mask) != 0xFFFFFFFF) {
                    // ~mask has 1s where non-zero exists
                    return idx + __builtin_ctz(~mask);
                }
                idx += simd_width;
            }
#elif defined(USE_NEON)
            const size_t simd_width = 16;
            uint8x16_t zero_vec = vdupq_n_u8(0);

            while(idx + simd_width <= size) {
                uint8x16_t data_vec = vld1q_u8(values + idx);
                uint8x16_t cmp = vceqq_u8(data_vec, zero_vec);

                // Check if any element is NOT zero (cmp is 0x00 for non-zero, 0xFF for zero)
                // If all are zero, cmp is all 0xFF. vminvq_u8 will be 0xFF.
                // If any is non-zero, cmp has a 0x00. vminvq_u8 will be 0x00.
                if(vminvq_u8(cmp) == 0) {
                    for(size_t i = 0; i < simd_width; ++i) {
                        if(values[idx + i] != 0) {
                            return idx + i;
                        }
                    }
                }
                idx += simd_width;
            }
#elif defined(USE_SVE2)
            svbool_t pg = svwhilelt_b8(idx, size);
            while(svptest_any(svptrue_b8(), pg)) {
                svuint8_t data_vec = svld1_u8(pg, values + idx);
                svbool_t cmp = svcmpne_n_u8(pg, data_vec, 0);  // Not equal to 0

                if(svptest_any(pg, cmp)) {
                    svbool_t before_match = svbrkb_z(pg, cmp);
                    return idx + svcntp_b8(pg, before_match);
                }
                idx += svcntb();
                pg = svwhilelt_b8(idx, size);
            }
            return idx;
#endif

            while(idx < size) {
                if(values[idx] != 0) {
                    return idx;
                }
                idx++;
            }
            return idx;
        }

        bool loadTermBlocksIndex() {
            MDBX_txn* txn;
            int rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_RDONLY, &txn);
            if(rc != 0) {
                LOG_ERROR("Failed to begin transaction for loading index: " << mdbx_strerror(rc));
                return false;
            }

            MDBX_cursor* cursor;
            rc = mdbx_cursor_open(txn, term_blocks_index_dbi_, &cursor);
            if(rc != 0) {
                mdbx_txn_abort(txn);
                return false;
            }

            MDBX_val key, data;
            rc = mdbx_cursor_get(cursor, &key, &data, MDBX_FIRST);
            while(rc == MDBX_SUCCESS) {
                if(key.iov_len == sizeof(uint32_t)) {
                    uint32_t term_id;
                    std::memcpy(&term_id, key.iov_base, sizeof(uint32_t));

                    size_t count = data.iov_len / sizeof(BlockIdx);
                    std::vector<BlockIdx> blocks(count);
                    std::memcpy(blocks.data(), data.iov_base, data.iov_len);

                    term_blocks_index_[term_id] = std::move(blocks);
                }
                rc = mdbx_cursor_get(cursor, &key, &data, MDBX_NEXT);
            }

            mdbx_cursor_close(cursor);
            mdbx_txn_abort(txn);
            return true;
        }

        bool removeDocumentInternal(MDBX_txn* txn, ndd::idInt doc_id, const SparseVector& vec) {
            for(size_t i = 0; i < vec.indices.size(); ++i) {
                uint32_t term_id = vec.indices[i];
                if(!removeFromBlock(txn, term_id, doc_id)) {
                    // Ignore errors
                }
            }
            return true;
        }

        bool
        addDocumentsBatchInternal(MDBX_txn* txn,
                                  const std::vector<std::pair<ndd::idInt, SparseVector>>& docs) {
            // Group updates by term_id
            std::unordered_map<uint32_t, std::vector<std::pair<ndd::idInt, float>>> term_updates;
            for(const auto& [doc_id, sparse_vec] : docs) {
                for(size_t i = 0; i < sparse_vec.indices.size(); ++i) {
                    term_updates[sparse_vec.indices[i]].emplace_back(doc_id, sparse_vec.values[i]);
                }
            }

            // Process each term
            for(auto& [term_id, updates] : term_updates) {
                // Sort by doc_id to access blocks sequentially
                std::sort(updates.begin(), updates.end());

                for(const auto& [doc_id, value] : updates) {
                    if(!addToBlock(txn, term_id, doc_id, value)) {
                        LOG_ERROR("Failed to add doc " << doc_id << " term " << term_id
                                                       << " to block");
                        return false;
                    }
                }

                // Save index structure for this term after all updates
                if(!saveTermIndex(txn, term_id)) {
                    return false;
                }
            }
            return true;
        }

        // Save the index structure (block list) for a single term
        bool saveTermIndex(MDBX_txn* txn, uint32_t term_id) {
            auto it = term_blocks_index_.find(term_id);
            if(it == term_blocks_index_.end()) {
                return true;
            }

            const auto& blocks = it->second;

            MDBX_val key, data;

            // Safe to cast away const for API as MDBX copies
            key.iov_base = const_cast<void*>(static_cast<const void*>(&term_id));
            key.iov_len = sizeof(uint32_t);

            data.iov_base = const_cast<void*>(static_cast<const void*>(blocks.data()));
            data.iov_len = blocks.size() * sizeof(BlockIdx);

            int rc = mdbx_put(txn, term_blocks_index_dbi_, &key, &data, MDBX_UPSERT);
            if(rc != 0) {
                LOG_ERROR("Failed to save term index for term " << term_id << ": "
                                                                << mdbx_strerror(rc));
                return false;
            }
            return true;
        }

        std::vector<BlockEntry>
        loadBlock(MDBX_txn* txn, uint32_t term_id, ndd::idInt start_doc_id) {
            // Zero-copy key creation
            struct {
                uint32_t t;
                ndd::idInt d;
            } __attribute__((packed)) key_struct;
            key_struct.t = term_id;
            key_struct.d = start_doc_id;

            MDBX_val key;
            key.iov_base = &key_struct;
            key.iov_len = sizeof(key_struct);

            MDBX_val data;
            int rc = mdbx_get(txn, term_blocks_dbi_, &key, &data);

            std::vector<BlockEntry> entries;
            if(rc == MDBX_SUCCESS && data.iov_len >= sizeof(BlockHeader)) {
                const BlockHeader* header = reinterpret_cast<const BlockHeader*>(data.iov_base);
                size_t n = header->n;

                entries.resize(n);
                const uint8_t* ptr =
                        static_cast<const uint8_t*>(data.iov_base) + sizeof(BlockHeader);

                if(header->version == 3) {
                    // Determine pointer locations based on diff_bits
                    const void* diff_ptr = ptr;
                    const uint8_t* val_ptr;

                    if(header->diff_bits == 16) {
                        val_ptr = ptr + n * sizeof(uint16_t);
                        const uint16_t* diffs = static_cast<const uint16_t*>(diff_ptr);
                        for(size_t i = 0; i < n; ++i) {
                            entries[i].doc_diff = diffs[i];
                            entries[i].value = dequantize(val_ptr[i], header->block_max_value);
                        }
                    } else if(header->diff_bits == 32) {
                        val_ptr = ptr + n * sizeof(uint32_t);
                        const uint32_t* diffs = static_cast<const uint32_t*>(diff_ptr);
                        for(size_t i = 0; i < n; ++i) {
                            entries[i].doc_diff = diffs[i];
                            entries[i].value = dequantize(val_ptr[i], header->block_max_value);
                        }
                    }
#ifdef NDD_USE_64BIT_IDS
                    else if(header->diff_bits == 64) {
                        val_ptr = ptr + n * sizeof(uint64_t);
                        const uint64_t* diffs = static_cast<const uint64_t*>(diff_ptr);
                        for(size_t i = 0; i < n; ++i) {
                            entries[i].doc_diff = diffs[i];
                            entries[i].value = dequantize(val_ptr[i], header->block_max_value);
                        }
                    }
#endif
                    else {
                        LOG_ERROR("Unsupported block diff_bits: " << (int)header->diff_bits);
                    }
                } else {
                    LOG_ERROR("Unsupported block version: " << (int)header->version);
                }
            }
            return entries;
        }

        // Helper for uint8 quantization
        static inline uint8_t quantize(float val, float max_val) {
            if(max_val <= 1e-9f) {
                return 0;
            }
            float scaled = (val / max_val) * 255.0f;
            if(scaled > 255.0f) {
                return 255;
            }
            return static_cast<uint8_t>(scaled);
        }

        bool saveBlock(MDBX_txn* txn,
                       uint32_t term_id,
                       ndd::idInt start_doc_id,
                       const std::vector<BlockEntry>& entries,
                       BlockHeader& header) {
            // Zero-copy key creation
            struct {
                uint32_t t;
                ndd::idInt d;
            } __attribute__((packed)) key_struct;
            key_struct.t = term_id;
            key_struct.d = start_doc_id;

            MDBX_val key;
            key.iov_base = &key_struct;
            key.iov_len = sizeof(key_struct);

            size_t n = entries.size();

            // Recalculate stats
            float max_val = 0.0f;
            ndd::idInt max_diff = 0;
            size_t live = 0;

            for(const auto& e : entries) {
                if(e.value > max_val) {
                    max_val = e.value;
                }
                if(e.doc_diff > max_diff) {
                    max_diff = e.doc_diff;
                }
                if(e.value > 1e-9f) {
                    live++;  // Approximate check for float > 0
                }
            }

            header.block_max_value = max_val;
            header.live_count = static_cast<uint16_t>(live);
            header.n = static_cast<uint16_t>(n);
            header.version = 3;
            header.alignment_pad = 0;

// Decide diff width (User requested 16-bit blocks when possible)
#ifdef NDD_USE_64BIT_IDS
            if(max_diff < 65536) {
                header.diff_bits = 16;
            } else if(max_diff < 4294967296ULL) {
                header.diff_bits = 32;
            } else {
                header.diff_bits = 64;
            }
#else
            if(max_diff < 65536) {
                header.diff_bits = 16;
            } else {
                header.diff_bits = 32;
            }
#endif

            size_t diff_size = header.diff_bits / 8;
            size_t total_size = sizeof(BlockHeader) + n * diff_size + n * sizeof(uint8_t);

            std::vector<uint8_t> buffer(total_size);

            // Copy header
            std::memcpy(buffer.data(), &header, sizeof(BlockHeader));

            uint8_t* ptr = buffer.data() + sizeof(BlockHeader);

            // Copy doc_diffs
            if(header.diff_bits == 16) {
                uint16_t* diffs = reinterpret_cast<uint16_t*>(ptr);
                for(size_t i = 0; i < n; ++i) {
                    diffs[i] = static_cast<uint16_t>(entries[i].doc_diff);
                }
                ptr += n * sizeof(uint16_t);
            } else if(header.diff_bits == 32) {
                uint32_t* diffs = reinterpret_cast<uint32_t*>(ptr);
                for(size_t i = 0; i < n; ++i) {
                    diffs[i] = static_cast<uint32_t>(entries[i].doc_diff);
                }
                ptr += n * sizeof(uint32_t);
            }
#ifdef NDD_USE_64BIT_IDS
            else {
                // 64-bit case
                uint64_t* diffs = reinterpret_cast<uint64_t*>(ptr);
                for(size_t i = 0; i < n; ++i) {
                    diffs[i] = static_cast<uint64_t>(entries[i].doc_diff);
                }
                ptr += n * sizeof(uint64_t);
            }
#endif

            // Copy values (Quantized)
            uint8_t* values = static_cast<uint8_t*>(ptr);
            for(size_t i = 0; i < n; ++i) {
                values[i] = quantize(entries[i].value, max_val);
            }

            MDBX_val data;
            data.iov_base = buffer.data();
            data.iov_len = buffer.size();

            int rc = mdbx_put(txn, term_blocks_dbi_, &key, &data, MDBX_UPSERT);
            return rc == 0;
        }

        // Returns pointer to block data valid for the duration of txn
        BlockView getReadOnlyBlock(MDBX_txn* txn, uint32_t term_id, ndd::idInt start_doc_id) {
            // Zero-copy key creation on stack
            struct {
                uint32_t t;
                ndd::idInt d;
            } __attribute__((packed)) key_struct;
            key_struct.t = term_id;
            key_struct.d = start_doc_id;

            MDBX_val key;
            key.iov_base = &key_struct;
            key.iov_len = sizeof(key_struct);

            MDBX_val data;
            int rc = mdbx_get(txn, term_blocks_dbi_, &key, &data);

            if(rc == MDBX_SUCCESS && data.iov_len >= sizeof(BlockHeader)) {
                const BlockHeader* header = reinterpret_cast<const BlockHeader*>(data.iov_base);

                // Validate size roughly (min size)
                if(data.iov_len < sizeof(BlockHeader) + header->n) {
                    return {nullptr, nullptr, 0, 0};
                }

                const uint8_t* ptr =
                        static_cast<const uint8_t*>(data.iov_base) + sizeof(BlockHeader);

                size_t diff_size =
                        (header->diff_bits == 16) ? sizeof(uint16_t) : sizeof(ndd::idInt);

                const void* doc_diffs = ptr;
                const uint8_t* values = ptr + header->n * diff_size;

                return {doc_diffs, values, header->n, header->diff_bits};
            }
            return {nullptr, nullptr, 0, 0};
        }

        ndd::idInt getBlockEndDocId(const std::vector<BlockIdx>& blocks, size_t block_idx) const {
            if(block_idx + 1 < blocks.size()) {
                return blocks[block_idx + 1].start_doc_id - 1;
            }
            return std::numeric_limits<ndd::idInt>::max();
        }

        bool addToBlock(MDBX_txn* txn, uint32_t term_id, ndd::idInt doc_id, float value) {
            // Get or create blocks for this term
            auto& blocks = term_blocks_index_[term_id];

            // Find the appropriate block
            auto block_it = findBlockIterator(blocks, doc_id);

            // Check if we need to split due to range (if > 65535, cannot fit in uint16 diff)
            // This is a constraint for 16-bit blocks. If we enable mix, we don't strict need to
            // check unless we want to force 16-bit.

            bool force_new_block = false;
            if(block_it != blocks.end() && block_it->start_doc_id <= doc_id) {
                if((doc_id - block_it->start_doc_id) >= 65536) {
                    force_new_block = true;
                }
            }

            if(block_it == blocks.end() || block_it->start_doc_id > doc_id || force_new_block) {
                // Need to create a new block

                // Insert into index list
                // If forcing new block, we insert AFTER block_it if block_it exists and is < doc_id
                // findBlockIterator returns iterator <= doc_id.

                std::vector<BlockIdx>::iterator insert_it;
                if(force_new_block && block_it != blocks.end()) {
                    insert_it = block_it + 1;
                } else {
                    insert_it = block_it;
                }

                blocks.insert(insert_it, BlockIdx(doc_id, value));

                // Create the actual block data
                std::vector<BlockEntry> entries;
                entries.emplace_back(0, value);  // doc_diff = 0

                BlockHeader header;
                // header fields set by saveBlock

                return saveBlock(txn, term_id, doc_id, entries, header);
            }

            // Add to existing block
            auto block_entries = loadBlock(txn, term_id, block_it->start_doc_id);

            ndd::idInt doc_diff = doc_id - block_it->start_doc_id;

            // Find insertion point (keep sorted by doc_diff)
            auto entry_it = std::lower_bound(
                    block_entries.begin(), block_entries.end(), BlockEntry(doc_diff, 0.0f));

            if(entry_it != block_entries.end() && entry_it->doc_diff == doc_diff) {
                // Update existing entry
                entry_it->value = value;
            } else {
                // Insert new entry
                block_entries.insert(entry_it, BlockEntry(doc_diff, value));
            }

            // Check if block needs splitting
            if(block_entries.size() > SPLIT_THRESHOLD) {
                return splitBlock(txn, term_id, block_it->start_doc_id);
            }

            BlockHeader header;
            // Fields set by saveBlock

            bool success = saveBlock(txn, term_id, block_it->start_doc_id, block_entries, header);
            if(success) {
                // Update index with possibly new max value
                if(header.block_max_value > block_it->block_max_value) {
                    block_it->block_max_value = header.block_max_value;  // Update cached max
                }
            }
            return success;
        }

        bool removeFromBlock(MDBX_txn* txn, uint32_t term_id, ndd::idInt doc_id) {
            auto it = term_blocks_index_.find(term_id);
            if(it == term_blocks_index_.end()) {
                return false;  // Term not found
            }

            auto& blocks = it->second;
            auto block_it = findBlockIterator(blocks, doc_id);

            if(block_it == blocks.end() || block_it->start_doc_id > doc_id) {
                return false;
            }

            // Logic similar to addToBlock. Load, modify, Save.
            // We do basic range check to avoid loading obviously wrong block
            if((doc_id - block_it->start_doc_id) > 200000) {  // Safety heuristic
                return false;
            }

            // Load block
            auto block_entries = loadBlock(txn, term_id, block_it->start_doc_id);
            ndd::idInt doc_diff = doc_id - block_it->start_doc_id;

            auto entry_it = std::lower_bound(
                    block_entries.begin(), block_entries.end(), BlockEntry(doc_diff, 0.0f));

            if(entry_it != block_entries.end() && entry_it->doc_diff == doc_diff) {
                entry_it->value = 0.0f;  // Mark as tombstone (0.0f)

                BlockHeader header;
                // Fields set by saveBlock
                return saveBlock(txn, term_id, block_it->start_doc_id, block_entries, header);
            }

            return false;
        }
    };

}  // namespace ndd