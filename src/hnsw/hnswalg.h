#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include "log.hpp"
#include "../utils/settings.hpp"
#include "../quant/dispatch.hpp"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <functional>
#include <sstream>
#include <iostream>
#include <thread>
#include <set>
#include <unordered_map>
#include <shared_mutex>
#include <memory>

namespace hnswlib {

    template <typename T> struct CompareByFirst {
        constexpr bool operator()(const T& p1, const T& p2) const noexcept {
            return p1.first < p2.first;
        }
    };
    template <typename T> struct CompareBySecond {
        constexpr bool operator()(const T& p1, const T& p2) const noexcept {
            return p2.first < p1.first;
        }
    };

    template <typename dist_t> class HierarchicalNSW : public AlgorithmInterface<dist_t> {
        using distance_type = std::pair<dist_t, idhInt>;
        using max_heap_pq = std::priority_queue<distance_type,
                                                std::vector<distance_type>,
                                                CompareByFirst<distance_type>>;
        using min_heap_pq = std::priority_queue<distance_type,
                                                std::vector<distance_type>,
                                                CompareBySecond<distance_type>>;
        using VectorFetcher = std::function<bool(idInt, uint8_t*)>;

    public:
        // Constructors and destructor
        HierarchicalNSW(SpaceInterface<dist_t>* s) {}
        // Used for loading an existing index
        HierarchicalNSW(const std::string& location, size_t max_elements = 0) :
            linkListLocks_(settings::MAX_LINK_LIST_LOCKS) {
            // Initilaize label lookup vector. Values will be filled in loadIndex
            loadIndex(location, max_elements);
        }
        // Used for creating a new index
        HierarchicalNSW(
                size_t max_elements,
                SpaceType space_type,
                size_t dimension,
                size_t M = settings::DEFAULT_M,
                size_t ef_construction = settings::DEFAULT_EF_CONSTRUCT,
                size_t random_seed = settings::RANDOM_SEED,
                ndd::quant::QuantizationLevel quant_level = ndd::quant::QuantizationLevel::INT8,
                int32_t checksum = -1) :
            maxElements_(max_elements),
            dimension_(dimension),
            space_type_(space_type),
            quant_level_(quant_level),
            M_(M),
            M0_(M * 2),
            maxM_(M_ + settings::MAX_EXTRA_NEIGHBORS),
            maxM0_(M0_ + 2 * settings::MAX_EXTRA_NEIGHBORS),
            efConstruction_(std::max(ef_construction, M_)),
            linkListLocks_(settings::MAX_LINK_LIST_LOCKS),
            checksum_(checksum),
            dataUpperLayer_(maxElements_),
            labelLookup_(maxElements_, INVALID_ID) {

            // Create appropriate space based on type
            space_ = std::unique_ptr<SpaceInterface<float>>(
                    createSpace<float>(space_type_, dimension_, quant_level_));
            data_size_ = space_->get_data_size();
            fstDistFunc_ = space_->get_dist_func();
            fstSimFunc_ = space_->get_sim_func();
            dist_func_param_ = space_->get_dist_func_param();
            LOG_DEBUG("Space initialized with data size: "
                      << data_size_ << ", dimension: " << dimension_
                      << ", quant_level: " << static_cast<int>(quant_level_));

#ifndef DISABLE_HYBRID_QUANTIZATION
            // Create INT8 space for upper layers
            space_int8_ = std::unique_ptr<SpaceInterface<float>>(createSpace<float>(
                    space_type_, dimension_, ndd::quant::QuantizationLevel::INT8));
            data_size_int8_ = space_int8_->get_data_size();
            fstSimFuncInt8_ = space_int8_->get_sim_func();
            dist_func_param_int8_ = space_int8_->get_dist_func_param();
            LOG_DEBUG("Space also initialized (Hybrid Quantization) with data size: "
                      << data_size_int8_);
#endif  //DISABLE_HYBRID_QUANTIZATION

            // M_ cannot be more than settings::MAX_M
            if(M_ > settings::MAX_M) {
                M_ = settings::MAX_M;
                M0_ = M_ * 2;
                maxM_ = M_ + settings::MAX_EXTRA_NEIGHBORS;
                maxM0_ = M0_ + settings::MAX_EXTRA_NEIGHBORS;
                LOG_DEBUG("Capping M parameter to settings::MAX_M" << settings::MAX_M);
            }
            //efConstruction cannot be more than MAX_EF_CONSTRUCT
            if(efConstruction_ > settings::MAX_EF_CONSTRUCT) {
                efConstruction_ = settings::MAX_EF_CONSTRUCT;
                LOG_DEBUG("Capping efConstruction to settings::MAX_EF_CONSTRUCT "
                          << settings::MAX_EF_CONSTRUCT);
            }

            level_generator_.seed(random_seed);
            update_probability_generator_.seed(random_seed + 1);

            // links will also store number of linked elements in the first element
            sizeLinksUpperLayers_ = sizeof(idhInt) + maxM_ * sizeof(idhInt);
            sizeLinksBaseLayer_ = sizeof(idhInt) + maxM0_ * sizeof(idhInt);
            // We are not storing the data in the level 0 memory, only the links and label
            sizeDataAtBaseLayer_ = sizeLinksBaseLayer_ + sizeof(flagInt) + sizeof(idInt);
            labelOffset_ = sizeLinksBaseLayer_ + sizeof(flagInt);

            dataBaseLayer_ = (char*)malloc(maxElements_ * sizeDataAtBaseLayer_);
            if(!dataBaseLayer_) {
                throw std::runtime_error(
                        "Unable to allocate "
                        + std::to_string((maxElements_ * sizeDataAtBaseLayer_) / KB) + " KB");
            }

            mult_ = 1 / log(1.0 * M_);

            visited_list_pool_ =
                    std::unique_ptr<VisitedListPool>(new VisitedListPool(1, maxElements_));
        }

        ~HierarchicalNSW() {
            LOG_DEBUG("HierarchicalNSW destructor called");
            if(dataBaseLayer_) {
                free(dataBaseLayer_);
            }
        }
        // Public getters and setters
        ndd::quant::QuantizationLevel getQuantLevel() const { return quant_level_; }
        int32_t getChecksum() const { return checksum_; }

        SpaceType getSpaceType() const { return space_type_; }
        std::string getSpaceTypeStr() const { return getSpaceTypeString(space_type_); }
        SpaceInterface<dist_t>* getSpace() const { return space_.get(); }
        size_t getDataSize() const { return data_size_; }
        void setVectorFetcher(VectorFetcher fetcher) { vector_fetcher_ = fetcher; }
        size_t getDimension() const { return dimension_; }
        size_t getM() const { return M_; }
        size_t getEfConstruction() const { return efConstruction_; }
        size_t getRemainingCapacity() const { return maxElements_ - curElementsCount_; }
        size_t getMaxElements() const { return maxElements_; }
        // Get active elements count
        size_t getElementsCount() const { return curElementsCount_ - deletedElementsCount_; }
        size_t getDeletedCount() const { return deletedElementsCount_; }
        std::string getElementStats() const {
            std::stringstream ss;
            ss << "Elements: " << curElementsCount_ << ", MaxLevel: " << maxLevel_
               << ", Deleted: " << deletedElementsCount_;
            return ss.str();
        }
        size_t getApproxSizeGB() const {
            size_t size = 0;

            // Level 0: links + flags + labels
            size += maxElements_ * sizeDataAtBaseLayer_;

            // Approximate level > 0 count
            size_t upper_layer_estimate = maxElements_ / M_;

#ifdef DISABLE_HYBRID_QUANTIZATION
            // Typical Quantization
            size += upper_layer_estimate * (data_size_ + sizeof(levelInt) + sizeLinksUpperLayers_);
#else
            // int 8 Quantization for upper layers
            size += upper_layer_estimate
                    * (data_size_int8_ + sizeof(levelInt) + sizeLinksUpperLayers_);
#endif  //DISABLE_HYBRID_QUANTIZATION

            return size / GB;  // GB
        }

        // Helper to quantize to INT8 (check DISABLE_HYBRID_QUANTIZATION)
        std::vector<uint8_t> quantizeToInt8(const void* datapoint) {
            // Use direct conversion from current level to INT8
            // This avoids double allocation and conversion (Base -> Float -> Int8)
            auto dispatch = ndd::quant::get_quantizer_dispatch(quant_level_);
            // datapoint is the raw stored format for the base layer
            return dispatch.quantize_to_int8(datapoint, dimension_);
        }

        // Cache management getters/setters
        // Removed as cache is managed externally

        std::vector<std::pair<dist_t, idInt>>
        searchKnn(const void* query_data,
                  size_t k,
                  size_t ef = settings::DEFAULT_EF_SEARCH,
                  BaseFilterFunctor* isIdAllowed = nullptr) const override {
            int x = 0;
            LOG_DEBUG("Inside searchKnn, element count: " << curElementsCount_);
            std::vector<std::pair<dist_t, idInt>> result;
            if(curElementsCount_ == 0) {
                return result;
            }
            LOG_DEBUG("Searching for k=" << k << " nearest neighbors");
            idhInt currObj = entryPoint_;
            dist_t curSim;

#ifdef DISABLE_HYBRID_QUANTIZATION
            std::vector<uint8_t> ep_vector(data_size_);
            if(!getDataByInternalId(currObj, maxLevel_, ep_vector.data())) {
                return result;
            }
            curSim = fstSimFunc_(query_data, ep_vector.data(), dist_func_param_);
            std::vector<uint8_t> candidate_vector(data_size_);
#else
            // Only convert to INT8 if we have upper layers to traverse
            std::vector<uint8_t> query_data_int8;
            if(maxLevel_ > 0) {
                query_data_int8 = const_cast<HierarchicalNSW*>(this)->quantizeToInt8(query_data);

                // Use direct pointer for upper layers
                const uint8_t* ep_data = getUpperLayerDataPtr(currObj);
                if(!ep_data) {
                    return result;
                }

                curSim = fstSimFuncInt8_(query_data_int8.data(), ep_data, dist_func_param_int8_);
            }
#endif  //DISABLE_HYBRID_QUANTIZATION

            dist_t s;
            // Upper layer traversal - greedy search
            for(levelInt level = maxLevel_; level > 0; level--) {
                bool changed = true;
                while(changed) {
                    changed = false;
                    // TODO - This is dead-locking the mutex
                    //std::unique_lock<std::shared_mutex> lock(linkListLocks_[currObj]);
                    idhInt* ll_cur = (idhInt*)get_linklist(currObj, level);
                    if(!ll_cur) {
                        continue;
                    }

                    int size = getListCount(ll_cur);
                    idhInt* data = (idhInt*)(ll_cur + 1);

                    for(int i = 0; i < size; i++) {
                        idhInt candidate = data[i];
                        if(candidate >= curElementsCount_) {
                            continue;
                        }

#ifdef DISABLE_HYBRID_QUANTIZATION
                        if(!getDataByInternalId(candidate, level, candidate_vector.data())) {
                            continue;
                        }
                        s = fstSimFunc_(query_data, candidate_vector.data(), dist_func_param_);
#else
                        const uint8_t* candidate_data = getUpperLayerDataPtr(candidate);
                        if(!candidate_data) {
                            continue;
                        }
                        s = fstSimFuncInt8_(
                                query_data_int8.data(), candidate_data, dist_func_param_int8_);
#endif  //DISABLE_HYBRID_QUANTIZATION

                        if(s > curSim) {
                            curSim = s;
                            currObj = candidate;
                            changed = true;
                        }
                    }
                }
            }

            std::vector<std::pair<dist_t, idhInt>> top_candidates;
            LOG_DEBUG("Starting search in level 0..current object " << currObj);
            if(deletedElementsCount_) {
                top_candidates = searchBaseLayer<false, true>(
                        currObj, query_data, 0, std::max(ef, k));  // Level 0 for final search
            } else {
                top_candidates = searchBaseLayer<false, false>(
                        currObj, query_data, 0, std::max(ef, k));  // Level 0 for final search
            }
            LOG_DEBUG("Search in level 0 completed. Found " << top_candidates.size()
                                                            << " candidates");

            // Get external labels and return k elements
            for(size_t i = 0; i < std::min(k, top_candidates.size()); ++i) {
                result.emplace_back(top_candidates[i].first,
                                    getExternalLabel(top_candidates[i].second));
            }
            return result;
        }

        void saveIndex(const std::string& location) override {
            // Lock the index so that addPoint and markDelete are not called
            std::unique_lock<std::shared_mutex> lock(index_lock_);

            std::ofstream output(location, std::ios::binary);

            // Save version (first 2 bytes)
            writeBinaryPOD(output, settings::INDEX_VERSION);

            // 8 byte space for future flags
            writeBinaryPOD(output, flags_);

            // Save index parameters
            writeBinaryPOD(output, checksum_);
            writeBinaryPOD(output, space_type_);
            writeBinaryPOD(output, dimension_);
            writeBinaryPOD(output, quant_level_);
            writeBinaryPOD(output, maxElements_);
            writeBinaryPOD(output, curElementsCount_);
            writeBinaryPOD(output, deletedElementsCount_);
            writeBinaryPOD(output, labelOffset_);
            writeBinaryPOD(output, maxLevel_);
            writeBinaryPOD(output, entryPoint_);
            writeBinaryPOD(output, M_);
            writeBinaryPOD(output, M0_);
            writeBinaryPOD(output, mult_);
            writeBinaryPOD(output, efConstruction_);

            // Save level 0 data
            output.write(dataBaseLayer_, maxElements_ * sizeDataAtBaseLayer_);
            // Marker to check alignment of data
            uint64_t upper_marker = 0xDEADBEEFDEADBEEF;
            writeBinaryPOD(output, upper_marker);

            levelInt level;
            size_t total_size;
            // Write upper layer data using sentinel-based stream
            for(size_t i = 0; i < dataUpperLayer_.size(); ++i) {
                if(!dataUpperLayer_[i]) {
                    continue;
                }

#ifdef DISABLE_HYBRID_QUANTIZATION
                level = *reinterpret_cast<levelInt*>(dataUpperLayer_[i].get() + data_size_);
                total_size = data_size_ + sizeof(levelInt) + level * sizeLinksUpperLayers_;
#else
                // Use data_size_int8_
                level = *reinterpret_cast<levelInt*>(dataUpperLayer_[i].get() + data_size_int8_);
                total_size = data_size_int8_ + sizeof(levelInt) + level * sizeLinksUpperLayers_;
#endif  //DISABLE_HYBRID_QUANTIZATION

                writeBinaryPOD(output, static_cast<idhInt>(i));  // write ID
                output.write(reinterpret_cast<char*>(dataUpperLayer_[i].get()),
                             total_size);  // write blob
            }

            // Sentinel to mark end
            idhInt sentinel = INVALID_ID;
            writeBinaryPOD(output, sentinel);
            output.close();
        }

        void loadIndex(const std::string& location, size_t maxElements_i = 0) {
            std::ifstream input(location, std::ios::binary);
            if(!input.is_open()) {
                throw std::runtime_error("Cannot open file");
            }

            // Read version
            uint16_t version;
            readBinaryPOD(input, version);
            if(version != settings::INDEX_VERSION) {
                LOG_DEBUG("Index version mismatch. Expected: " << settings::INDEX_VERSION
                                                               << ", Found: " << version);
                throw std::runtime_error("Index version mismatch");
            }

            // Read flags
            readBinaryPOD(input, flags_);

            // Load index parameters
            readBinaryPOD(input, checksum_);
            readBinaryPOD(input, space_type_);
            readBinaryPOD(input, dimension_);
            readBinaryPOD(input, quant_level_);
            readBinaryPOD(input, maxElements_);
            LOG_DEBUG("Loading index with maxElements: " << maxElements_);
            readBinaryPOD(input, curElementsCount_);
            LOG_DEBUG("Current elements count: " << curElementsCount_);
            readBinaryPOD(input, deletedElementsCount_);
            readBinaryPOD(input, labelOffset_);
            readBinaryPOD(input, maxLevel_);
            readBinaryPOD(input, entryPoint_);
            readBinaryPOD(input, M_);
            readBinaryPOD(input, M0_);
            readBinaryPOD(input, mult_);
            readBinaryPOD(input, efConstruction_);

            if(maxElements_i > 0) {
                maxElements_ = maxElements_i;
            }
            maxM_ = M_ + settings::MAX_EXTRA_NEIGHBORS;
            maxM0_ = M0_ + 2 * settings::MAX_EXTRA_NEIGHBORS;
            // links will also store number of linked elements
            sizeLinksUpperLayers_ = sizeof(idInt) + maxM_ * sizeof(idInt);
            sizeLinksBaseLayer_ = sizeof(idInt) + maxM0_ * sizeof(idInt);
            // We are not storing the data in the level 0 memory, only the links and labels
            sizeDataAtBaseLayer_ = sizeLinksBaseLayer_ + sizeof(flagInt) + sizeof(idInt);
            labelOffset_ = sizeLinksBaseLayer_ + sizeof(flagInt);

            // Create appropriate space based on stored type
            space_ = std::unique_ptr<SpaceInterface<float>>(
                    createSpace<float>(space_type_, dimension_, quant_level_));
            data_size_ = space_->get_data_size();
            fstDistFunc_ = space_->get_dist_func();
            fstSimFunc_ = space_->get_sim_func();
            dist_func_param_ = space_->get_dist_func_param();

#ifndef DISABLE_HYBRID_QUANTIZATION
            // Create INT8 space for upper layers
            space_int8_ = std::unique_ptr<SpaceInterface<float>>(createSpace<float>(
                    space_type_, dimension_, ndd::quant::QuantizationLevel::INT8));
            data_size_int8_ = space_int8_->get_data_size();
            fstSimFuncInt8_ = space_int8_->get_sim_func();
            dist_func_param_int8_ = space_int8_->get_dist_func_param();
#endif  //DISABLE_HYBRID_QUANTIZATION

            // Allocate memory and load level 0 data
            dataBaseLayer_ = (char*)malloc(maxElements_ * sizeDataAtBaseLayer_);
            if(dataBaseLayer_ == nullptr) {
                throw std::runtime_error("Not enough memory");
            }
            input.read(dataBaseLayer_, maxElements_ * sizeDataAtBaseLayer_);

            uint64_t upper_marker_check;
            readBinaryPOD(input, upper_marker_check);
            if(upper_marker_check != 0xDEADBEEFDEADBEEF) {
                LOG_DEBUG("Corrupt index file: dataUpperLayer_ marker missing or mismatched");
                throw std::runtime_error(
                        "Corrupt index file: dataUpperLayer_ marker missing or mismatched");
            }
            labelLookup_.resize(maxElements_, INVALID_ID);
            for(size_t i = 0; i < curElementsCount_; i++) {
                idInt label = getExternalLabel(i);
                if(label >= maxElements_) {
                    // Oops.. The index is corrupted
                    LOG_DEBUG("Corrupt index: label "
                              << label << " at i=" << i
                              << " exceeds maxElements_ = " << maxElements_);
                    throw std::runtime_error("Corrupt index: label " + std::to_string(label)
                                             + " at i=" + std::to_string(i)
                                             + " exceeds maxElements_ = "
                                             + std::to_string(maxElements_));
                }
                labelLookup_[label] = i;
                //labelLookup_[getExternalLabel(i)] = i;
            }

            dataUpperLayer_.resize(maxElements_);
            while(true) {
                idhInt id;
                readBinaryPOD(input, id);
                if(id == INVALID_ID) {
                    break;
                }

                size_t header_size;
#ifdef DISABLE_HYBRID_QUANTIZATION
                // Step 1: Read vector + level header (data_size_ + sizeof(levelInt))
                header_size = data_size_ + sizeof(levelInt);
#else
                // Step 1: Read vector + level header (data_size_int8_ + sizeof(levelInt))
                header_size = data_size_int8_ + sizeof(levelInt);
#endif  //DISABLE_HYBRID_QUANTIZATION

                std::vector<uint8_t> header_buf(header_size);
                input.read(reinterpret_cast<char*>(header_buf.data()), header_size);
                if(!input) {
                    throw std::runtime_error("Failed to read upper layer header");
                }

                levelInt level;
#ifdef DISABLE_HYBRID_QUANTIZATION
                level = *reinterpret_cast<levelInt*>(header_buf.data() + data_size_);
#else
                level = *reinterpret_cast<levelInt*>(header_buf.data() + data_size_int8_);
#endif  //DISABLE_HYBRID_QUANTIZATION

                size_t total_size = header_size + level * sizeLinksUpperLayers_;

                // Step 2: Allocate and copy header
                auto mem = std::make_unique<uint8_t[]>(total_size);
                memcpy(mem.get(), header_buf.data(), header_size);

                // Step 3: Read linklists
                input.read(reinterpret_cast<char*>(mem.get() + header_size),
                           level * sizeLinksUpperLayers_);
                if(!input) {
                    throw std::runtime_error("Failed to read upper layer linklists");
                }

                dataUpperLayer_[id] = std::move(mem);
            }

            input.close();

            visited_list_pool_ =
                    std::unique_ptr<VisitedListPool>(new VisitedListPool(1, maxElements_));
            if(visited_list_pool_ == nullptr) {
                throw std::runtime_error("Not enough memory");
            }

            // Adjust cache based on element count and cache percentage threshold (default
            // VECTOR_CACHE_PERCENTAGE) adjustCacheForElementCount(curElementsCount_);
        }

        // Adjust cache bits based on element count and percentage threshold
        // cache_percent: percentage of element count the cache should cover (e.g., 5 for 5%)
        // void adjustCacheForElementCount(size_t element_count) {
        // Cache adjustment is now handled externally or disabled
        // }

        template <bool is_new> void addPoint(const void* datapoint, idInt label) {
            LOG_TIME("addPoint");

#ifndef DISABLE_HYBRID_QUANTIZATION
            // Generate INT8 representation for upper layers
            std::vector<uint8_t> datapoint_int8 = quantizeToInt8(datapoint);
#endif  //DISABLE_HYBRID_QUANTIZATION

            //std::shared_lock<std::shared_mutex> lock(index_lock_);
            idhInt cur_c = 0;
            levelInt curLevel = 0;
            if(is_new) {
                // Adding a new point
                // Using fetch_add (or post-increment) ensures unique IDs even under contention.
                cur_c = curElementsCount_.fetch_add(1);

                if(cur_c >= maxElements_) {
                    // Restore count if we exceeded limit (optional, but good for correctness)
                    curElementsCount_--;
                    throw std::runtime_error("The number of elements exceeds the specified limit");
                }

                labelLookup_[label] = cur_c;
                setExternalLabel(cur_c, label);
                curLevel = getRandomLevel(mult_);
            } else {
                idhInt searchId = labelLookup_[label];
                if(searchId != INVALID_ID) {
                    // If the element is deleted, mark is undeleted first before calling update
                    // point
                    if(isMarkedDeleted(searchId)) {
                        unmarkDeletedInternal(searchId);
                    }
                    curLevel = getElementLevel(searchId);
                    removeAllConnections(searchId, curLevel);
                    cur_c = searchId;
                } else {
                    LOG_DEBUG("Label not found, can't update the point" << label);
                    return;
                }
            }
            // TODO - Check this ..is it thread safe to comment this
            // std::unique_lock <std::shared_mutex> lock_el(getLinkListMutex(cur_c));

            // Put the data in level 0 memory.
            // if (curLevel == 0) {
            //      memcpy(dataBaseLayer_ + cur_c * sizeDataAtBaseLayer_ + sizeDataAtBaseLayer_ -
            //      data_size_, datapoint, data_size_);
            // }
            // Initialize level 0 links
            // TODO - check if it is required
            {
                char* linklist = get_linklist0(cur_c);
                memset(linklist, 0, sizeLinksBaseLayer_);
            }

            // Create data in upper levels
            size_t total_size;
            if(curLevel > 0) {

#ifdef DISABLE_HYBRID_QUANTIZATION
                total_size = data_size_ + sizeof(levelInt) + curLevel * sizeLinksUpperLayers_;
#else
                total_size = data_size_int8_ + sizeof(levelInt) + curLevel * sizeLinksUpperLayers_;
#endif  //DISABLE_HYBRID_QUANTIZATION

                auto mem = std::make_unique<uint8_t[]>(total_size);

                // copy vector
#ifdef DISABLE_HYBRID_QUANTIZATION
                memcpy(mem.get(), datapoint, data_size_);
                memcpy(mem.get() + data_size_, &curLevel, sizeof(levelInt));
                // zero initialize linklists
                memset(mem.get() + data_size_ + sizeof(levelInt),
                       0,
                       curLevel * sizeLinksUpperLayers_);
#else
                memcpy(mem.get(), datapoint_int8.data(), data_size_int8_);
                memcpy(mem.get() + data_size_int8_, &curLevel, sizeof(levelInt));
                // zero initialize linklists
                memset(mem.get() + data_size_int8_ + sizeof(levelInt),
                       0,
                       curLevel * sizeLinksUpperLayers_);
#endif  //DISABLE_HYBRID_QUANTIZATION

                dataUpperLayer_[cur_c] = std::move(mem);
            }

            if(cur_c != 0) {

                levelInt maxlevelcopy = maxLevel_;

                // IMPORTANT: Check if this element has a higher level than the current max
                bool has_higher_level = (curLevel > maxlevelcopy);
                idhInt currObj = entryPoint_;

                // Traverse to find closest neighbors at each level
                // Greedy search till the current level
                for(int level = maxlevelcopy; level > curLevel; level--) {
                    bool changed = true;
                    while(changed) {
                        changed = false;
                        int* ll_cur;
                        ll_cur = (int*)get_linklist(currObj, level);
                        if(!ll_cur) {
                            continue;
                        }
                        int size = getListCount((idhInt*)ll_cur);
                        idhInt* datal = (idhInt*)(ll_cur + 1);

#ifdef DISABLE_HYBRID_QUANTIZATION
                        std::vector<uint8_t> curr_vec(data_size_);
#else
                        std::vector<uint8_t> curr_vec(data_size_int8_);
#endif  //DISABLE_HYBRID_QUANTIZATION

                        if(!getDataByInternalId(currObj, level, curr_vec.data())) {
                            continue;
                        }

                        dist_t curr_sim;
#ifdef DISABLE_HYBRID_QUANTIZATION
                        curr_sim = fstSimFunc_(datapoint, curr_vec.data(), dist_func_param_);
#else
                        curr_sim = fstSimFuncInt8_(
                                datapoint_int8.data(), curr_vec.data(), dist_func_param_int8_);
#endif  //DISABLE_HYBRID_QUANTIZATION

                        for(int i = 0; i < size; i++) {
                            idhInt candidate_id = datal[i];
                            if(candidate_id >= curElementsCount_) {
                                continue;
                            }

                            dist_t s;
#ifdef DISABLE_HYBRID_QUANTIZATION
                            std::vector<uint8_t> candidate_vec(data_size_);
                            if(!getDataByInternalId(candidate_id, level, candidate_vec.data())) {
                                continue;
                            }
                            s = fstSimFunc_(datapoint, candidate_vec.data(), dist_func_param_);
#else
                            std::vector<uint8_t> candidate_vec(data_size_int8_);
                            if(!getDataByInternalId(candidate_id, level, candidate_vec.data())) {
                                continue;
                            }
                            s = fstSimFuncInt8_(datapoint_int8.data(),
                                                candidate_vec.data(),
                                                dist_func_param_int8_);
#endif  //DISABLE_HYBRID_QUANTIZATION

                            if(s > curr_sim) {
                                curr_sim = s;
                                currObj = candidate_id;
                                changed = true;
                            }
                        }
                    }
                }
                // Connect at all levels from curLevel down to 0
                for(int level = std::min(curLevel, maxlevelcopy); level >= 0; level--) {
                    std::vector<std::pair<dist_t, idhInt>> sorted_candidates;

#ifdef DISABLE_HYBRID_QUANTIZATION
                    if(deletedElementsCount_) {
                        sorted_candidates = searchBaseLayer<true, true>(
                                currObj, datapoint, level, efConstruction_);
                    } else {  // No deleted elements
                        sorted_candidates = searchBaseLayer<true, false>(
                                currObj, datapoint, level, efConstruction_);
                    }
                    currObj = mutuallyConnectNewElement(datapoint, cur_c, sorted_candidates, level);
#else
                    const void* level_datapoint = (level == 0) ? datapoint : datapoint_int8.data();
                    if(deletedElementsCount_) {
                        sorted_candidates = searchBaseLayer<true, true>(
                                currObj, level_datapoint, level, efConstruction_);
                    } else {  // No deleted elements
                        sorted_candidates = searchBaseLayer<true, false>(
                                currObj, level_datapoint, level, efConstruction_);
                    }
                    currObj = mutuallyConnectNewElement(
                            level_datapoint, cur_c, sorted_candidates, level);
#endif  //DISABLE_HYBRID_QUANTIZATION
                }

                // If this element has a higher level than the current max,
                // update the max level and entry point
                if(has_higher_level) {
                    std::lock_guard<std::mutex> glock(global);
                    maxLevel_ = curLevel;
                    entryPoint_ = cur_c;
                }
            } else {
                // First element becomes enterpoint
                entryPoint_ = 0;
                maxLevel_ = curLevel;
                // preload element in vector cache at its proper cache location
                // if (curLevel == 0) {
                // memcpy(dataBaseLayer_ + cur_c * sizeDataAtBaseLayer_ + sizeDataAtBaseLayer_ -
                // data_size_, datapoint, data_size_);
                // }
            }
        }

        void markDelete(idInt label) {
            std::shared_lock<std::shared_mutex> lock(index_lock_);
            auto searchId = labelLookup_[label];
            if(searchId == INVALID_ID) {
                throw std::runtime_error("Label not found");
            }
            markDeletedInternal(searchId);
        }

        inline bool isMarkedDeleted(idhInt internal_id) const {
            const flagInt* flags = reinterpret_cast<const flagInt*>(get_linklist0(internal_id)
                                                                    + sizeLinksBaseLayer_);
            return (*flags & DELETE_MARK) != 0;
        }

        void resizeIndex(size_t new_max_elements) {
            std::unique_lock<std::shared_mutex> lock(index_lock_);
            if(new_max_elements < curElementsCount_) {
                throw std::runtime_error(
                        "Cannot resize, max element is less than the current number of elements");
            }

            // Reset and reallocate visited list pool with new size
            visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));

            // Reallocate base layer (dataBaseLayer_)
            char* dataBaseLayer_new =
                    (char*)realloc(dataBaseLayer_, new_max_elements * sizeDataAtBaseLayer_);
            if(dataBaseLayer_new == nullptr) {
                throw std::runtime_error(
                        "Not enough memory: resizeIndex failed to allocate base layer");
            }
            dataBaseLayer_ = dataBaseLayer_new;

            // Reallocate upper layer (dataUpperLayer_)
            dataUpperLayer_.resize(new_max_elements);

            // Resize label lookup vector and fill it with INVALID_ID
            labelLookup_.resize(new_max_elements, INVALID_ID);

            // Update maxElements_ count
            maxElements_ = new_max_elements;
        }

    private:
        // Invalid id for the label
        static constexpr idhInt INVALID_ID = static_cast<idhInt>(-1);
        static const unsigned char DELETE_MARK = 0x01;
        // TODO - We need to pass indexId in the constructor.
        // This may be helpful for logs
        std::string indexId_;
        size_t M_{0};
        size_t M0_{0};
        size_t maxM_{0};
        size_t maxM0_{0};
        size_t efConstruction_{0};
        size_t ef_{0};
        SpaceType space_type_;  // Now using SpaceType
        ndd::quant::QuantizationLevel quant_level_;
        int32_t checksum_;
        uint64_t flags_{0};  //Not using flags now. We can use it in future for various options
        std::unique_ptr<SpaceInterface<dist_t>> space_;

#ifndef DISABLE_HYBRID_QUANTIZATION
        std::unique_ptr<SpaceInterface<dist_t>> space_int8_;
#endif  //DISABLE_HYBRID_QUANTIZATION

        size_t dimension_;

        VectorFetcher vector_fetcher_;
        mutable std::shared_mutex index_lock_;

        size_t maxElements_{0};
        mutable std::atomic<size_t> curElementsCount_{0};
        mutable std::atomic<size_t> deletedElementsCount_{0};
        size_t sizeDataAtBaseLayer_{0};
        // Link list size for upper levels for each level (a node may exit at multiple upper levels)
        size_t sizeLinksUpperLayers_{0};
        // Link list size for level 0
        size_t sizeLinksBaseLayer_{0};

        double mult_{0.0};
        levelInt maxLevel_{0};

        std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};
        mutable std::mutex global;
        mutable std::vector<std::shared_mutex> linkListLocks_;

        idhInt entryPoint_{0};

        size_t labelOffset_{0};

        // Stores link lists and labels
        // Structure: idInt + linklist + label
        char* dataBaseLayer_{nullptr};
        // Since upper layer can have variable layers, the size of the link list
        // is not fixed. So we use a vector of unique_ptrs to store the list
        // Structure: vector_data + level (unint32_t) + [idInt + linklist]
        std::vector<std::unique_ptr<uint8_t[]>> dataUpperLayer_;

        // This will vary based on fp16 or fp32
        size_t data_size_{0};
        DISTFUNC<dist_t> fstDistFunc_;
        SIMFUNC<dist_t> fstSimFunc_;
        void* dist_func_param_{nullptr};

#ifndef DISABLE_HYBRID_QUANTIZATION
        size_t data_size_int8_{0};
        SIMFUNC<dist_t> fstSimFuncInt8_;
        void* dist_func_param_int8_{nullptr};
#endif  //DISABLE_HYBRID_QUANTIZATION

        // Maps external label to internal id
        std::vector<idhInt> labelLookup_;

        std::default_random_engine level_generator_;
        std::default_random_engine update_probability_generator_;

    public:
        std::shared_mutex& getLinkListMutex(idhInt id) const {
            size_t lock_id = id & (settings::MAX_LINK_LIST_LOCKS - 1);
            return linkListLocks_[lock_id];
        }

        // Internal function to mark an element as deleted
        void markDeletedInternal(idhInt internal_id) {
            flagInt* flags =
                    reinterpret_cast<flagInt*>(get_linklist0(internal_id) + sizeLinksBaseLayer_);
            *flags |= DELETE_MARK;
            deletedElementsCount_++;
        }

        void unmarkDeletedInternal(idhInt internal_id) {
            flagInt* flags =
                    reinterpret_cast<flagInt*>(get_linklist0(internal_id) + sizeLinksBaseLayer_);
            *flags &= ~DELETE_MARK;
            deletedElementsCount_--;
        }

        // Generate level for a new point
        levelInt getRandomLevel(double mult) {
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            double r = -log(distribution(level_generator_)) * mult;
            return (levelInt)r;
        }

        const uint8_t* getUpperLayerDataPtr(idhInt internal_id) const {
            if(dataUpperLayer_[internal_id] == nullptr) {
                return nullptr;
            }
            return dataUpperLayer_[internal_id].get();
        }

        // Modified function returning bool and filling buffer
        bool getDataByInternalId(idhInt internal_id, levelInt layer, uint8_t* buffer) const {
            if(layer == 0) {
                idInt external_label = getExternalLabel(internal_id);
                if(vector_fetcher_) {
                    // Directly fetch to buffer
                    return vector_fetcher_(external_label, buffer);
                }
                return false;
            } else {
                // FALLBACK: ideally callers should use getUpperLayerDataPtr
                if(dataUpperLayer_[internal_id] == nullptr) {
                    return false;
                }
#ifdef DISABLE_HYBRID_QUANTIZATION
                memcpy(buffer, dataUpperLayer_[internal_id].get(), data_size_);
#else
                memcpy(buffer, dataUpperLayer_[internal_id].get(), data_size_int8_);
#endif  //DISABLE_HYBRID_QUANTIZATION
                return true;
            }
            return false;
        }

        char* get_linklist0(idhInt internal_id) const {
            return dataBaseLayer_ + internal_id * sizeDataAtBaseLayer_;
        }

        inline char* get_linklist(idhInt id, levelInt level) const {
            if(level == 0) {
                return get_linklist0(id);
            }
            // int levels = getElementLevel(id);
            // if (level > levels) return nullptr;
            return reinterpret_cast<char*>(
#ifdef DISABLE_HYBRID_QUANTIZATION
                    dataUpperLayer_[id].get() + data_size_ + sizeof(levelInt)
                    + (level - 1) * sizeLinksUpperLayers_
#else
                    dataUpperLayer_[id].get() + data_size_int8_ + sizeof(levelInt)
                    + (level - 1) * sizeLinksUpperLayers_
#endif  //DISABLE_HYBRID_QUANTIZATION

            );
        }
        inline idhInt getListCount(idhInt* ptr) const { return *ptr; }

        inline void setListCount(idhInt* ptr, idhInt size) const { *ptr = size; }

        idInt getExternalLabel(idhInt internal_id) const {
            idInt return_label;
            memcpy(&return_label,
                   dataBaseLayer_ + internal_id * sizeDataAtBaseLayer_ + labelOffset_,
                   sizeof(idInt));
            return return_label;
        }

        void setExternalLabel(idhInt internal_id, idInt label) const {
            memcpy(dataBaseLayer_ + internal_id * sizeDataAtBaseLayer_ + labelOffset_,
                   &label,
                   sizeof(idInt));
        }

        inline levelInt getElementLevel(idhInt id) const {
            if(!dataUpperLayer_[id]) {
                return 0;
            }
#ifdef DISABLE_HYBRID_QUANTIZATION
            return *reinterpret_cast<const levelInt*>(dataUpperLayer_[id].get() + data_size_);
#else
            return *reinterpret_cast<const levelInt*>(dataUpperLayer_[id].get() + data_size_int8_);
#endif  //DISABLE_HYBRID_QUANTIZATION
        }

        // This function is used to get the neighbors based on heuristic
        // We let the neighbors grow beyond M and then prune them based on heuristic
        // The input is a sorted list (reverse order) by similarity
        std::vector<std::pair<dist_t, idhInt>>
        getNeighborsByHeuristic2(const std::vector<std::pair<dist_t, idhInt>>& candidates_sorted,
                                 size_t M,
                                 levelInt level) {
            if(candidates_sorted.size() <= M) {
                return candidates_sorted;
            }

            std::vector<std::pair<dist_t, idhInt>> result;
            result.reserve(M);

#ifdef DISABLE_HYBRID_QUANTIZATION
            std::vector<uint8_t> cand_vec(data_size_);
            std::vector<uint8_t> selected_vec(data_size_);
#else
            // INT8 awareness
            auto curSimFunc = (level == 0) ? fstSimFunc_ : fstSimFuncInt8_;
            auto curDistParam = (level == 0) ? dist_func_param_ : dist_func_param_int8_;
            size_t curDataSize = (level == 0) ? data_size_ : data_size_int8_;

            std::vector<uint8_t> cand_buf(curDataSize);      // Only used for level 0
            std::vector<uint8_t> selected_buf(curDataSize);  // Only used for level 0
#endif  //DISABLE_HYBRID_QUANTIZATION

            for(const auto& candidate : candidates_sorted) {
                if(result.size() == M) {
                    break;
                }

#ifdef DISABLE_HYBRID_QUANTIZATION
                if(!getDataByInternalId(candidate.second, level, cand_vec.data())) {
                    continue;
                }
#else
                const void* cand_vec = nullptr;
                if(level == 0) {
                    if(getDataByInternalId(candidate.second, level, cand_buf.data())) {
                        cand_vec = cand_buf.data();
                    }
                } else {
                    cand_vec = getUpperLayerDataPtr(candidate.second);
                }

                if(!cand_vec) {
                    continue;
                }
#endif  //DISABLE_HYBRID_QUANTIZATION

                bool good = true;
                dist_t sim;
                for(const auto& selected : result) {

#ifdef DISABLE_HYBRID_QUANTIZATION
                    if(!getDataByInternalId(selected.second, level, selected_vec.data())) {
                        continue;
                    }
                    sim = fstSimFunc_(selected_vec.data(), cand_vec.data(), dist_func_param_);
#else
                    const void* selected_vec_ptr = nullptr;
                    if(level == 0) {
                        if(getDataByInternalId(selected.second, level, selected_buf.data())) {
                            selected_vec_ptr = selected_buf.data();
                        }
                    } else {
                        selected_vec_ptr = getUpperLayerDataPtr(selected.second);
                    }

                    if(!selected_vec_ptr) {
                        continue;
                    }

                    sim = curSimFunc(selected_vec_ptr, cand_vec, curDistParam);
#endif  //DISABLE_HYBRID_QUANTIZATION

                    if(sim > candidate.first) {
                        good = false;
                        break;
                    }
                }

                if(good) {
                    result.push_back(candidate);
                }
            }

            return result;
        }

        // This function is used to connect the new element to its neighbors
        // It takes the data point, current element id, sorted candidates and level
        idhInt
        mutuallyConnectNewElement(const void* data_point,
                                  idhInt cur_c,
                                  const std::vector<std::pair<dist_t, idhInt>>& sorted_candidates,
                                  levelInt level) {
            LOG_TIME("mutuallyConnectNewElement");

            size_t curMaxM = level ? maxM_ : maxM0_;
            size_t curM = level ? M_ : M0_;

#ifndef DISABLE_HYBRID_QUANTIZATION
            // INT8 awareness
            auto curSimFunc = (level == 0) ? fstSimFunc_ : fstSimFuncInt8_;
            auto curDistParam = (level == 0) ? dist_func_param_ : dist_func_param_int8_;
            size_t curDataSize = (level == 0) ? data_size_ : data_size_int8_;
#endif  //DISABLE_HYBRID_QUANTIZATION

            // Step 1: Heuristic pruning
            const auto& selected = getNeighborsByHeuristic2(sorted_candidates, curM, level);

            if(selected.empty()) {
                // If no candidates selected, just return current entry point or invalid
                // This can happen if the graph is empty or disconnected
                return 0;  // Or better handling
            }

            idhInt next_closest_entry_point = selected[0].second;

            // Step 2: Set connections for cur_c
            {
                idhInt* ll_cur = reinterpret_cast<idhInt*>(level == 0 ? get_linklist0(cur_c)
                                                                      : get_linklist(cur_c, level));
                if(ll_cur) {
                    setListCount(ll_cur, selected.size());
                    idhInt* data = (ll_cur + 1);
                    for(size_t idx = 0; idx < selected.size(); idx++) {
                        data[idx] = selected[idx].second;
                    }
                }
            }

            // Step 3: Add cur_c to neighbors' lists
#ifdef DISABLE_HYBRID_QUANTIZATION
            std::vector<uint8_t> neighbor_vec(data_size_);
            std::vector<uint8_t> data_vec(data_size_);
#else
            std::vector<uint8_t> neighbor_buf(curDataSize);  // Used for level 0
            std::vector<uint8_t> data_buf(curDataSize);      // Used for level 0
#endif  //DISABLE_HYBRID_QUANTIZATION

            for(const auto& p : selected) {
                idhInt neighbor = p.second;
                if(neighbor == cur_c) {
                    continue;
                }

                std::unique_lock<std::shared_mutex> lock(getLinkListMutex(neighbor));
                idhInt* ll_other = reinterpret_cast<idhInt*>(
                        level == 0 ? get_linklist0(neighbor) : get_linklist(neighbor, level));
                if(!ll_other) {
                    continue;
                }

                idhInt sz = getListCount(ll_other);
                idhInt* data = (ll_other + 1);

                if(sz < curMaxM) {
                    data[sz] = cur_c;
                    setListCount(ll_other, sz + 1);
                } else {
#ifdef DISABLE_HYBRID_QUANTIZATION
                    if(!getDataByInternalId(neighbor, level, neighbor_vec.data())) {
                        continue;
                    }
#else
                    const void* neighbor_data = nullptr;
                    if(level == 0) {
                        if(getDataByInternalId(neighbor, level, neighbor_buf.data())) {
                            neighbor_data = neighbor_buf.data();
                        }
                    } else {
                        neighbor_data = getUpperLayerDataPtr(neighbor);
                    }

                    if(!neighbor_data) {
                        continue;
                    }
#endif  //DISABLE_HYBRID_QUANTIZATION

                    std::vector<std::pair<dist_t, idhInt>> all_candidates;
                    all_candidates.reserve(sz + 1);

#ifdef DISABLE_HYBRID_QUANTIZATION
                    all_candidates.emplace_back(
                            fstSimFunc_(neighbor_vec.data(), data_point, dist_func_param_), cur_c);
#else
                    all_candidates.emplace_back(curSimFunc(neighbor_data, data_point, curDistParam),
                                                cur_c);
#endif  //DISABLE_HYBRID_QUANTIZATION

                    for(size_t j = 0; j < sz; j++) {
                        dist_t sim;
#ifdef DISABLE_HYBRID_QUANTIZATION
                        if(!getDataByInternalId(data[j], level, data_vec.data())) {
                            continue;
                        }
                        sim = fstSimFunc_(neighbor_vec.data(), data_vec.data(), dist_func_param_);
#else
                        const void* other_neighbor_data = nullptr;
                        if(level == 0) {
                            if(getDataByInternalId(data[j], level, data_buf.data())) {
                                other_neighbor_data = data_buf.data();
                            }
                        } else {
                            other_neighbor_data = getUpperLayerDataPtr(data[j]);
                        }
                        if(!other_neighbor_data) {
                            continue;
                        }

                        sim = curSimFunc(neighbor_data, other_neighbor_data, curDistParam);
#endif  //DISABLE_HYBRID_QUANTIZATION

                        all_candidates.emplace_back(sim, data[j]);
                    }
                    std::sort(all_candidates.begin(),
                              all_candidates.end(),
                              [](const auto& a, const auto& b) { return a.first > b.first; });

                    auto pruned = getNeighborsByHeuristic2(all_candidates, curM, level);
                    for(size_t j = 0; j < pruned.size(); j++) {
                        data[j] = pruned[j].second;
                    }
                    setListCount(ll_other, pruned.size());
                }
            }

            return next_closest_entry_point;
        }

        // Search function for the base layer
        // Returns a vector of top candidates sorted by similarity (1-distance) in reverse order
        template <bool is_insert, bool has_deletions>
        std::vector<std::pair<dist_t, idhInt>>
        searchBaseLayer(idhInt ep_id, const void* data_point, idhInt layer, size_t ef) const {
            LOG_TIME("searchBaseLayer");
            VisitedList* vl = visited_list_pool_->getFreeVisitedList();
            vl_type* visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            max_heap_pq candidate_set;
            min_heap_pq top_candidates;

#ifdef DISABLE_HYBRID_QUANTIZATION
            std::vector<uint8_t> buffer(data_size_);
#else
            // INT8 awareness
            auto curSimFunc = (layer == 0) ? fstSimFunc_ : fstSimFuncInt8_;
            auto curDistParam = (layer == 0) ? dist_func_param_ : dist_func_param_int8_;
            size_t curDataSize = (layer == 0) ? data_size_ : data_size_int8_;
            std::vector<uint8_t> buffer;
            if(layer == 0) {
                buffer.resize(curDataSize);
            }
#endif  //DISABLE_HYBRID_QUANTIZATION

            dist_t lowerBound;
            if(!has_deletions || !isMarkedDeleted(ep_id)) {

#ifdef DISABLE_HYBRID_QUANTIZATION
                if(getDataByInternalId(ep_id, layer, buffer.data())) {
                    dist_t sim = fstSimFunc_(data_point, buffer.data(), dist_func_param_);
#else
                const void* vec_data = nullptr;
                if(layer == 0) {
                    if(getDataByInternalId(ep_id, layer, buffer.data())) {
                        vec_data = buffer.data();
                    }
                } else {
                    vec_data = getUpperLayerDataPtr(ep_id);
                }

                if(vec_data) {
                    dist_t sim = curSimFunc(data_point, vec_data, curDistParam);
#endif  //DISABLE_HYBRID_QUANTIZATION

                    top_candidates.emplace(sim, ep_id);
                    candidate_set.emplace(sim, ep_id);
                    lowerBound = sim;
                } else {
                    lowerBound = std::numeric_limits<dist_t>::lowest();
                    candidate_set.emplace(lowerBound, ep_id);
                }
            } else {
                // If entry point is deleted, lower bound will be minimum
                lowerBound = std::numeric_limits<dist_t>::lowest();
                candidate_set.emplace(lowerBound, ep_id);
            }
            visited_array[ep_id] = visited_array_tag;
            int below_threshold_count = 0;
            int max_below_threshold = is_insert ? settings::EARLY_EXIT_BUFFER_INSERT
                                                : settings::EARLY_EXIT_BUFFER_QUERY;

            while(!candidate_set.empty()) {
                auto current_pair = candidate_set.top();
                idhInt current_id = current_pair.second;
                // Early exit if we have enough candidates
                if(current_pair.first < lowerBound && top_candidates.size() >= ef) {
                    below_threshold_count++;
                    if(below_threshold_count > max_below_threshold) {
                        break;
                    }
                } else {
                    below_threshold_count = 0;
                }

                candidate_set.pop();

                // Get neighbors
                idhInt* data = (layer == 0) ? (idhInt*)get_linklist0(current_id)
                                            : (idhInt*)get_linklist(current_id, layer);
                if(!data) {
                    LOG_DEBUG("No linklist found for id: " << current_id);
                    continue;
                }
                idhInt size = getListCount((idhInt*)data);
                idhInt* datal = (idhInt*)(data + 1);

                for(idhInt j = 0; j < size; j++) {
                    idhInt candidate_id = *(datal + j);
                    if(visited_array[candidate_id] == visited_array_tag) {
                        continue;
                    }
                    visited_array[candidate_id] = visited_array_tag;
                    if(has_deletions && isMarkedDeleted(candidate_id)) {
                        continue;
                    }

                    dist_t sim;
#ifdef DISABLE_HYBRID_QUANTIZATION
                    if(!getDataByInternalId(candidate_id, layer, buffer.data())) {
                        continue;
                    }
                    sim = fstSimFunc_(data_point, buffer.data(), dist_func_param_);
#else
                    const void* neighbor_data = nullptr;
                    if(layer == 0) {
                        if(getDataByInternalId(candidate_id, layer, buffer.data())) {
                            neighbor_data = buffer.data();
                        }
                    } else {
                        neighbor_data = getUpperLayerDataPtr(candidate_id);
                    }

                    if(!neighbor_data) {
                        continue;
                    }

                    sim = curSimFunc(data_point, neighbor_data, curDistParam);
#endif  //DISABLE_HYBRID_QUANTIZATION

                    if(top_candidates.size() < ef || sim > lowerBound) {
                        candidate_set.emplace(sim, candidate_id);

                        if(!has_deletions || !isMarkedDeleted(candidate_id)) {
                            top_candidates.emplace(sim, candidate_id);
                            if(top_candidates.size() > ef) {
                                top_candidates.pop();
                            }
                            if(!top_candidates.empty()) {
                                lowerBound = top_candidates.top().first;
                            }
                        }
                    }
                }
            }

            visited_list_pool_->releaseVisitedList(vl);
            // return a vector of top candidates sorted by similarity (1-distance) in reverse order
            std::vector<std::pair<dist_t, idhInt>> sorted_candidates;
            sorted_candidates.reserve(top_candidates.size());
            while(!top_candidates.empty()) {
                sorted_candidates.push_back(top_candidates.top());
                top_candidates.pop();
            }
            std::reverse(sorted_candidates.begin(), sorted_candidates.end());
            // Return the top candidates
            return sorted_candidates;
        }

        // For a given internal id, go to all neighbors and remove the connection. Then set the
        // linklist count to 0
        //  Repeat this for all levels
        void removeAllConnections(idhInt internal_id, levelInt elem_level) {

            for(int level = 0; level <= elem_level; ++level) {
                idhInt* ll_self = (idhInt*)get_linklist(internal_id, level);
                idhInt size = getListCount(ll_self);
                idhInt* neighbors = (idhInt*)(ll_self + 1);

                for(idhInt i = 0; i < size; ++i) {
                    idhInt neighbor_id = neighbors[i];
                    if(neighbor_id >= curElementsCount_) {
                        continue;
                    }

                    // Fix neighbor's link list
                    {
                        std::unique_lock<std::shared_mutex> lock_neighbor(
                                getLinkListMutex(neighbor_id));
                        idhInt* ll_other = (idhInt*)get_linklist(neighbor_id, level);
                        idhInt sz = getListCount(ll_other);
                        idhInt* data = (idhInt*)(ll_other + 1);

                        idhInt new_size = 0;
                        for(idhInt j = 0; j < sz; ++j) {
                            if(data[j] != internal_id) {
                                data[new_size++] = data[j];
                            }
                        }
                        setListCount(ll_other, new_size);
                    }
                }
                // Now clear own links (make neighbor count 0)
                std::unique_lock<std::shared_mutex> lock_self(getLinkListMutex(internal_id));
                setListCount(ll_self, 0);
            }
        }
    };
}  // namespace hnswlib
