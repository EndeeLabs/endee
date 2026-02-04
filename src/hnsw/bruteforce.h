#pragma once
#include <unordered_map>
#include <fstream>
#include <mutex>
#include <algorithm>
#include <assert.h>

namespace hnswlib {
    template <typename dist_t> class BruteforceSearch : public AlgorithmInterface<dist_t> {
    public:
        char* data_;
        size_t maxelements_;
        size_t cur_element_count;
        size_t size_per_element_;

        size_t data_size_;
        DISTFUNC<dist_t> fstdistfunc_;
        void* dist_func_param_;
        std::mutex index_lock;

        std::unordered_map<ndd::idInt, size_t> dict_external_to_internal;

        BruteforceSearch(SpaceInterface<dist_t>* s) :
            data_(nullptr),
            maxelements_(0),
            cur_element_count(0),
            size_per_element_(0),
            data_size_(0),
            dist_func_param_(nullptr) {}

        BruteforceSearch(SpaceInterface<dist_t>* s, const std::string& location) :
            data_(nullptr),
            maxelements_(0),
            cur_element_count(0),
            size_per_element_(0),
            data_size_(0),
            dist_func_param_(nullptr) {
            loadIndex(location, s);
        }

        BruteforceSearch(SpaceInterface<dist_t>* s, size_t maxElements) {
            maxelements_ = maxElements;
            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            size_per_element_ = data_size_ + sizeof(ndd::idInt);
            data_ = (char*)malloc(maxElements * size_per_element_);
            if(data_ == nullptr) {
                throw std::runtime_error(
                        "Not enough memory: BruteforceSearch failed to allocate data");
            }
            cur_element_count = 0;
        }

        ~BruteforceSearch() { free(data_); }

        void addPoint(const void* datapoint, ndd::idInt label, bool replace_deleted = false) {
            int idx;
            {
                std::unique_lock<std::mutex> lock(index_lock);

                auto search = dict_external_to_internal.find(label);
                if(search != dict_external_to_internal.end()) {
                    idx = search->second;
                } else {
                    if(cur_element_count >= maxelements_) {
                        throw std::runtime_error(
                                "The number of elements exceeds the specified limit\n");
                    }
                    idx = cur_element_count;
                    dict_external_to_internal[label] = idx;
                    cur_element_count++;
                }
            }
            memcpy(data_ + size_per_element_ * idx + data_size_, &label, sizeof(ndd::idInt));
            memcpy(data_ + size_per_element_ * idx, datapoint, data_size_);
        }

        void removePoint(ndd::idInt cur_external) {
            std::unique_lock<std::mutex> lock(index_lock);

            auto found = dict_external_to_internal.find(cur_external);
            if(found == dict_external_to_internal.end()) {
                return;
            }

            dict_external_to_internal.erase(found);

            size_t cur_c = found->second;
            ndd::idInt label = *((ndd::idInt*)(data_ + size_per_element_ * (cur_element_count - 1)
                                               + data_size_));
            dict_external_to_internal[label] = cur_c;
            memcpy(data_ + size_per_element_ * cur_c,
                   data_ + size_per_element_ * (cur_element_count - 1),
                   data_size_ + sizeof(ndd::idInt));
            cur_element_count--;
        }

        std::priority_queue<std::pair<dist_t, ndd::idInt>> searchKnn(
                const void* query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
            assert(k <= cur_element_count);
            std::priority_queue<std::pair<dist_t, ndd::idInt>> topResults;
            if(cur_element_count == 0) {
                return topResults;
            }
            for(int i = 0; i < k; i++) {
                dist_t dist =
                        fstdistfunc_(query_data, data_ + size_per_element_ * i, dist_func_param_);
                ndd::idInt label = *((ndd::idInt*)(data_ + size_per_element_ * i + data_size_));
                if((!isIdAllowed) || (*isIdAllowed)(label)) {
                    topResults.emplace(dist, label);
                }
            }
            dist_t lastdist = topResults.empty() ? std::numeric_limits<dist_t>::max()
                                                 : topResults.top().first;
            for(int i = k; i < cur_element_count; i++) {
                dist_t dist =
                        fstdistfunc_(query_data, data_ + size_per_element_ * i, dist_func_param_);
                if(dist <= lastdist) {
                    ndd::idInt label = *((ndd::idInt*)(data_ + size_per_element_ * i + data_size_));
                    if((!isIdAllowed) || (*isIdAllowed)(label)) {
                        topResults.emplace(dist, label);
                    }
                    if(topResults.size() > k) {
                        topResults.pop();
                    }

                    if(!topResults.empty()) {
                        lastdist = topResults.top().first;
                    }
                }
            }
            return topResults;
        }

        void saveIndex(const std::string& location) {
            std::ofstream output(location, std::ios::binary);
            std::streampos position;

            writeBinaryPOD(output, maxelements_);
            writeBinaryPOD(output, size_per_element_);
            writeBinaryPOD(output, cur_element_count);

            output.write(data_, maxelements_ * size_per_element_);

            output.close();
        }

        void loadIndex(const std::string& location, SpaceInterface<dist_t>* s) {
            std::ifstream input(location, std::ios::binary);
            std::streampos position;

            readBinaryPOD(input, maxelements_);
            readBinaryPOD(input, size_per_element_);
            readBinaryPOD(input, cur_element_count);

            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            size_per_element_ = data_size_ + sizeof(ndd::idInt);
            data_ = (char*)malloc(maxelements_ * size_per_element_);
            if(data_ == nullptr) {
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate data");
            }

            input.read(data_, maxelements_ * size_per_element_);

            input.close();
        }
    };

    // Standalone function for bruteforce search on a subset of vectors
    // Uses the same SpaceInterface as the HNSW index for consistency
    template <typename dist_t>
    std::vector<std::pair<dist_t, idInt>>
    searchKnnSubset(const void* query_data,
                    const std::vector<std::pair<idInt, std::vector<uint8_t>>>& vector_subset,
                    size_t k,
                    hnswlib::SpaceInterface<dist_t>* space) {

        if(vector_subset.empty() || k == 0) {
            return {};
        }

        // Get distance function from space interface (same as HNSW uses)
        hnswlib::DISTFUNC<dist_t> distance_func = space->get_dist_func();
        void* dist_func_param = space->get_dist_func_param();

        // Use priority queue to maintain top k results (max heap for smallest distances)
        std::priority_queue<std::pair<dist_t, idInt>> top_results;

        // Compute distances for all vectors in subset
        for(const auto& [label, vec_bytes] : vector_subset) {
            dist_t distance = distance_func(query_data, vec_bytes.data(), dist_func_param);

            if(top_results.size() < k) {
                top_results.emplace(distance, label);
            } else if(distance < top_results.top().first) {
                top_results.pop();
                top_results.emplace(distance, label);
            }
        }

        // Convert to vector and reverse to get ascending order (best first)
        std::vector<std::pair<dist_t, idInt>> results;
        results.reserve(top_results.size());

        while(!top_results.empty()) {
            results.push_back(top_results.top());
            top_results.pop();
        }

        std::reverse(results.begin(), results.end());
        return results;
    }

}  // namespace hnswlib
