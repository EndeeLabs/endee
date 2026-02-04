#pragma once
#include "../hnsw/hnswlib.h"
#include "../quant/dispatch.hpp"
#include <stdexcept>

namespace hnswlib {

    class UnifiedSpace final : public SpaceInterface<float> {
    private:
        ndd::quant::QuantizerDispatch dispatch_;
        DISTFUNC<float> selected_dist_func_{nullptr};
        SIMFUNC<float> selected_sim_func_{nullptr};
        size_t dim_;
        size_t data_size_;
        DistParams dist_params_;
        SpaceType space_type_;

    public:
        UnifiedSpace(SpaceType space_type, size_t dim, ndd::quant::QuantizationLevel quant_level) :
            dim_(dim),
            space_type_(space_type) {

            // 1. Get the capabilities for this quantization level
            dispatch_ = ndd::quant::get_quantizer_dispatch(quant_level);

            // 2. Set up parameters
            data_size_ = dispatch_.get_storage_size(dim);
            dist_params_.dim = dim;
            dist_params_.quant_level = static_cast<uint8_t>(quant_level);

            // 3. Pick the right distance and similarity functions based on the metric
            switch(space_type_) {
                case L2_SPACE:
                    selected_dist_func_ = dispatch_.dist_l2;
                    selected_sim_func_ = dispatch_.sim_l2;
                    break;
                case IP_SPACE:
                    selected_dist_func_ = dispatch_.dist_ip;
                    selected_sim_func_ = dispatch_.sim_ip;
                    break;
                case COSINE_SPACE:
                    selected_dist_func_ = dispatch_.dist_cosine;
                    selected_sim_func_ = dispatch_.sim_cosine;
                    break;
                default:
                    throw std::runtime_error("Unknown space type");
            }
        }

        size_t get_data_size() override { return data_size_; }

        DISTFUNC<float> get_dist_func() override { return selected_dist_func_; }

        SIMFUNC<float> get_sim_func() override { return selected_sim_func_; }

        void* get_dist_func_param() override { return &dist_params_; }
    };

}  // namespace hnswlib
