#pragma once
#include <vector>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include "common.hpp"

// Include all quantizer implementations to ensure they are registered
#include "float16.hpp"
#include "float32.hpp"
#include "int8d.hpp"
#include "int16d.hpp"
#include "binary.hpp"

namespace ndd {
    namespace quant {

        // The "One Function" to get the behavior
        inline QuantizerDispatch get_quantizer_dispatch(QuantizationLevel level) {
            auto quantizer = QuantizationRegistry::instance().getQuantizer(level);
            if(!quantizer) {
                throw std::runtime_error("Quantization level not registered: "
                                         + quantLevelToString(level));
            }
            return quantizer->getDispatch();
        }

    }  // namespace quant
}  // namespace ndd
