#pragma once
#include <cstdint>

// Compile-time configuration for ID width.
// Define NDD_USE_64BIT_IDS in your build system (e.g., CMake -DNDD_USE_64BIT_IDS=ON)
// to enable 64-bit IDs. Default is 32-bit for performance/memory efficiency.

#include "../../third_party/roaring_bitmap/roaring.hh"

namespace ndd {

#ifdef NDD_USE_64BIT_IDS
    // --- 64-bit Configuration ---
    using idInt = uint64_t;   // External ID (stored in DB, exposed to user)
    using idhInt = uint64_t;  // Internal HNSW ID (used inside HNSW structures)
    using RoaringBitmap = roaring::Roaring64Map;
#else
    // --- 32-bit Configuration (Default) ---
    using idInt = uint32_t;   // External ID (stored in DB, exposed to user)
    using idhInt = uint32_t;  // Internal HNSW ID (used inside HNSW structures)
    using RoaringBitmap = roaring::Roaring;
#endif

}  //namespace ndd
