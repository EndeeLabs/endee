#pragma once

#define MSGPACK_NO_BOOST
#define MSGPACK_USE_X3_PARSE 0
#include <string>
#include <vector>
#include <sstream>

#include "msgpack/object.hpp"
#include "msgpack/iterator.hpp"
#include "msgpack/zone.hpp"
#include "msgpack/pack.hpp"
#include "msgpack/null_visitor.hpp"
#include "msgpack/parse.hpp"
#include "msgpack/unpack.hpp"
#include "msgpack/sbuffer.hpp"
#include "msgpack/vrefbuffer.hpp"
#include "msgpack/version.hpp"
#include "msgpack/type.hpp"

// Define msgpack-friendly structures and utility functions
namespace ndd {

    // Vector metadata structure
    struct VectorMeta {
        std::string id;             // String identifier
        std::vector<uint8_t> meta;  // Binary metadata (zipped)
        std::string filter;         // Filter as JSON string
        float norm;                 // Vector norm (only for cosine distance)

        MSGPACK_DEFINE(id, meta, filter, norm)
    };

    // Complete vector object structure
    struct VectorObject {
        std::string id;             // String identifier
        std::vector<uint8_t> meta;  // Binary metadata (zipped)
        std::string filter;         // Filter as JSON string
        float norm;                 // Vector norm (only for cosine distance)
        std::vector<float> vector;  // Vector data

        MSGPACK_DEFINE(id, meta, filter, norm, vector)
    };

    struct HybridVectorObject {
        std::string id;                    // String identifier
        std::vector<uint8_t> meta;         // Binary metadata (zipped)
        std::string filter;                // Filter as JSON string
        float norm;                        // Vector norm (only for cosine distance)
        std::vector<float> vector;         // Vector data
        std::vector<uint32_t> sparse_ids;  // Sparse vector indices
        std::vector<float> sparse_values;  // Sparse vector values

        MSGPACK_DEFINE(id, meta, filter, norm, vector, sparse_ids, sparse_values)
    };

    // Search result structure
    struct VectorResult {
        float similarity;           // Similarity from query (1-distance)
        std::string id;             // String identifier
        std::vector<uint8_t> meta;  // Binary metadata (zipped)
        std::string filter;         // Filter as JSON string
        float norm;                 // Vector norm (only for cosine distance)
        std::vector<float> vector;  // Vector data (optional)

        MSGPACK_DEFINE(similarity, id, meta, filter, norm, vector)
    };

    struct SparseVectorResult {
        float similarity;                  // Similarity from query (dot product)
        std::string id;                    // String identifier
        std::vector<uint8_t> meta;         // Binary metadata (zipped)
        std::string filter;                // Filter as JSON string
        std::vector<uint32_t> sparse_ids;  // Sparse vector indices
        std::vector<float> sparse_values;  // Sparse vector values

        MSGPACK_DEFINE(similarity, id, meta, filter, sparse_ids, sparse_values)
    };

    // Batch of vectors for bulk operations
    struct VectorBatch {
        std::vector<VectorObject> vectors;

        MSGPACK_DEFINE(vectors)
    };
    struct HybridVectorBatch {
        std::vector<HybridVectorObject> vectors;

        MSGPACK_DEFINE(vectors)
    };

    // Collection of search results
    struct ResultSet {
        std::vector<VectorResult> results;

        MSGPACK_DEFINE(results)
    };
    struct HybridResultSet {
        std::vector<VectorResult> dense;
        std::vector<SparseVectorResult> sparse;

        MSGPACK_DEFINE(dense, sparse)
    };

}  // namespace ndd