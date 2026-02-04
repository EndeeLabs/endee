#pragma once

#include <cstdlib>
#include <string>
#include <sstream>
#include <algorithm>
#include <cstdint>

constexpr uint64_t KB = (1024ULL);
constexpr uint64_t MB = (1024ULL * KB);
constexpr uint64_t GB = (1024ULL * MB);
constexpr uint64_t TB = (1024ULL * GB);

namespace settings {
    // === Compile-time constants ===
    // For strings we use inline const and not constexpr. Some compilers
    // do not support constexpr for std::string
    inline const std::string NAME = "Endee";
    inline const std::string VERSION = "1.0.0-beta";
    inline uint16_t INDEX_VERSION = 1;
    inline const std::string DEFAULT_SPACE_TYPE = "cosine";
    constexpr size_t DEFAULT_STORAGE_BITS =
            16;  // 16 bits = 2 bytes per element. Only for dense vectors
    constexpr size_t MIN_DIMENSION = 2;
    constexpr size_t MAX_DIMENSION = 16'384;
    constexpr size_t DEFAULT_M = 16;
    constexpr size_t MIN_M = 4;
    constexpr size_t MAX_M = 512;
    constexpr size_t DEFAULT_EF_CONSTRUCT = 128;
    constexpr size_t MIN_EF_CONSTRUCT = 8;
    constexpr size_t MAX_EF_CONSTRUCT = 4096;
    constexpr size_t DEFAULT_EF_SEARCH = 128;
    constexpr size_t MIN_K = 1;
    constexpr size_t MAX_K = 4096;
    constexpr size_t RANDOM_SEED = 100;
    constexpr size_t SAVE_EVERY_N_UPDATES = 10'000;
    constexpr size_t RECOVERY_BATCH_SIZE = 20'000;
    constexpr size_t SAVE_EVERY_N_MINUTES = 30;
    // Number of threads for http server - 0 means it will default to hardware concurrency
    constexpr size_t NUM_SERVER_THREADS = 0;
    // Number of save mutexes for parallel saves
    constexpr size_t NUM_INDEX_SAVE_MUTEXES = 16;

    // We allow some extra neighbors before pruning
    constexpr size_t MAX_EXTRA_NEIGHBORS = 3;

    // MDBX default map sizes. Growth step and initial size are the same for all databases.
    // System tables
    constexpr size_t INDEX_META_MAP_SIZE_BITS = 21;      // 2 MiB
    constexpr size_t INDEX_META_MAP_SIZE_MAX_BITS = 27;  // 128 MiB
    // Index-related tables
    constexpr size_t ID_MAPPER_MAP_SIZE_BITS = 24;      // 16 MiB
    constexpr size_t ID_MAPPER_MAP_SIZE_MAX_BITS = 33;  // 8 GiB
    constexpr size_t FILTER_MAP_SIZE_BITS = 24;         // 16 MiB
    constexpr size_t FILTER_MAP_SIZE_MAX_BITS = 36;     // 64 GiB
    constexpr size_t METADATA_MAP_SIZE_BITS = 27;       // 128 MiB
    constexpr size_t METADATA_MAP_SIZE_MAX_BITS = 39;   // 512 GiB
    constexpr size_t VECTOR_MAP_SIZE_BITS = 30;         // 1 GiB
    constexpr size_t VECTOR_MAP_SIZE_MAX_BITS = 42;     // 4 TiB

    constexpr size_t MAX_LINK_LIST_LOCKS = 65536;

    // Sparse Storage settings
    constexpr uint16_t MAX_BLOCK_SIZE = 128;    // Number of elements in a block
    constexpr uint32_t DEFAULT_VOCAB_SIZE = 0;  // 0 means dense vectors only
    constexpr uint8_t DEFAULT_QUANT_BITS = 8;
    constexpr uint16_t BLOCK_SPLIT_THRESHOLD =
            160;  // Bloc will be split if more than this many elements (including tombstones)

    // Maximum number of elements in the index
    constexpr size_t MAX_VECTORS_ADMIN = 1'000'000'000;

    // Buffer for early exit in search base layer
    constexpr int EARLY_EXIT_BUFFER_INSERT = 16;
    constexpr int EARLY_EXIT_BUFFER_QUERY = 8;

    // Pre-filter threshold - use pre-filter when cardinality is below this value
    constexpr size_t PREFILTER_CARDINALITY_THRESHOLD = 1000;
    // Use pre-filter if post-filter results are poor (less than this ratio of k)
    constexpr float PREFILTER_RESULT_RATIO_THRESHOLD = 0.25f;  // k/4

    //DEFAULT VALUES
    constexpr size_t DEFAULT_NUM_PARALLEL_INSERTS = 4;
    constexpr size_t DEFAULT_NUM_RECOVERY_THREADS = 16;
    constexpr size_t DEFAULT_MAX_MEMORY_GB = 24;
    constexpr bool DEFAULT_ENABLE_DEBUG_LOG = true;
    const std::string DEFAULT_AUTH_TOKEN = "";
    inline static std::string DEFAULT_USERNAME = "endee";
    constexpr size_t DEFAULT_SERVER_PORT = 8080;
    const std::string DEFAULT_SERVER_TYPE = "OSS";
    const std::string DEFAULT_DATA_DIR = "/mnt/data";
    constexpr size_t DEFAULT_MAX_ACTIVE_INDICES = 64;
    constexpr size_t DEFAULT_MAX_ELEMENTS = 100'000;
    constexpr size_t DEFAULT_MAX_ELEMENTS_INCREMENT = 100'000;
    constexpr size_t DEFAULT_MAX_ELEMENTS_INCREMENT_TRIGGER = 50'000;
    const std::string DEFAULT_SERVER_ID = "unknown";

    //For Backups
    static const int MAX_BACKUP_NAME_LENGTH = 200;

    /*Lib tkn: ONLY PLACEHOLDER*/
    const std::string DEFAULT_LIB_TOKEN = "3a5f08c7d9e1b2a43a5f08c7d9e1b2a4";

    // === Runtime-configurable settings ===
    inline static std::string SERVER_ID = [] {
        const char* env = std::getenv("NDD_SERVER_ID");
        return env ? std::string(env) : DEFAULT_SERVER_ID;
    }();
    inline static size_t SERVER_PORT = [] {
        const char* env = std::getenv("NDD_SERVER_PORT");
        return env ? std::stoull(env) : DEFAULT_SERVER_PORT;
    }();

    // Server type can be "SERVERLESS", "ON-PREM", "DEV"
    inline static std::string SERVER_TYPE = [] {
        const char* env = std::getenv("NDD_SERVER_TYPE");
        return env ? std::string(env) : DEFAULT_SERVER_TYPE;
    }();
    inline static std::string DATA_DIR = [] {
        const char* env = std::getenv("NDD_DATA_DIR");
        return env ? std::string(env) : DEFAULT_DATA_DIR;
    }();

    inline static size_t MAX_ACTIVE_INDICES = [] {
        const char* env = std::getenv("NDD_MAX_ACTIVE_INDICES");
        return env ? std::stoull(env) : DEFAULT_MAX_ACTIVE_INDICES;
    }();

    inline static size_t MAX_ELEMENTS = [] {
        const char* env = std::getenv("NDD_MAX_ELEMENTS");
        return env ? std::stoull(env) : DEFAULT_MAX_ELEMENTS;
    }();
    inline static size_t MAX_ELEMENTS_INCREMENT = [] {
        const char* env = std::getenv("NDD_MAX_ELEMENTS_INCREMENT");
        return env ? std::stoull(env) : DEFAULT_MAX_ELEMENTS_INCREMENT;
    }();
    inline static size_t MAX_ELEMENTS_INCREMENT_TRIGGER = [] {
        const char* env = std::getenv("NDD_MAX_INCREMENT_TRIGGER");
        return env ? std::stoull(env) : DEFAULT_MAX_ELEMENTS_INCREMENT_TRIGGER;
    }();

    // Number of parallel inserts. It will use this many threads to insert data in parallel
    inline static size_t NUM_PARALLEL_INSERTS = [] {
        const char* env = std::getenv("NDD_NUM_PARALLEL_INSERTS");
        return env ? std::stoull(env) : DEFAULT_NUM_PARALLEL_INSERTS;
    }();
    inline static size_t NUM_RECOVERY_THREADS = [] {
        const char* env = std::getenv("NDD_NUM_RECOVERY_THREADS");
        return env ? std::stoull(env) : DEFAULT_NUM_RECOVERY_THREADS;
    }();
    // TODO - Check if we can set this dynamically based on system memory
    // Max memory for HNSW index. It will evict the oldest index if it exceeds this limit
    inline static size_t MAX_MEMORY_GB = [] {
        const char* env = std::getenv("NDD_MAX_MEMORY_GB");
        return env ? std::stoull(env) : DEFAULT_MAX_MEMORY_GB;  // 24 GB by default
    }();

    inline static bool ENABLE_DEBUG_LOG = [] {
        const char* env = std::getenv("NDD_DEBUG_LOG");
        return env ? (std::string(env) == "1" || std::string(env) == "true")
                   : DEFAULT_ENABLE_DEBUG_LOG;
    }();

    // Authentication settings for open-source mode
    // If NDD_AUTH_TOKEN is set, authentication is required
    // If NDD_AUTH_TOKEN is empty/not set, all APIs work without authentication
    inline static std::string AUTH_TOKEN = [] {
        const char* env = std::getenv("NDD_AUTH_TOKEN");
        return env ? std::string(env) : DEFAULT_AUTH_TOKEN;
    }();

    inline static bool AUTH_ENABLED = !AUTH_TOKEN.empty();

    // Function to get all settings values as a multiline string
    inline std::string getAllSettingsAsString() {
        std::ostringstream oss;
        oss << "\n=== NDD Server ===\n";
        oss << "VERSION: " << VERSION << "\n";
        oss << "SERVER_ID: " << SERVER_ID << "\n";
        oss << "SERVER_PORT: " << SERVER_PORT << "\n";
        oss << "DATA_DIR: " << DATA_DIR << "\n";
        oss << "MAX_ELEMENTS: " << MAX_ELEMENTS << "\n";
        oss << "MAX_ELEMENTS_INCREMENT: " << MAX_ELEMENTS_INCREMENT << "\n";
        oss << "MAX_ELEMENTS_INCREMENT_TRIGGER: " << MAX_ELEMENTS_INCREMENT_TRIGGER << "\n";
        oss << "NUM_PARALLEL_INSERTS: " << NUM_PARALLEL_INSERTS << "\n";
        oss << "NUM_RECOVERY_THREADS: " << NUM_RECOVERY_THREADS << "\n";
        oss << "MAX_MEMORY_GB: " << MAX_MEMORY_GB << "\n";
        oss << "ENABLE_DEBUG_LOG: " << (ENABLE_DEBUG_LOG ? "true" : "false") << "\n";
        oss << "AUTH_ENABLED: " << (AUTH_ENABLED ? "true" : "false") << "\n";
        oss << "DEFAULT_USERNAME: " << DEFAULT_USERNAME << "\n";
        oss << "\n=== End Settings ===\n";
        return oss.str();
    }

}  //namespace settings