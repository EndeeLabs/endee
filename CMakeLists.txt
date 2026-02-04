# For Server x86_64 install clang for fp16 support and use the following commands
cmake_minimum_required(VERSION 3.14)
project(ndd)


#check for the correct clang version
find_program(CLANG_CANDIDATE_C
    NAMES clang-21 clang-20 clang-19 clang
)

find_program(CLANG_CANDIDATE_CXX
    NAMES clang++-21 clang++-20 clang++-19 clang++
)

if (NOT CLANG_CANDIDATE_C OR NOT CLANG_CANDIDATE_CXX)
    message(FATAL_ERROR "Clang not found. Please install clang >= 19.")
endif()

# Query clang version
execute_process(
    COMMAND ${CLANG_CANDIDATE_C} --version
    OUTPUT_VARIABLE CLANG_VERSION_OUTPUT
    ERROR_VARIABLE CLANG_VERSION_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (NOT CLANG_VERSION_OUTPUT)
    message(FATAL_ERROR "Failed to query clang version.")
endif()

# Extract major version
string(REGEX MATCH "clang version ([0-9]+)" _match "${CLANG_VERSION_OUTPUT}")
set(CLANG_VERSION_MAJOR "${CMAKE_MATCH_1}")

if (NOT CLANG_VERSION_MAJOR)
    message(FATAL_ERROR
        "Unable to determine clang version from:\n${CLANG_VERSION_OUTPUT}\n"
    )
endif()

if (CLANG_VERSION_MAJOR LESS 17)
    message(FATAL_ERROR
        "Clang ${CLANG_VERSION_MAJOR} detected, but clang >= 17 is required "
    )
endif()

# Lock compilers
set(CMAKE_C_COMPILER   "${CLANG_CANDIDATE_C}"   CACHE STRING "" FORCE)
set(CMAKE_CXX_COMPILER "${CLANG_CANDIDATE_CXX}" CACHE STRING "" FORCE)

message(STATUS "Using Clang ${CLANG_VERSION_MAJOR}")
message(STATUS "C compiler : ${CMAKE_C_COMPILER}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")



# Set C++ standard explicitly
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add DEBUG option
option(DEBUG "Build with debug symbols and without optimization" OFF)

# Configure build type based on DEBUG option
if(DEBUG)
    set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3 -O0")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g3 -O0")
    message(STATUS "Debug build enabled - adding debug symbols")
else()
    set(CMAKE_BUILD_TYPE Release)
    # Add minimal debug symbols even in Release mode for better crash reports
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -g")
endif()

# Add ND_DEBUG option for logging (separate from build type)
option(ND_DEBUG "Enable NDD debug logging and timing" OFF)
if(ND_DEBUG)
    add_definitions(-DND_DEBUG)
endif()

# SIMD Optimization Options
option(USE_AVX512 "Enable AVX512 (F, BW, VNNI, FP16)" OFF)
option(USE_AVX2   "Enable AVX2 (FMA, F16C)" OFF)
option(USE_SVE2   "Enable SVE2 (INT8/16, FP16)" OFF)
option(USE_NEON   "Enable NEON (FP16, DotProd)" OFF)

# Check if any SIMD option is selected
if(NOT USE_AVX512 AND NOT USE_AVX2 AND NOT USE_SVE2 AND NOT USE_NEON)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch64)")
        message(FATAL_ERROR "ARM architecture detected but no SIMD option selected.\n"
                            "Please specify one of the following flags:\n"
                            "  -DUSE_SVE2=ON  : For ARMv9/SVE2 capable processors (requires SVE2, FP16)\n"
                            "  -DUSE_NEON=ON  : For standard ARMv8/NEON processors (requires FP16, DotProd)")
    else()
        message(FATAL_ERROR "x86 architecture detected but no SIMD option selected.\n"
                            "Please specify one of the following flags:\n"
                            "  -DUSE_AVX512=ON : For processors with AVX512F, BW, VNNI, FP16\n"
                            "  -DUSE_AVX2=ON   : For processors with AVX2, FMA, F16C")
    endif()
endif()

# Include FetchContent for dependencies
include(FetchContent)

# =======================
# Find OpenSSL
# =======================
find_package(OpenSSL REQUIRED)
include_directories(${OPENSSL_INCLUDE_DIR})

# =======================
# Find libcurl
# =======================
find_package(CURL REQUIRED)
include_directories(${CURL_INCLUDE_DIRS})

# =======================
# Find or Fetch ASIO
# =======================
set(ASIO_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/asio/asio/include")
if(NOT EXISTS "${ASIO_PATH}/asio.hpp")
    message(STATUS "ASIO not found, downloading...")
    FetchContent_Declare(
        asio
        GIT_REPOSITORY https://github.com/chriskohlhoff/asio.git
        GIT_TAG        asio-1-36-0
    )
    FetchContent_MakeAvailable(asio)
    set(ASIO_INCLUDE_DIR "${CMAKE_BINARY_DIR}/_deps/asio-src/asio/include")
else()
    set(ASIO_INCLUDE_DIR "${ASIO_PATH}")
endif()

# =======================
# Find or Fetch Crow
# =======================
set(CROW_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/crow/include")
if(NOT EXISTS "${CROW_PATH}/crow.h")
    message(STATUS "Crow not found, downloading...")
    FetchContent_Declare(
        crow
        GIT_REPOSITORY https://github.com/CrowCpp/Crow.git
        GIT_TAG        v1.3.0
    )
    FetchContent_MakeAvailable(crow)
    set(CROW_INCLUDE_DIR "${CMAKE_BINARY_DIR}/_deps/crow-src/include")
else()
    set(CROW_INCLUDE_DIR "${CROW_PATH}")
endif()

# =======================
# Find or Fetch msgpack-c (header-only)
# =======================
set(MSGPACK_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/msgpack/include")
if(NOT EXISTS "${MSGPACK_PATH}/msgpack.hpp")
    message(STATUS "msgpack-c not found, downloading...")
    FetchContent_Declare(
        msgpack
        GIT_REPOSITORY https://github.com/msgpack/msgpack-c.git
        GIT_TAG        cpp-6.1.1
    )
    # Only download, don't configure/build (header-only library)
    FetchContent_GetProperties(msgpack)
    if(NOT msgpack_POPULATED)
        cmake_policy(PUSH)
        # Only set the policy if the current CMake version knows about it (CMake 3.29+)
        if(POLICY CMP0169)
                cmake_policy(SET CMP0169 OLD)
        endif()
        FetchContent_Populate(msgpack)
#cmake_policy(SET CMP0169 OLD) # Suppress warning about FetchContent_Populate
#        FetchContent_Populate(msgpack)
        cmake_policy(POP)
    endif()
    set(MSGPACK_INCLUDE_DIR "${msgpack_SOURCE_DIR}/include")
else()
    set(MSGPACK_INCLUDE_DIR "${MSGPACK_PATH}")
endif()

# =======================
# Find or Fetch libarchive
# =======================
set(LIBARCHIVE_PATH "${CMAKE_BINARY_DIR}/third_party/libarchive/libarchive")
if(NOT EXISTS "${LIBARCHIVE_PATH}/archive.h")
    message(STATUS "libarchive not found, downloading...")
    FetchContent_Declare(
        libarchive
        GIT_REPOSITORY https://github.com/libarchive/libarchive.git
        GIT_TAG        v3.8.5
    )
    # Configure libarchive options - disable tools and tests
    FetchContent_GetProperties(libarchive)
    if(NOT libarchive_POPULATED)
        set(ENABLE_TEST OFF CACHE BOOL "" FORCE)
        set(ENABLE_TAR OFF CACHE BOOL "" FORCE)
        set(ENABLE_CPIO OFF CACHE BOOL "" FORCE)
        set(ENABLE_CAT OFF CACHE BOOL "" FORCE)
        set(ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(libarchive)
    endif()
    set(LIBARCHIVE_INCLUDE_DIR "${libarchive_SOURCE_DIR}/libarchive")
else()
    set(LIBARCHIVE_INCLUDE_DIR "${LIBARCHIVE_PATH}")
endif()

# =======================
# Find Other Dependencies
# =======================
find_package(Threads REQUIRED)

# Find MDBX (replaces LMDB)
find_path(LMDB_INCLUDE_DIR NAMES mdbx.h PATHS ${CMAKE_SOURCE_DIR}/third_party/mdbx NO_DEFAULT_PATH)
file(GLOB LMDB_SOURCES ${CMAKE_SOURCE_DIR}/third_party/mdbx/*.c)

# -----------------------
# Derive binary name
# -----------------------
set(NDD_BINARY_NAME "ndd")

if(USE_AVX512)
    set(NDD_BINARY_NAME "ndd-avx512")
elseif(USE_AVX2)
    set(NDD_BINARY_NAME "ndd-avx2")
elseif(USE_SVE2)
    set(NDD_BINARY_NAME "ndd-sve2")
elseif(USE_NEON)
    # macOS Apple Silicon
    if(APPLE AND (CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64"))
        set(NDD_BINARY_NAME "ndd-neon-darwin")
    else()
        set(NDD_BINARY_NAME "ndd-neon")
    endif()
endif()

message(STATUS "Binary name: ${NDD_BINARY_NAME}")



# Create the target
add_executable(${NDD_BINARY_NAME} src/main.cpp ${LMDB_SOURCES} third_party/roaring_bitmap/roaring.c)

# Set MDBX-specific compile flags
set_source_files_properties(${LMDB_SOURCES} PROPERTIES
    COMPILE_FLAGS "-DMDBX_BUILD_SHARED_LIBRARY=0 -DMDBX_BUILD_FLAGS=\\\"NDD_EMBEDDED\\\""
)

# Include directories
target_include_directories(${NDD_BINARY_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/server
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/storage
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CROW_INCLUDE_DIR}
    ${MSGPACK_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/roaring
    ${LMDB_INCLUDE_DIR}
    ${ASIO_INCLUDE_DIR}
    ${OPENSSL_INCLUDE_DIR}
    ${CURL_INCLUDE_DIRS}
)

# Set compiler flags
if(NOT DEBUG)
    target_compile_options(${NDD_BINARY_NAME} PRIVATE -O3 -ffast-math -fno-finite-math-only)
endif()

# Apply Flags based on selection
if(USE_AVX512)
    message(STATUS "SIMD: AVX512 enabled (F, BW, VNNI, FP16)")
    target_compile_options(${NDD_BINARY_NAME} PRIVATE -mavx512f -mavx512bw -mavx512vnni -mavx512fp16 -mavx512vpopcntdq)
    target_compile_definitions(${NDD_BINARY_NAME} PRIVATE USE_AVX512)
elseif(USE_AVX2)
    message(STATUS "SIMD: AVX2 enabled")
    target_compile_options(${NDD_BINARY_NAME} PRIVATE -mavx2 -mfma -mf16c)
    target_compile_definitions(${NDD_BINARY_NAME} PRIVATE USE_AVX2)
elseif(USE_SVE2)
    message(STATUS "SIMD: SVE2 enabled (ARMv8.6-a + SVE2 + FP16)")
    target_compile_options(${NDD_BINARY_NAME} PRIVATE -march=armv8.6-a+sve2+fp16)
    target_compile_definitions(${NDD_BINARY_NAME} PRIVATE USE_SVE2)
elseif(USE_NEON)
    message(STATUS "SIMD: NEON enabled")
    if(APPLE AND (CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64"))
        target_compile_options(${NDD_BINARY_NAME} PRIVATE -mcpu=native)
    else()
        target_compile_options(${NDD_BINARY_NAME} PRIVATE -march=armv8.2-a+fp16+fp16fml+dotprod)
    endif()
    target_compile_definitions(${NDD_BINARY_NAME} PRIVATE USE_NEON)
endif()

# Add ASIO definitions
target_compile_definitions(${NDD_BINARY_NAME} PRIVATE 
    ASIO_STANDALONE 
    ASIO_HAS_STD_CHRONO 
    ASIO_HAS_STD_STRING_VIEW
    MDB_MAXKEYSIZE=512
)

# Link libraries
target_link_libraries(${NDD_BINARY_NAME} PRIVATE
    Threads::Threads
    OpenSSL::SSL
    OpenSSL::Crypto
    CURL::libcurl
    archive_static
)

# Installation rules
install(TARGETS ${NDD_BINARY_NAME} RUNTIME DESTINATION bin)

# Print configuration info
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Debug mode: ${DEBUG}")
message(STATUS "Processor: ${CMAKE_SYSTEM_PROCESSOR}")
if(USE_AVX512)
    message(STATUS "SIMD Mode: AVX512")
elseif(USE_AVX2)
    message(STATUS "SIMD Mode: AVX2")
elseif(USE_SVE2)
    message(STATUS "SIMD Mode: SVE2")
elseif(USE_NEON)
    message(STATUS "SIMD Mode: NEON")
endif()
message(STATUS "ASIO include dir: ${ASIO_INCLUDE_DIR}")
message(STATUS "LMDB include dir: ${LMDB_INCLUDE_DIR}")
message(STATUS "OpenSSL include dir: ${OPENSSL_INCLUDE_DIR}")
