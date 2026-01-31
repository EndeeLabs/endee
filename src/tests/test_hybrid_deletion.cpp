// test_hybrid_deletion.cpp
// Comprehensive tests to verify that deleted vectors don't appear in hybrid query results

#include <cassert>
#include <iostream>
#include <vector>
#include <random>
#include "core/ndd.hpp"
#include "settings.hpp"

// Helper function to generate random vector
std::vector<float> generateRandomVector(size_t dim, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    std::vector<float> vec(dim);
    for (size_t i = 0; i < dim; i++) {
        vec[i] = dis(gen);
    }
    return vec;
}

// Test 1: Basic hybrid deletion - delete with filter and verify not in results
void test_basic_hybrid_deletion() {
    std::cout << "Test 1: Basic Hybrid Deletion" << std::endl;
    
    std::string data_dir = "/tmp/test_hybrid_deletion_1";
    std::filesystem::remove_all(data_dir);
    std::filesystem::create_directories(data_dir);
    
    PersistenceConfig config;
    IndexManager manager(10, data_dir, config);
    
    // Create hybrid index
    IndexConfig idx_config{
        128,           // dim
        1000,          // sparse_dim
        10000,         // max_elements
        "cosine",      // space_type
        16,            // M
        200,           // ef_construction
        ndd::quant::QuantizationLevel::INT8,
        -1             // checksum
    };
    
    manager.createIndex("default/test_hybrid", idx_config);
    
    // Insert 100 hybrid vectors
    std::vector<ndd::HybridVectorObject> vectors;
    for (int i = 0; i < 100; i++) {
        ndd::HybridVectorObject vec;
        vec.id = "vec_" + std::to_string(i);
        vec.vector = generateRandomVector(128, i);
        vec.sparse_ids = {1, 2, 3};
        vec.sparse_values = {0.5f, 0.3f, 0.2f};
        
        // Set category filter
        if (i < 50) {
            vec.filter = R"({"category": "to_delete"})";
        } else {
            vec.filter = R"({"category": "keep"})";
        }
        vec.meta = "{}";
        vectors.push_back(vec);
    }
    
    assert(manager.addVectors("default/test_hybrid", vectors));
    
    // Delete vectors with category "to_delete"
    nlohmann::json filter_array = nlohmann::json::parse(
        R"([{"category": {"$eq": "to_delete"}}])"
    );
    size_t deleted = manager.deleteVectorsByFilter("default/test_hybrid", filter_array);
    
    std::cout << "  Deleted " << deleted << " vectors" << std::endl;
    assert(deleted == 50);
    
    // Perform hybrid query
    std::vector<float> dense_query = generateRandomVector(128, 999);
    std::vector<uint32_t> sparse_indices = {1, 2};
    std::vector<float> sparse_values = {0.6f, 0.4f};
    
    auto results = manager.searchKNN(
        "default/test_hybrid",
        dense_query,
        sparse_indices,
        sparse_values,
        100,  // k - request all
        nlohmann::json::array(),  // no additional filter
        true,  // include vectors
        0      // ef
    );
    
    assert(results.has_value());
    std::cout << "  Query returned " << results.value().size() << " results" << std::endl;
    
    // Verify: No deleted vectors in results
    for (const auto& result : results.value()) {
        nlohmann::json filter_obj = nlohmann::json::parse(result.filter);
        std::string category = filter_obj["category"].get<std::string>();
        
        // Should only have "keep" category, never "to_delete"
        if (category == "to_delete") {
            std::cerr << "  FAIL: Found deleted vector in results: " << result.id << std::endl;
            assert(false);
        }
        assert(category == "keep");
    }
    
    // Should have at most 50 results (the non-deleted ones)
    assert(results.value().size() <= 50);
    
    std::cout << "  PASS: No deleted vectors in hybrid query results" << std::endl;
    std::filesystem::remove_all(data_dir);
}

// Test 2: Dense-only query after deletion
void test_dense_only_deletion() {
    std::cout << "Test 2: Dense-Only Query After Deletion" << std::endl;
    
    std::string data_dir = "/tmp/test_hybrid_deletion_2";
    std::filesystem::remove_all(data_dir);
    std::filesystem::create_directories(data_dir);
    
    PersistenceConfig config;
    IndexManager manager(10, data_dir, config);
    
    IndexConfig idx_config{
        64,            // dim
        0,             // sparse_dim = 0 (dense only)
        10000,
        "cosine",
        16,
        200,
        ndd::quant::QuantizationLevel::INT8,
        -1
    };
    
    manager.createIndex("default/dense_test", idx_config);
    
    // Insert vectors
    std::vector<ndd::VectorObject> vectors;
    for (int i = 0; i < 50; i++) {
        ndd::VectorObject vec;
        vec.id = "vec_" + std::to_string(i);
        vec.vector = generateRandomVector(64, i);
        vec.filter = (i < 25) ? R"({"type": "A"})" : R"({"type": "B"})";
        vec.meta = "{}";
        vectors.push_back(vec);
    }
    
    assert(manager.addVectors("default/dense_test", vectors));
    
    // Delete type A
    nlohmann::json filter_array = nlohmann::json::parse(R"([{"type": {"$eq": "A"}}])");
    size_t deleted = manager.deleteVectorsByFilter("default/dense_test", filter_array);
    assert(deleted == 25);
    
    // Query with dense vector only
    std::vector<float> query = generateRandomVector(64, 888);
    auto results = manager.searchKNN(
        "default/dense_test",
        query,
        50,  // k
        nlohmann::json::array(),
        false,
        0
    );
    
    assert(results.has_value());
    std::cout << "  Dense query returned " << results.value().size() << " results" << std::endl;
    
    // Verify no type A in results
    for (const auto& result : results.value()) {
        nlohmann::json filter_obj = nlohmann::json::parse(result.filter);
        assert(filter_obj["type"].get<std::string>() == "B");
    }
    
    assert(results.value().size() <= 25);
    std::cout << "  PASS: Dense-only query respects deletions" << std::endl;
    std::filesystem::remove_all(data_dir);
}

// Test 3: Sparse-only query after deletion
void test_sparse_only_deletion() {
    std::cout << "Test 3: Sparse-Only Query After Deletion" << std::endl;
    
    std::string data_dir = "/tmp/test_hybrid_deletion_3";
    std::filesystem::remove_all(data_dir);
    std::filesystem::create_directories(data_dir);
    
    PersistenceConfig config;
    IndexManager manager(10, data_dir, config);
    
    IndexConfig idx_config{
        64,
        5000,  // sparse_dim
        10000,
        "cosine",
        16,
        200,
        ndd::quant::QuantizationLevel::INT8,
        -1
    };
    
    manager.createIndex("default/sparse_test", idx_config);
    
    // Insert hybrid vectors
    std::vector<ndd::HybridVectorObject> vectors;
    for (int i = 0; i < 50; i++) {
        ndd::HybridVectorObject vec;
        vec.id = "vec_" + std::to_string(i);
        vec.vector = generateRandomVector(64, i);
        vec.sparse_ids = {static_cast<uint32_t>(i % 100), static_cast<uint32_t>((i + 1) % 100)};
        vec.sparse_values = {0.7f, 0.3f};
        vec.filter = (i < 25) ? R"({"group": "X"})" : R"({"group": "Y"})";
        vec.meta = "{}";
        vectors.push_back(vec);
    }
    
    assert(manager.addVectors("default/sparse_test", vectors));
    
    // Delete group X
    nlohmann::json filter_array = nlohmann::json::parse(R"([{"group": {"$eq": "X"}}])");
    size_t deleted = manager.deleteVectorsByFilter("default/sparse_test", filter_array);
    assert(deleted == 25);
    
    // Query with sparse only (empty dense vector)
    std::vector<float> empty_dense;
    std::vector<uint32_t> sparse_q_indices = {5, 10};
    std::vector<float> sparse_q_values = {0.8f, 0.2f};
    
    auto results = manager.searchKNN(
        "default/sparse_test",
        empty_dense,
        sparse_q_indices,
        sparse_q_values,
        50,
        nlohmann::json::array(),
        false,
        0
    );
    
    assert(results.has_value());
    std::cout << "  Sparse query returned " << results.value().size() << " results" << std::endl;
    
    // Verify no group X in results
    for (const auto& result : results.value()) {
        nlohmann::json filter_obj = nlohmann::json::parse(result.filter);
        std::string group = filter_obj["group"].get<std::string>();
        assert(group == "Y");
    }
    
    assert(results.value().size() <= 25);
    std::cout << "  PASS: Sparse-only query respects deletions" << std::endl;
    std::filesystem::remove_all(data_dir);
}

// Test 4: Delete all vectors - query should return empty
void test_delete_all() {
    std::cout << "Test 4: Delete All Vectors" << std::endl;
    
    std::string data_dir = "/tmp/test_hybrid_deletion_4";
    std::filesystem::remove_all(data_dir);
    std::filesystem::create_directories(data_dir);
    
    PersistenceConfig config;
    IndexManager manager(10, data_dir, config);
    
    IndexConfig idx_config{128, 1000, 10000, "cosine", 16, 200, 
                           ndd::quant::QuantizationLevel::INT8, -1};
    manager.createIndex("default/test_all", idx_config);
    
    // Insert vectors
    std::vector<ndd::HybridVectorObject> vectors;
    for (int i = 0; i < 30; i++) {
        ndd::HybridVectorObject vec;
        vec.id = "vec_" + std::to_string(i);
        vec.vector = generateRandomVector(128, i);
        vec.sparse_ids = {1, 2};
        vec.sparse_values = {0.5f, 0.5f};
        vec.filter = R"({"status": "active"})";
        vec.meta = "{}";
        vectors.push_back(vec);
    }
    
    assert(manager.addVectors("default/test_all", vectors));
    
    // Delete all
    nlohmann::json filter_array = nlohmann::json::parse(R"([{"status": {"$eq": "active"}}])");
    size_t deleted = manager.deleteVectorsByFilter("default/test_all", filter_array);
    assert(deleted == 30);
    
    // Query - should return empty
    auto results = manager.searchKNN(
        "default/test_all",
        generateRandomVector(128, 777),
        {1, 2},
        {0.5f, 0.5f},
        50,
        nlohmann::json::array(),
        false,
        0
    );
    
    assert(results.has_value());
    assert(results.value().empty());
    std::cout << "  PASS: Query returns empty after deleting all vectors" << std::endl;
    std::filesystem::remove_all(data_dir);
}

// Test 5: Delete and re-insert with same ID
void test_delete_and_reinsert() {
    std::cout << "Test 5: Delete and Re-insert" << std::endl;
    
    std::string data_dir = "/tmp/test_hybrid_deletion_5";
    std::filesystem::remove_all(data_dir);
    std::filesystem::create_directories(data_dir);
    
    PersistenceConfig config;
    IndexManager manager(10, data_dir, config);
    
    IndexConfig idx_config{64, 100, 10000, "cosine", 16, 200, 
                           ndd::quant::QuantizationLevel::INT8, -1};
    manager.createIndex("default/reinsert_test", idx_config);
    
    // Insert initial vector
    std::vector<ndd::HybridVectorObject> initial;
    ndd::HybridVectorObject vec1;
    vec1.id = "special_vec";
    vec1.vector = generateRandomVector(64, 100);
    vec1.sparse_ids = {1};
    vec1.sparse_values = {1.0f};
    vec1.filter = R"({"version": 1})";
    vec1.meta = "{}";
    initial.push_back(vec1);
    
    assert(manager.addVectors("default/reinsert_test", initial));
    
    // Query - should find it
    auto results1 = manager.searchKNN(
        "default/reinsert_test",
        generateRandomVector(64, 100),
        {1},
        {1.0f},
        10,
        nlohmann::json::array(),
        false,
        0
    );
    
    assert(results1.has_value());
    assert(!results1.value().empty());
    
    // Delete by ID
    assert(manager.deleteVector("default/reinsert_test", "special_vec"));
    
    // Query - should NOT find it
    auto results2 = manager.searchKNN(
        "default/reinsert_test",
        generateRandomVector(64, 100),
        {1},
        {1.0f},
        10,
        nlohmann::json::array(),
        false,
        0
    );
    
    assert(results2.has_value());
    // Should be empty or not contain special_vec
    for (const auto& r : results2.value()) {
        assert(r.id != "special_vec");
    }
    
    // Re-insert with same ID but different data
    std::vector<ndd::HybridVectorObject> reinsert;
    ndd::HybridVectorObject vec2;
    vec2.id = "special_vec";  // Same ID
    vec2.vector = generateRandomVector(64, 200);  // Different vector
    vec2.sparse_ids = {2};
    vec2.sparse_values = {1.0f};
    vec2.filter = R"({"version": 2})";  // Different filter
    vec2.meta = "{}";
    reinsert.push_back(vec2);
    
    assert(manager.addVectors("default/reinsert_test", reinsert));
    
    // Query - should find NEW version
    auto results3 = manager.searchKNN(
        "default/reinsert_test",
        generateRandomVector(64, 200),
        {2},
        {1.0f},
        10,
        nlohmann::json::array(),
        false,
        0
    );
    
    assert(results3.has_value());
    bool found_new_version = false;
    for (const auto& r : results3.value()) {
        if (r.id == "special_vec") {
            // Verify it's the new version
            nlohmann::json filter_obj = nlohmann::json::parse(r.filter);
            assert(filter_obj["version"].get<int>() == 2);
            found_new_version = true;
        }
    }
    assert(found_new_version);
    
    std::cout << "  PASS: Delete and re-insert works correctly" << std::endl;
    std::filesystem::remove_all(data_dir);
}

// Test 6: Multiple deletions in sequence
void test_multiple_deletions() {
    std::cout << "Test 6: Multiple Sequential Deletions" << std::endl;
    
    std::string data_dir = "/tmp/test_hybrid_deletion_6";
    std::filesystem::remove_all(data_dir);
    std::filesystem::create_directories(data_dir);
    
    PersistenceConfig config;
    IndexManager manager(10, data_dir, config);
    
    IndexConfig idx_config{64, 100, 10000, "cosine", 16, 200, 
                           ndd::quant::QuantizationLevel::INT8, -1};
    manager.createIndex("default/multi_del", idx_config);
    
    // Insert vectors with priority levels
    std::vector<ndd::HybridVectorObject> vectors;
    for (int i = 0; i < 100; i++) {
        ndd::HybridVectorObject vec;
        vec.id = "vec_" + std::to_string(i);
        vec.vector = generateRandomVector(64, i);
        vec.sparse_ids = {static_cast<uint32_t>(i % 10)};
        vec.sparse_values = {1.0f};
        
        int priority = i % 5;  // 0-4
        vec.filter = R"({"priority": )" + std::to_string(priority) + R"(})";
        vec.meta = "{}";
        vectors.push_back(vec);
    }
    
    assert(manager.addVectors("default/multi_del", vectors));
    
    // Delete priority 0 (20 vectors)
    nlohmann::json filter1 = nlohmann::json::parse(R"([{"priority": {"$eq": 0}}])");
    assert(manager.deleteVectorsByFilter("default/multi_del", filter1) == 20);
    
    // Query - should have 80 results
    auto r1 = manager.searchKNN("default/multi_del", generateRandomVector(64, 999),
                                {1}, {1.0f}, 100, nlohmann::json::array(), false, 0);
    assert(r1.has_value());
    assert(r1.value().size() <= 80);
    
    // Delete priority 1 (20 more vectors)
    nlohmann::json filter2 = nlohmann::json::parse(R"([{"priority": {"$eq": 1}}])");
    assert(manager.deleteVectorsByFilter("default/multi_del", filter2) == 20);
    
    // Query - should have 60 results
    auto r2 = manager.searchKNN("default/multi_del", generateRandomVector(64, 998),
                                {2}, {1.0f}, 100, nlohmann::json::array(), false, 0);
    assert(r2.has_value());
    assert(r2.value().size() <= 60);
    
    // Delete priority 2 (20 more vectors)
    nlohmann::json filter3 = nlohmann::json::parse(R"([{"priority": {"$eq": 2}}])");
    assert(manager.deleteVectorsByFilter("default/multi_del", filter3) == 20);
    
    // Query - should have 40 results
    auto r3 = manager.searchKNN("default/multi_del", generateRandomVector(64, 997),
                                {3}, {1.0f}, 100, nlohmann::json::array(), false, 0);
    assert(r3.has_value());
    assert(r3.value().size() <= 40);
    
    // Verify remaining are only priority 3 and 4
    for (const auto& result : r3.value()) {
        nlohmann::json filter_obj = nlohmann::json::parse(result.filter);
        int priority = filter_obj["priority"].get<int>();
        assert(priority == 3 || priority == 4);
    }
    
    std::cout << "  PASS: Multiple sequential deletions work correctly" << std::endl;
    std::filesystem::remove_all(data_dir);
}

int main() {
    std::cout << "=== Running Hybrid Index Deletion Tests ===" << std::endl << std::endl;
    
    try {
        test_basic_hybrid_deletion();
        test_dense_only_deletion();
        test_sparse_only_deletion();
        test_delete_all();
        test_delete_and_reinsert();
        test_multiple_deletions();
        
        std::cout << std::endl << "=== ALL TESTS PASSED ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << std::endl << "=== TEST FAILED ===" << std::endl;
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}