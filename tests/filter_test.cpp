#include <gtest/gtest.h>
#include <filesystem>
#include <string>
#include <vector>
#include "filter/filter.hpp"
#include "json/nlohmann_json.hpp"
#include "filter/numeric_index.hpp" // For Bucket test

namespace fs = std::filesystem;
using json = nlohmann::json;

TEST(BucketTest, Serialization) {
    ndd::filter::Bucket b;
    b.base_value = 100;
    b.add(105, 1);
    b.add(110, 2);
    
    auto bytes = b.serialize();
    EXPECT_GT(bytes.size(), 6);
    
    auto b2 = ndd::filter::Bucket::deserialize(bytes.data(), bytes.size(), 100);
    EXPECT_EQ(b2.ids.size(), 2);
    EXPECT_EQ(b2.ids[0], 1);
    EXPECT_EQ(b2.ids[1], 2);
}

class FilterTest : public ::testing::Test {
protected:
    std::string db_path;
    std::unique_ptr<Filter> filter;

    void SetUp() override {
        // Create a unique temporary directory for each test
        db_path = "./test_db_" + std::to_string(rand());
        if (fs::exists(db_path)) {
            fs::remove_all(db_path);
        }
        
        // Initialize Filter
        filter = std::make_unique<Filter>(db_path);
    }

    void TearDown() override {
        // Clean up
        filter.reset(); // Close DB environment first
        if (fs::exists(db_path)) {
            fs::remove_all(db_path);
        }
    }
};

TEST_F(FilterTest, CategoryFilterBasics) {
    // Add simple category filters
    // ID 1: City=Paris
    // ID 2: City=London
    // ID 3: City=Paris
    
    filter->add_to_filter("city", "Paris", 1);
    filter->add_to_filter("city", "London", 2);
    filter->add_to_filter("city", "Paris", 3);

    // Query for City=Paris
    json query = json::array({
        {{"city", {{"$eq", "Paris"}}}}
    });

    std::vector<ndd::idInt> ids = filter->getIdsMatchingFilter(query);
    
    // Should find 1 and 3
    EXPECT_EQ(ids.size(), 2);
    EXPECT_NE(std::find(ids.begin(), ids.end(), 1), ids.end());
    EXPECT_NE(std::find(ids.begin(), ids.end(), 3), ids.end());
    EXPECT_EQ(std::find(ids.begin(), ids.end(), 2), ids.end());
}

TEST_F(FilterTest, BooleanFilterBasics) {
    // Boolean is just a special category "0" or "1"
    // ID 10: Active=true
    // ID 11: Active=false
    
    // Using JSON add interface for variety
    filter->add_filters_from_json(10, R"({"is_active": true})");
    filter->add_filters_from_json(11, R"({"is_active": false})");

    // Query Active=true
    json query_true = json::array({
        {{"is_active", {{"$eq", true}}}}
    });
    
    auto ids_true = filter->getIdsMatchingFilter(query_true);
    EXPECT_EQ(ids_true.size(), 1);
    EXPECT_EQ(ids_true[0], 10);

    // Query Active=false
    json query_false = json::array({
        {{"is_active", {{"$eq", false}}}}
    });
    
    auto ids_false = filter->getIdsMatchingFilter(query_false);
    EXPECT_EQ(ids_false.size(), 1);
    EXPECT_EQ(ids_false[0], 11);
}

TEST_F(FilterTest, NumericFilterBasics) {
    // ID 100: Age=25
    // ID 101: Age=30
    // ID 102: Age=35
    
    filter->add_filters_from_json(100, R"({"age": 25})");
    filter->add_filters_from_json(101, R"({"age": 30})");
    filter->add_filters_from_json(102, R"({"age": 35})");

    // Range Query: 20 <= Age <= 32
    json query_range = json::array({
        {{"age", {{"$range", {20, 32}}}}}
    });

    auto ids = filter->getIdsMatchingFilter(query_range);
    
    // Should match 100 (25) and 101 (30)
    EXPECT_EQ(ids.size(), 2);
    bool found100 = false, found101 = false;
    for(auto id : ids) {
        if(id == 100) found100 = true;
        if(id == 101) found101 = true;
    }
    EXPECT_TRUE(found100);
    EXPECT_TRUE(found101);
}

TEST_F(FilterTest, FloatNumericFilter) {
    // ID 1: Price=10.5
    // ID 2: Price=20.0
    
    filter->add_filters_from_json(1, R"({"price": 10.5})");
    filter->add_filters_from_json(2, R"({"price": 20.0})");

    json query = json::array({
        {{"price", {{"$range", {10.0, 15.0}}}}}
    });

    auto ids = filter->getIdsMatchingFilter(query);
    EXPECT_EQ(ids.size(), 1);
    EXPECT_EQ(ids[0], 1);
}

TEST_F(FilterTest, MixedAndLogic) {
    // ID 1: City=NY, Age=30 (Match)
    // ID 2: City=NY, Age=40 (Age fail)
    // ID 3: City=LA, Age=30 (City fail)
    
    filter->add_filters_from_json(1, R"({"city": "NY", "age": 30})");
    filter->add_filters_from_json(2, R"({"city": "NY", "age": 40})");
    filter->add_filters_from_json(3, R"({"city": "LA", "age": 30})");

    // Filter: City=NY AND Age < 35
    json query = json::array({
        {{"city", {{"$eq", "NY"}}}},
        {{"age", {{"$range", {0, 35}}}}}
    });

    auto ids = filter->getIdsMatchingFilter(query);
    EXPECT_EQ(ids.size(), 1);
    EXPECT_EQ(ids[0], 1);
}

TEST_F(FilterTest, InOperator) {
    // ID 1: Color=Red
    // ID 2: Color=Blue
    // ID 3: Color=Green
    
    filter->add_to_filter("color", "Red", 1);
    filter->add_to_filter("color", "Blue", 2);
    filter->add_to_filter("color", "Green", 3);

    // Query: Color IN [Red, Green]
    json query = json::array({
        {{"color", {{"$in", {"Red", "Green"}}}}}
    });

    auto ids = filter->getIdsMatchingFilter(query);
    EXPECT_EQ(ids.size(), 2); // 1 and 3
}

TEST_F(FilterTest, DeleteFilter) {
    // ID 1: Tag=A
    filter->add_to_filter("tag", "A", 1);
    
    json query = json::array({
        {{"tag", {{"$eq", "A"}}}}
    });
    
    EXPECT_EQ(filter->countIdsMatchingFilter(query), 1);
    
    // Remove functionality test
    // Usually removal requires us to know what to remove or we remove entire ID?
    // The Filter class has: remove_from_filter(field, value, id)
    
    filter->remove_from_filter("tag", "A", 1);
    
    EXPECT_EQ(filter->countIdsMatchingFilter(query), 0);
}

TEST_F(FilterTest, NumericDelete) {
    // ID 1: Score=100
    filter->add_filters_from_json(1, R"({"score": 100})");
    
    // Check it exists
    json query = json::array({
        {{"score", {{"$eq", 100}}}}
    });
    EXPECT_EQ(filter->countIdsMatchingFilter(query), 1);
    
    // Remove
    // remove_filters_from_json uses the whole object
    filter->remove_filters_from_json(1, R"({"score": 100})");
    
    EXPECT_EQ(filter->countIdsMatchingFilter(query), 0);
}
