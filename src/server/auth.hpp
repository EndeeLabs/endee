#pragma once

#include <string>
#include <chrono>
#include <optional>
#include <filesystem>
#include <iostream>

#include "json/nlohmann_json.hpp"
#include "settings.hpp"

// Simplified for open-source mode - only Admin type exists
enum class UserType { Admin };

inline std::string userTypeToString(UserType type) {
    return "Admin";
}

inline UserType userTypeFromString(const std::string& type) {
    return UserType::Admin;
}

// Get max indices for each user type
inline int getMaxAllowedIndices(UserType type) {
    return settings::MAX_ACTIVE_INDICES;
}

// Get max vectors per index - No limits in open-source mode
inline size_t getMaxVectorsPerIndex(UserType type) {
    return settings::MAX_VECTORS_ADMIN;  // 1 billion vectors
}

struct User {
    std::string username;
    bool is_active;
    UserType user_type;
    std::chrono::system_clock::time_point created_at;

    nlohmann::json to_json() const {
        return {{"username", username},
                {"is_active", is_active},
                {"user_type", userTypeToString(user_type)},
                {"created_at", std::chrono::system_clock::to_time_t(created_at)}};
    }

    static User from_json(const nlohmann::json& j) {
        UserType type = UserType::Admin;
        if(j.contains("user_type")) {
            type = userTypeFromString(j["user_type"].get<std::string>());
        }

        return {j["username"].get<std::string>(),
                j["is_active"].get<bool>(),
                type,
                std::chrono::system_clock::from_time_t(j["created_at"].get<time_t>())};
    }
};

/**
 * Simplified AuthManager for open-source mode.
 *
 * Authentication modes:
 * 1. If NDD_AUTH_TOKEN is NOT set: All APIs work without authentication
 * 2. If NDD_AUTH_TOKEN is set: Token is required in Authorization header
 *
 * All operations use a single "default" user with Admin privileges (no limits).
 */
class AuthManager {
private:
    std::string base_dir_;

public:
    AuthManager(const std::string& base_dir) :
        base_dir_(base_dir) {
        // Create default user directory
        std::filesystem::path default_user_dir =
                std::filesystem::path(base_dir) / settings::DEFAULT_USERNAME;
        std::filesystem::create_directories(default_user_dir);

        if(settings::AUTH_ENABLED) {
            std::cerr << "Authentication ENABLED - NDD_AUTH_TOKEN is set" << std::endl;
        } else {
            std::cerr << "Authentication DISABLED - Running in open mode (no token required)"
                      << std::endl;
        }
    }

    ~AuthManager() = default;

    /**
     * Validate the provided token.
     *
     * @param provided_token The token from the Authorization header
     * @return The username ("default") if valid, empty string if invalid
     */
    std::string validateToken(const std::string& provided_token) {
        // If auth is disabled, always return default user
        if(!settings::AUTH_ENABLED) {
            return settings::DEFAULT_USERNAME;
        }

        // Compare against configured token
        if(provided_token == settings::AUTH_TOKEN) {
            return settings::DEFAULT_USERNAME;
        }

        return "";  // Invalid token
    }

    /**
     * Get user type - always returns Admin in open-source mode.
     */
    std::optional<UserType> getUserType(const std::string& username) { return UserType::Admin; }

    /**
     * Get user info - returns the default user.
     */
    std::optional<User> getUser(const std::string& username) {
        return User{settings::DEFAULT_USERNAME,
                    true,
                    UserType::Admin,
                    std::chrono::system_clock::now()};
    }

    /**
     * Get user info as JSON.
     */
    std::optional<nlohmann::json> getUserInfo(const std::string& requestingUser,
                                              const std::string& targetUser) {
        auto user = getUser(targetUser);
        if(!user) {
            return std::nullopt;
        }

        nlohmann::json userInfo = user->to_json();
        userInfo["token_count"] = 1;  // Single token (from env var)

        return userInfo;
    }
};
