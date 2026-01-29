// System includes
#include <iostream>
#include <filesystem>

// ASIO/Crow related
#include "crow/query_string.h"
#include "crow/http_parser_merged.h"
#include "crow/ci_map.h"
#include "crow/TinySHA1.hpp"
#include "crow/settings.h"
#include "crow/socket_adaptors.h"
#include "crow/json.h"
#include "crow/mustache.h"
#include "crow/logging.h"
#include "crow/task_timer.h"
#include "crow/utility.h"
#include "crow/common.h"
#include "crow/http_request.h"
#include "crow/websocket.h"
#include "crow/parser.h"
#include "crow/http_response.h"
#include "crow/multipart.h"
#include "crow/multipart_view.h"
#include "crow/routing.h"
#include "crow/middleware.h"
#include "crow/middleware_context.h"
#include "crow/compression.h"
#include "crow/http_connection.h"
#include "crow/http_server.h"
#include "crow/app.h"

// local includes
#include "settings.hpp"
#include "core/ndd.hpp"
#include "auth.hpp"
#include "quant/common.hpp"
#include "cpu_compat_check/check_avx_compat.hpp"
#include "cpu_compat_check/check_arm_compat.hpp"

using ndd::quant::quantLevelToString;
using ndd::quant::stringToQuantLevel;

// Global CPU feature flags (initialized once at startup)
// Note: No longer needed since VNNI INT16 is part of standard AVX512-VNNI

// Authentication middleware for open-source mode
// If NDD_AUTH_TOKEN is set: token is required
// If NDD_AUTH_TOKEN is not set: all requests are allowed
struct AuthMiddleware : crow::ILocalMiddleware {
    AuthManager& auth_manager;

    AuthMiddleware(AuthManager& am) :
        auth_manager(am) {}

    struct context {
        std::string username;
    };

    void before_handle(crow::request& req, crow::response& res, context& ctx) {
        ctx.username = settings::DEFAULT_USERNAME;  // Always "default"

        if(!settings::AUTH_ENABLED) {
            return;  // No auth required - open mode
        }

        // Auth is enabled - token is REQUIRED
        auto auth_header = req.get_header_value("Authorization");
        if(auth_header.empty()) {
            res.code = 401;
            res.write("Authorization header required");
            res.end();
            return;
        }

        if(auth_header != settings::AUTH_TOKEN) {
            res.code = 401;
            res.write("Invalid token");
            res.end();
            return;
        }
    }

    void after_handle(crow::request&, crow::response&, context&) {}
};
// Helper function to send error messages in JSON format
inline crow::response json_error(int code, const std::string& message) {
    crow::json::wvalue err_json({{"error", message}});
    return crow::response(code, err_json.dump());
}
// Special helper function to log and send error messages in JSON format for 500 errors
inline crow::response
json_error_500(const std::string& username, const std::string& path, const std::string& message) {
    LOG_ERROR("500 Error | user: " << username << " | path: " << path << " | message: " << message);

    crow::json::wvalue err_json({{"error", message}});
    return crow::response(500, err_json.dump());
}

/**
 * Checks if the CPU is compatible with all
 * the instruction sets being used for x86, ARM and MAC Mxx
 */
bool is_cpu_compatible() {
    bool ret = true;

#if defined(USE_AVX2) && (defined(__x86_64__) || defined(_M_X64))
    ret &= is_avx2_compatible();
#endif  //AVX2 checks

#if defined(USE_AVX512) && (defined(__x86_64__) || defined(_M_X64))
    ret &= is_avx512_compatible();
#endif  //AVX512 checks

#if defined(USE_NEON)
    ret &= is_neon_compatible();
#endif

#if defined(USE_SVE2)
    ret &= is_sve2_compatible();
#endif

    return ret;
}

// Read file contents
std::string read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if(!file.is_open()) {
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Get MIME type based on file extension
std::string get_mime_type(const std::string& path) {
    if(path.ends_with(".html")) {
        return "text/html";
    }
    if(path.ends_with(".css")) {
        return "text/css";
    }
    if(path.ends_with(".js")) {
        return "application/javascript";
    }
    if(path.ends_with(".json")) {
        return "application/json";
    }
    if(path.ends_with(".png")) {
        return "image/png";
    }
    if(path.ends_with(".jpg") || path.ends_with(".jpeg")) {
        return "image/jpeg";
    }
    if(path.ends_with(".svg")) {
        return "image/svg+xml";
    }
    if(path.ends_with(".ico")) {
        return "image/x-icon";
    }
    if(path.ends_with(".woff")) {
        return "font/woff";
    }
    if(path.ends_with(".woff2")) {
        return "font/woff2";
    }
    if(path.ends_with(".ttf")) {
        return "font/ttf";
    }
    if(path.ends_with(".map")) {
        return "application/json";
    }
    return "application/octet-stream";
}

// Check if file exists
bool file_exists(const std::string& path) {
    return std::filesystem::exists(path) && std::filesystem::is_regular_file(path);
}

int main(int argc, char** argv) {

    if(!is_cpu_compatible()) {
        printf("CPU is not compatible. Can't run Endee\n");
        return 0;
    }
    LOG_DEBUG("SERVER_ID: " << settings::SERVER_ID);
    LOG_DEBUG("SERVER_PORT: " << settings::SERVER_PORT);
    LOG_DEBUG("DATA_DIR: " << settings::DATA_DIR);
    LOG_DEBUG("NUM_PARALLEL_INSERTS: " << settings::NUM_PARALLEL_INSERTS);
    LOG_DEBUG("NUM_RECOVERY_THREADS: " << settings::NUM_RECOVERY_THREADS);
    LOG_DEBUG("MAX_MEMORY_GB: " << settings::MAX_MEMORY_GB);
    LOG_DEBUG("ENABLE_DEBUG_LOG: " << settings::ENABLE_DEBUG_LOG);
    LOG_DEBUG("AUTH_TOKEN: " << settings::AUTH_TOKEN);
    LOG_DEBUG("AUTH_ENABLED: " << settings::AUTH_ENABLED);
    LOG_DEBUG("DEFAULT_USERNAME: " << settings::DEFAULT_USERNAME);
    LOG_DEBUG("DEFAULT_SERVER_TYPE: " << settings::DEFAULT_SERVER_TYPE);
    LOG_DEBUG("DEFAULT_DATA_DIR: " << settings::DEFAULT_DATA_DIR);
    LOG_DEBUG("DEFAULT_MAX_ACTIVE_INDICES: " << settings::DEFAULT_MAX_ACTIVE_INDICES);
    LOG_DEBUG("DEFAULT_MAX_ELEMENTS: " << settings::DEFAULT_MAX_ELEMENTS);
    LOG_DEBUG("DEFAULT_MAX_ELEMENTS_INCREMENT: " << settings::DEFAULT_MAX_ELEMENTS_INCREMENT);
    LOG_DEBUG("DEFAULT_MAX_ELEMENTS_INCREMENT_TRIGGER: "
              << settings::DEFAULT_MAX_ELEMENTS_INCREMENT_TRIGGER);

    // Path to React build directory
    // Get the executable's directory and resolve frontend/dist relative to it
    // Get executable directory using argv[0] (cross-platform)
    std::filesystem::path exe_path = std::filesystem::canonical(argv[0]).parent_path();
    const std::string BUILD_DIR = (exe_path / "../frontend/dist").string();
    const std::string INDEX_PATH = BUILD_DIR + "/index.html";

    // Initialize index manager with persistence config
    std::string data_dir = settings::DATA_DIR;
    std::filesystem::create_directories(data_dir);

    PersistenceConfig persistence_config{
            settings::SAVE_EVERY_N_UPDATES,                        // Save every n updates
            std::chrono::minutes(settings::SAVE_EVERY_N_MINUTES),  // Save every n minutes
            true                                                   // Save on shutdown
    };
    // Initialize auth manager and user manager
    LOG_INFO("Starting the server");
    AuthManager auth_manager(data_dir);
    LOG_INFO("Created auth manager");
    IndexManager index_manager(settings::MAX_ACTIVE_INDICES, data_dir, persistence_config);
    LOG_INFO("Created index manager");

    // Initialize the app
    crow::App<AuthMiddleware> app{AuthMiddleware(auth_manager)};

    // ========== GENERAL ==========
    // Health check endpoint (no auth required)
    CROW_ROUTE(app, "/api/v1/health").methods("GET"_method)([](const crow::request& req) {
        crow::json::wvalue response(
                {{"status", "ok"},
                 {"timestamp", std::chrono::system_clock::now().time_since_epoch().count()}});
        return crow::response(200, response.dump());
    });

    // ========= USER ENDPOINTS ==========
    // Get user info - returns default user info
    CROW_ROUTE(app, "/api/v1/users/<string>/info")
            .CROW_MIDDLEWARES(app, AuthMiddleware)
            .methods("GET"_method)([&auth_manager, &app](const crow::request& req,
                                                         const std::string& target_username) {
                auto& ctx = app.get_context<AuthMiddleware>(req);

                try {
                    auto user_info = auth_manager.getUserInfo(ctx.username, target_username);
                    if(!user_info) {
                        return json_error(404, "User not found");
                    }

                    return crow::response(200, user_info->dump());
                } catch(const std::exception& e) {
                    return json_error_500(ctx.username, req.url, std::string("Error: ") + e.what());
                }
            });

    // Get user type - always returns Admin in open-source mode
    CROW_ROUTE(app, "/api/v1/users/<string>/type")
            .CROW_MIDDLEWARES(app, AuthMiddleware)
            .methods("GET"_method)([&auth_manager, &app](const crow::request& req,
                                                         const std::string& target_username) {
                auto& ctx = app.get_context<AuthMiddleware>(req);

                try {
                    auto user_type = auth_manager.getUserType(target_username);
                    if(!user_type) {
                        return json_error(404, "User not found");
                    }

                    crow::json::wvalue response(
                            {{"username", settings::DEFAULT_USERNAME}, {"user_type", "Admin"}});

                    return crow::response(200, response.dump());
                } catch(const std::exception& e) {
                    return json_error_500(ctx.username, req.url, std::string("Error: ") + e.what());
                }
            });

    CROW_ROUTE(app, "/api/v1/stats").methods("GET"_method)([](const crow::request& req) {
        crow::json::wvalue response(
                {{"version", settings::VERSION}, {"uptime", 0}, {"total_requests", 0}});
        return crow::response(200, response.dump());
    });

    // Create index
    CROW_ROUTE(app, "/api/v1/index/create")
            .CROW_MIDDLEWARES(app, AuthMiddleware)
            .methods("POST"_method)([&index_manager, &app](const crow::request& req) {
                auto& ctx = app.get_context<AuthMiddleware>(req);

                LOG_DEBUG(" Request Body: " << req.body);

                auto body = crow::json::load(req.body);

                if(!body) {
                    return json_error(400, "Invalid JSON");
                }

                if(!body.has("index_name") || !body.has("dim") || !body.has("space_type")) {
                    return json_error(400, "Missing required parameters");
                }

                // Format index_id as username/index_name
                std::string index_id = ctx.username + "/" + std::string(body["index_name"].s());

                // Get checksum (optional, for queryable encryption)
                int32_t checksum = body.has("checksum") ? body["checksum"].i() : -1;
                LOG_DEBUG("Checksum: " << checksum);

                size_t dim = (size_t)body["dim"].i();
                // Validate dimension
                if(dim < settings::MIN_DIMENSION || dim > settings::MAX_DIMENSION) {
                    LOG_ERROR("Invalid dimension: " << dim);
                    return json_error(400,
                                      "Dimension must be between "
                                              + std::to_string(settings::MIN_DIMENSION) + " and "
                                              + std::to_string(settings::MAX_DIMENSION));
                }

                // Validate M
                size_t m = body.has("M") ? (size_t)body["M"].i() : settings::DEFAULT_M;
                if(m < settings::MIN_M || m > settings::MAX_M) {
                    LOG_ERROR("Invalid M: " << m);
                    return json_error(400,
                                      "M must be between " + std::to_string(settings::MIN_M)
                                              + " and " + std::to_string(settings::MAX_M));
                }

                // Validate ef_con
                size_t ef_con = body.has("ef_con") ? (size_t)body["ef_con"].i()
                                                   : settings::DEFAULT_EF_CONSTRUCT;
                if(ef_con < settings::MIN_EF_CONSTRUCT || ef_con > settings::MAX_EF_CONSTRUCT) {
                    LOG_ERROR("Invalid ef_con: " << ef_con);
                    return json_error(400,
                                      "ef_con must be between "
                                              + std::to_string(settings::MIN_EF_CONSTRUCT) + " and "
                                              + std::to_string(settings::MAX_EF_CONSTRUCT));
                }

                // Get quantization level (default to INT8)
                ndd::quant::QuantizationLevel quant_level =
                        body.has("precision") ? stringToQuantLevel(body["precision"].s())
                                              : ndd::quant::QuantizationLevel::INT8;

                // Validate quantization level
                if(quant_level == ndd::quant::QuantizationLevel::UNKNOWN) {
                    LOG_ERROR("Invalid precision: " << body["precision"].s());
                    std::vector<std::string> names = ndd::quant::getAvailableQuantizationNames();
                    std::string names_str;
                    for(size_t i = 0; i < names.size(); ++i) {
                        names_str += names[i];
                        if(i < names.size() - 1) {
                            names_str += ", ";
                        }
                    }
                    return json_error(400, "Invalid precision. Must be one of: " + names_str);
                }

                // Get custom size in millions (optional)
                size_t size_in_millions = 0;
                if(body.has("size_in_millions")) {
                    size_in_millions = static_cast<size_t>(body["size_in_millions"].i());
                    if(size_in_millions == 0 || size_in_millions > 10000) {  // Cap at 10B vectors
                        return json_error(400, "size_in_millions must be between 1 and 10000");
                    }
                    LOG_INFO("Creating index with custom size: " << size_in_millions
                                                                 << "M vectors");
                }

                size_t sparse_dim = body.has("sparse_dim") ? (size_t)body["sparse_dim"].i() : 0;

                IndexConfig config{dim,
                                   sparse_dim,
                                   settings::MAX_ELEMENTS,  // max elements
                                   body["space_type"].s(),
                                   m,
                                   ef_con,
                                   quant_level,
                                   checksum};

                try {
                    // Pass the full index_id to index_manager with Admin user type (no limits)
                    index_manager.createIndex(index_id, config, UserType::Admin, size_in_millions);
                    return crow::response(200, "Index created successfully");
                } catch(const std::runtime_error& e) {
                    return json_error(409, e.what());
                } catch(const std::exception& e) {
                    return json_error_500(ctx.username, req.url, std::string("Error: ") + e.what());
                }
            });

    // Create Backup
    CROW_ROUTE(app, "/api/v1/index/<string>/backup")
            .CROW_MIDDLEWARES(app, AuthMiddleware)
            .methods("POST"_method)([&index_manager, &app](const crow::request& req,
                                                           const std::string& index_name) {
                auto& ctx = app.get_context<AuthMiddleware>(req);
                auto body = crow::json::load(req.body);

                if(!body || !body.has("name")) {
                    return json_error(400, "Missing backup name");
                }

                std::string backup_name = body["name"].s();
                std::string index_id = ctx.username + "/" + index_name;

                try {
                    std::pair<bool, std::string> result =
                            index_manager.createBackup(index_id, backup_name);
                    if(!result.first) {
                        return json_error(400, result.second);
                    }
                    return crow::response(201, "Backup created successfully");
                } catch(const std::exception& e) {
                    return json_error(500, e.what());
                }
            });

    // List Backups
    CROW_ROUTE(app, "/api/v1/backups")
            .CROW_MIDDLEWARES(app, AuthMiddleware)
            .methods("GET"_method)([&index_manager, &app](const crow::request& req) {
                auto& ctx = app.get_context<AuthMiddleware>(req);
                try {
                    auto backups = index_manager.listBackups();
                    nlohmann::json result_json = backups;
                    crow::response res;
                    res.code = 200;
                    res.set_header("Content-Type", "application/json");
                    res.body = result_json.dump();
                    return res;
                } catch(const std::exception& e) {
                    return json_error(500, e.what());
                }
            });

    // Restore Backup
    CROW_ROUTE(app, "/api/v1/backups/<string>/restore")
            .CROW_MIDDLEWARES(app, AuthMiddleware)
            .methods("POST"_method)([&index_manager, &app](const crow::request& req,
                                                           const std::string& backup_name) {
                auto& ctx = app.get_context<AuthMiddleware>(req);
                auto body = crow::json::load(req.body);

                if(!body || !body.has("target_index_name")) {
                    return json_error(400, "Missing target_index_name");
                }

                std::string target_index_name = body["target_index_name"].s();

                try {
                    std::pair<bool, std::string> result =
                            index_manager.restoreBackup(backup_name, target_index_name);
                    if(!result.first) {
                        return json_error(400, result.second);
                    }
                    return crow::response(201, "Backup restored successfully");
                } catch(const std::exception& e) {
                    return json_error(500, e.what());
                }
            });

    // Delete Backup
    CROW_ROUTE(app, "/api/v1/backups/<string>")
            .CROW_MIDDLEWARES(app, AuthMiddleware)
            .methods("DELETE"_method)([&index_manager, &app](const crow::request& req,
                                                             const std::string& backup_name) {
                auto& ctx = app.get_context<AuthMiddleware>(req);
                try {
                    std::pair<bool, std::string> result = index_manager.deleteBackup(backup_name);
                    if(!result.first) {
                        return json_error(400, result.second);
                    }
                    return crow::response(204, "Backup deleted successfully");
                } catch(const std::exception& e) {
                    return json_error(500, e.what());
                }
            });

    // Download Backup
    CROW_ROUTE(app, "/api/v1/backups/<string>/download")
            .CROW_MIDDLEWARES(app, AuthMiddleware)
            .methods("GET"_method)([&index_manager, &app](const crow::request& req,
                                                          const std::string& backup_name) {
                auto& ctx = app.get_context<AuthMiddleware>(req);
                try {
                    std::string backup_tar =
                            settings::DATA_DIR + "/backups/" + backup_name + ".tar.gz";
                    if(!std::filesystem::exists(backup_tar)) {
                        return json_error(404, "Backup not found");
                    }
                    std::string file_content = read_file(backup_tar);
                    if(file_content.empty()) {
                        return json_error(500, "Failed to read backup file");
                    }
                    crow::response response;
                    response.set_header("Content-Type", "application/gzip");
                    response.set_header("Content-Disposition",
                                        "attachment; filename=\"" + backup_name + ".tar.gz\"");
                    response.set_header("Content-Length", std::to_string(file_content.size()));
                    response.set_header("Cache-Control", "no-cache");
                    response.body = std::move(file_content);
                    return response;
                } catch(const std::exception& e) {
                    return json_error(500, e.what());
                }
            });

    // upload Backup
    CROW_ROUTE(app, "/api/v1/backups/upload")
            .CROW_MIDDLEWARES(app, AuthMiddleware)
            .methods("POST"_method)([&index_manager, &app](const crow::request& req) {
                auto& ctx = app.get_context<AuthMiddleware>(req);
                try {
                    // Parse multipart message
                    crow::multipart::message msg(req);

                    // Find the file part
                    std::string backup_name;
                    std::string file_content;

                    for(const auto& part : msg.parts) {
                        auto content_disposition = part.get_header_object("Content-Disposition");
                        std::string name = content_disposition.params.count("name")
                                                   ? content_disposition.params.at("name")
                                                   : "";

                        if(name == "backup") {
                            // Get filename from Content-Disposition
                            if(content_disposition.params.count("filename")) {
                                backup_name = content_disposition.params.at("filename");
                                // check if backup name ends with .tar.gz
                                if(backup_name.ends_with(".tar.gz")) {
                                    backup_name = backup_name.substr(0, backup_name.size() - 7);
                                } else {
                                    return json_error(400, "Invalid backup file extension");
                                }
                            }
                            file_content = part.body;
                            break;
                        }
                    }

                    if(backup_name.empty()) {
                        return json_error(400, "Missing backup name or filename");
                    }

                    if(file_content.empty()) {
                        return json_error(400, "Missing backup file content");
                    }

                    // Validate backup name
                    std::pair<bool, std::string> result =
                            index_manager.validateBackupName(backup_name);
                    if(!result.first) {
                        return json_error(400, result.second);
                    }

                    // Check if backup already exists
                    std::string backup_path =
                            settings::DATA_DIR + "/backups/" + backup_name + ".tar.gz";
                    if(std::filesystem::exists(backup_path)) {
                        return json_error(409,
                                          "Backup with name '" + backup_name + "' already exists");
                    }

                    // Write the file
                    std::ofstream out(backup_path, std::ios::binary);
                    if(!out.is_open()) {
                        return json_error(500, "Failed to create backup file");
                    }
                    out.write(file_content.data(), file_content.size());
                    out.close();

                    if(!out.good()) {
                        // Clean up partial file on error
                        std::filesystem::remove(backup_path);
                        return json_error(500, "Failed to write backup file");
                    }

                    return crow::response(201, "Backup uploaded successfully");
                } catch(const std::exception& e) {
                    return json_error(500, e.what());
                }
            });

    // List indexes for current user
    CROW_ROUTE(app, "/api/v1/index/list")
            .CROW_MIDDLEWARES(app, AuthMiddleware)
            .methods("GET"_method)([&index_manager, &app](const crow::request& req) {
                auto& ctx = app.get_context<AuthMiddleware>(req);

                // Use the method to get user indexes with metadata
                auto indexes_with_metadata = index_manager.listUserIndexes(ctx.username);

                // Build a detailed response with array of index objects
                std::vector<crow::json::wvalue> index_list;
                for(const auto& [index_name, metadata] : indexes_with_metadata) {
                    crow::json::wvalue index_info(
                            {{"name", index_name},
                             {"dimension", static_cast<int64_t>(metadata.dimension)},
                             {"sparse_dim", static_cast<int64_t>(metadata.sparse_dim)},
                             {"space_type", metadata.space_type_str},
                             {"precision", quantLevelToString(metadata.quant_level)},
                             {"total_elements", static_cast<int64_t>(metadata.total_elements)},
                             {"checksum", metadata.checksum},
                             {"M", static_cast<int64_t>(metadata.M)},
                             {"created_at",
                              static_cast<int64_t>(
                                      std::chrono::system_clock::to_time_t(metadata.created_at))}});
                    index_list.push_back(std::move(index_info));
                }

                crow::json::wvalue response;
                response["indexes"] = std::move(index_list);
                return crow::response(200, response.dump());
            });

    // Delete index
    CROW_ROUTE(app, "/api/v1/index/<string>/delete")
            .CROW_MIDDLEWARES(app, AuthMiddleware)
            .methods("DELETE"_method)(
                    [&index_manager, &app](const crow::request& req, std::string index_name) {
                        auto& ctx = app.get_context<AuthMiddleware>(req);

                        // Format full index_id
                        std::string index_id = ctx.username + "/" + index_name;

                        if(index_manager.deleteIndex(index_id)) {
                            return crow::response(200, "Index deleted successfully");
                        } else {
                            return json_error(404, "Index not found");
                        }
                    });

    // Search
    CROW_ROUTE(app, "/api/v1/index/<string>/search")
            .CROW_MIDDLEWARES(app, AuthMiddleware)
            .methods("POST"_method)([&index_manager, &app](const crow::request& req,
                                                           std::string index_name) {
                auto& ctx = app.get_context<AuthMiddleware>(req);
                // Format full index_id
                std::string index_id = ctx.username + "/" + index_name;

                auto body = crow::json::load(req.body);
                if(!body || !body.has("k")) {
                    return json_error(400, "Missing required parameters: k");
                }

                if(!body.has("vector") && !body.has("sparse_indices")) {
                    return json_error(400, "Missing query vector (dense or sparse)");
                }

                std::vector<float> query;
                if(body.has("vector")) {
                    for(const auto& elem : body["vector"]) {
                        query.push_back((float)elem.d());
                    }
                }

                std::vector<uint32_t> sparse_indices;
                std::vector<float> sparse_values;

                if(body.has("sparse_indices")) {
                    for(const auto& elem : body["sparse_indices"]) {
                        sparse_indices.push_back((uint32_t)elem.i());
                    }
                }

                if(body.has("sparse_values")) {
                    for(const auto& elem : body["sparse_values"]) {
                        sparse_values.push_back((float)elem.d());
                    }
                }

                if(sparse_indices.size() != sparse_values.size()) {
                    return json_error(400,
                                      "Mismatch between sparse_indices and sparse_values size");
                }

                size_t k = (size_t)body["k"].i();
                if(k < settings::MIN_K || k > settings::MAX_K) {
                    LOG_ERROR("Invalid k: " << k);
                    return json_error(400,
                                      "k must be between " + std::to_string(settings::MIN_K)
                                              + " and " + std::to_string(settings::MAX_K));
                }
                size_t ef = body.has("ef") ? (size_t)body["ef"].i() : 0;
                bool include_vectors =
                        body.has("include_vectors") ? body["include_vectors"].b() : false;
                nlohmann::json filter_array = nlohmann::json::array();  // default: empty filter

                if(body.has("filter")) {
                    try {
                        auto raw_filter = nlohmann::json::parse(body["filter"].s());
                        // Expect new array-based filter format
                        if(!raw_filter.is_array()) {
                            return json_error(400,
                                              "Filter must be an array. Please use format: "
                                              "[{\"field\":{\"$op\":value}}]");
                        }
                        filter_array = raw_filter;
                    } catch(const std::exception& e) {
                        return json_error(400, std::string("Invalid filter JSON: ") + e.what());
                    }
                }
                LOG_DEBUG("Filter: " << filter_array.dump());
                try {
                    auto search_response = index_manager.searchKNN(index_id,
                                                                   query,
                                                                   sparse_indices,
                                                                   sparse_values,
                                                                   k,
                                                                   filter_array,
                                                                   include_vectors,
                                                                   ef);
                    if(!search_response) {
                        return json_error(404, "Index not found or search failed");
                    }

                    // Serialize the ResultSet using MessagePack
                    msgpack::sbuffer sbuf;
                    msgpack::pack(sbuf, search_response.value());
                    crow::response resp(200, std::string(sbuf.data(), sbuf.size()));
                    resp.add_header("Content-Type", "application/msgpack");
                    return resp;
                } catch(const std::runtime_error& e) {
                    return json_error(400, e.what());
                } catch(const std::exception& e) {
                    LOG_DEBUG("Search failed: " << e.what());
                    return json_error_500(
                            ctx.username, req.url, std::string("Search failed: ") + e.what());
                }
            });

    //  Insert a list of vectors
    CROW_ROUTE(app, "/api/v1/index/<string>/vector/insert")
            .CROW_MIDDLEWARES(app, AuthMiddleware)
            .methods("POST"_method)([&index_manager, &app](const crow::request& req,
                                                           std::string index_name) {
                auto& ctx = app.get_context<AuthMiddleware>(req);
                std::string index_id = ctx.username + "/" + index_name;

                // Verify content type is application/msgpack or application/json
                auto content_type = req.get_header_value("Content-Type");

                if(content_type == "application/json") {
                    auto body = crow::json::load(req.body);
                    if(!body) {
                        return json_error(400, "Invalid JSON");
                    }

                    std::vector<ndd::HybridVectorObject> vectors;

                    // Helper to parse single object
                    auto parse_obj = [](const crow::json::rvalue& item) -> ndd::HybridVectorObject {
                        ndd::HybridVectorObject vec;
                        if(item.has("id")) {
                            if(item["id"].t() == crow::json::type::Number) {
                                vec.id = std::to_string(item["id"].i());
                            } else {
                                vec.id = item["id"].s();
                            }
                        }

                        if(item.has("vector")) {
                            for(const auto& v : item["vector"]) {
                                vec.vector.push_back(static_cast<float>(v.d()));
                            }
                        }

                        if(item.has("sparse_indices")) {
                            for(const auto& v : item["sparse_indices"]) {
                                vec.sparse_ids.push_back(static_cast<uint32_t>(v.i()));
                            }
                        }

                        if(item.has("sparse_values")) {
                            for(const auto& v : item["sparse_values"]) {
                                vec.sparse_values.push_back(static_cast<float>(v.d()));
                            }
                        }
                        return vec;
                    };

                    if(body.t() == crow::json::type::List) {
                        for(const auto& item : body) {
                            vectors.push_back(parse_obj(item));
                        }
                    } else {
                        vectors.push_back(parse_obj(body));
                    }

                    try {
                        bool success = index_manager.addVectors(index_id, vectors);
                        return crow::response(success ? 200 : 400);
                    } catch(const std::runtime_error& e) {
                        return json_error(400, e.what());
                    } catch(const std::exception& e) {
                        return json_error_500(ctx.username, req.url, e.what());
                    }
                } else if(content_type == "application/msgpack") {
                    // Deserialize MsgPack batch
                    try {
                        auto oh = msgpack::unpack(req.body.data(), req.body.size());
                        auto obj = oh.get();

                        try {
                            // Try HybridVectorObject first
                            auto vectors = obj.as<std::vector<ndd::HybridVectorObject>>();
                            LOG_DEBUG("Batch size (Hybrid): " << vectors.size());
                            bool success = index_manager.addVectors(index_id, vectors);
                            return crow::response(success ? 200 : 400);
                        } catch(...) {
                            // Fallback to VectorObject
                            auto vectors = obj.as<std::vector<ndd::VectorObject>>();
                            LOG_DEBUG("Batch size (Dense): " << vectors.size());
                            bool success = index_manager.addVectors(index_id, vectors);
                            return crow::response(success ? 200 : 400);
                        }
                    } catch(const std::runtime_error& e) {
                        return json_error(400, e.what());
                    } catch(const std::exception& e) {
                        LOG_DEBUG("Batch insertion failed: " << e.what());
                        return json_error_500(ctx.username, req.url, e.what());
                    }
                } else {
                    return crow::response(
                            400, "Content-Type must be application/msgpack or application/json");
                }
            });

    // Get a single vector
    CROW_ROUTE(app, "/api/v1/index/<string>/vector/get")
            .CROW_MIDDLEWARES(app, AuthMiddleware)
            .methods("POST"_method)(
                    [&index_manager, &app](const crow::request& req, std::string index_name) {
                        auto& ctx = app.get_context<AuthMiddleware>(req);
                        std::string index_id = ctx.username + "/" + index_name;

                        // Read vector ID from JSON input (still using JSON for ID here)
                        auto body = crow::json::load(req.body);
                        if(!body || !body.has("id")) {
                            return json_error(400, "Missing required parameter 'id'");
                        }
                        std::string vector_id = body["id"].s();
                        try {
                            auto vector = index_manager.getVector(index_id, vector_id);
                            if(!vector) {
                                return json_error(404, "Vector with the given ID does not exist");
                            }
                            // Serialize vector as MsgPack
                            msgpack::sbuffer sbuf;
                            msgpack::pack(sbuf, vector.value());
                            // Return as MessagePack
                            crow::response resp(200, std::string(sbuf.data(), sbuf.size()));
                            resp.add_header("Content-Type", "application/msgpack");
                            return resp;
                        } catch(const std::exception& e) {
                            LOG_DEBUG("Failed to get vector: " << e.what());
                            return json_error_500(ctx.username,
                                                  req.url,
                                                  std::string("Failed to get vector: ") + e.what());
                        }
                    });

    // Delete a vector
    CROW_ROUTE(app, "/api/v1/index/<string>/vector/<string>/delete")
            .CROW_MIDDLEWARES(app, AuthMiddleware)
            .methods("DELETE"_method)([&index_manager, &app](const crow::request& req,
                                                             std::string index_name,
                                                             std::string vector_id) {
                auto& ctx = app.get_context<AuthMiddleware>(req);
                std::string index_id = ctx.username + "/" + index_name;

                LOG_DEBUG("Deleting vector " << vector_id << " from index " << index_id);

                try {
                    if(index_manager.deleteVector(index_id, vector_id)) {
                        return crow::response(200, "Vector deleted successfully");
                    } else {
                        return json_error(404, "Vector with the given ID does not exist");
                    }
                } catch(const std::runtime_error& e) {
                    return json_error(400, e.what());
                } catch(const std::exception& e) {
                    LOG_DEBUG("Failed to delete vector: " << e.what());
                    return json_error_500(ctx.username,
                                          req.url,
                                          std::string("Failed to delete vector: ") + e.what());
                }
            });

    // Delete vectors by filter
    CROW_ROUTE(app, "/api/v1/index/<string>/vectors/delete")
            .CROW_MIDDLEWARES(app, AuthMiddleware)
            .methods("DELETE"_method)([&index_manager, &app](const crow::request& req,
                                                             std::string index_name) {
                auto& ctx = app.get_context<AuthMiddleware>(req);
                std::string index_id = ctx.username + "/" + index_name;

                nlohmann::json body;
                try {
                    body = nlohmann::json::parse(req.body);
                } catch(const std::exception& e) {
                    return json_error(400, "Invalid JSON body");
                }
                if(!body.contains("filter")) {
                    return json_error(400, "Invalid request body - missing filter");
                }
                try {
                    nlohmann::json filter_array = body["filter"];
                    // Expect new array-based filter format
                    if(!filter_array.is_array()) {
                        return json_error(400,
                                          "Filter must be an array. Please use format: "
                                          "[{\"field\":{\"$op\":value}}]");
                    }
                    size_t deleted_count =
                            index_manager.deleteVectorsByFilter(index_id, filter_array);

                    return crow::response(200, std::to_string(deleted_count) + " vectors deleted");
                } catch(const std::runtime_error& e) {
                    return json_error(400, e.what());
                } catch(const std::exception& e) {
                    return json_error_500(ctx.username,
                                          req.url,
                                          std::string("Failed to delete vectors: ") + e.what());
                }
            });

    // Update filters for vectors
    CROW_ROUTE(app, "/api/v1/index/<string>/filters/update")
            .CROW_MIDDLEWARES(app, AuthMiddleware)
            .methods("POST"_method)([&index_manager, &app](const crow::request& req,
                                                           std::string index_name) {
                auto& ctx = app.get_context<AuthMiddleware>(req);
                std::string index_id = ctx.username + "/" + index_name;

                nlohmann::json body;
                try {
                    body = nlohmann::json::parse(req.body);
                } catch(const std::exception& e) {
                    return json_error(400, "Invalid JSON body");
                }

                if(!body.contains("updates") || !body["updates"].is_array()) {
                    return json_error(400,
                                      "Missing or invalid 'updates' field. Must be a list of {id, "
                                      "filter} objects.");
                }

                std::vector<std::pair<std::string, std::string>> updates;
                try {
                    for(const auto& item : body["updates"]) {
                        if(!item.contains("id") || !item.contains("filter")) {
                            continue;  // Skip invalid items
                        }
                        std::string id = item["id"].get<std::string>();
                        // Convert filter object to string
                        std::string filter = item["filter"].dump();
                        updates.emplace_back(id, filter);
                    }

                    size_t count = index_manager.updateFilters(index_id, updates);
                    return crow::response(200, std::to_string(count) + " filters updated");

                } catch(const std::exception& e) {
                    return json_error_500(ctx.username,
                                          req.url,
                                          std::string("Failed to update filters: ") + e.what());
                }
            });

    CROW_ROUTE(app, "/api/v1/index/<string>/info")
            .CROW_MIDDLEWARES(app, AuthMiddleware)
            .methods("GET"_method)([&index_manager, &app](const crow::request& req,
                                                          std::string index_name) {
                auto& ctx = app.get_context<AuthMiddleware>(req);
                std::string index_id = ctx.username + "/" + index_name;
                try {
                    auto info = index_manager.getIndexInfo(index_id);
                    if(!info) {
                        return json_error(404, "Index does not exist");
                    }
                    crow::json::wvalue response(
                            {{"total_elements", static_cast<int64_t>(info->total_elements)},
                             {"dimension", static_cast<int64_t>(info->dimension)},
                             {"sparse_dim", static_cast<int64_t>(info->sparse_dim)},
                             {"space_type", info->space_type_str},
                             {"precision", quantLevelToString(info->quant_level)},
                             {"checksum", info->checksum},
                             {"M", static_cast<int64_t>(info->M)},
                             {"ef_con", static_cast<int64_t>(info->ef_con)},
                             {"lib_token", settings::DEFAULT_LIB_TOKEN}});
                    return crow::response(200, response.dump());
                } catch(const std::runtime_error& e) {
                    return json_error(404, std::string("Error: ") + e.what());
                } catch(const std::exception& e) {
                    return json_error_500(ctx.username, req.url, std::string("Error: ") + e.what());
                }
            });

    // ============================================================
    // REACT SPA SERVING WITH CLIENT-SIDE ROUTING SUPPORT
    // ============================================================

    // Serve static assets from React build (JS, CSS, images, fonts, etc.)
    // This catches all requests to /assets/* which is Vite's default output structure
    CROW_ROUTE(app, "/assets/<path>")
    ([BUILD_DIR](const crow::request&, std::string path) {
        std::string file_path = BUILD_DIR + "/assets/" + path;

        if(!file_exists(file_path)) {
            return crow::response(404, "Asset not found");
        }

        std::string content = read_file(file_path);
        if(content.empty()) {
            return crow::response(404, "Asset not found");
        }

        auto response = crow::response(content);
        response.set_header("Content-Type", get_mime_type(path));
        // Cache static assets for 1 year (they are content-hashed)
        response.set_header("Cache-Control", "public, max-age=31536000, immutable");
        return response;
    });

    // Serve other static files (favicon, manifest, etc.)
    CROW_ROUTE(app, "/<path>")
    ([BUILD_DIR, INDEX_PATH](const crow::request&, std::string path) {
        // Try to serve the specific file first
        std::string file_path = BUILD_DIR + "/" + path;

        // If file exists and is not a directory, serve it
        if(file_exists(file_path)) {
            std::string content = read_file(file_path);
            if(!content.empty()) {
                auto response = crow::response(content);
                response.set_header("Content-Type", get_mime_type(path));

                // Cache static files
                if(path.ends_with(".ico") || path.ends_with(".png") || path.ends_with(".svg")
                   || path.ends_with(".json")) {
                    response.set_header("Cache-Control", "public, max-age=3600");
                }

                return response;
            }
        }

        // SPA Fallback: For any route that doesn't match a file,
        // serve index.html to let React Router handle it
        // This enables client-side routing for paths like:
        // /, /dashboard, /settings, /items/123, etc.
        std::string index_content = read_file(INDEX_PATH);
        if(index_content.empty()) {
            return crow::response(404, "App not found");
        }

        auto response = crow::response(index_content);
        response.set_header("Content-Type", "text/html");
        response.set_header("Cache-Control", "no-cache");
        return response;
    });

    // Root route - serve index.html
    CROW_ROUTE(app, "/")
    ([INDEX_PATH] {
        std::string content = read_file(INDEX_PATH);
        if(content.empty()) {
            return crow::response(404, "App not found");
        }

        auto response = crow::response(content);
        response.set_header("Content-Type", "text/html");
        response.set_header("Cache-Control", "no-cache");
        return response;
    });

    unsigned int num_cores = std::thread::hardware_concurrency();
    LOG_INFO("Number of processor cores: " << num_cores);
    if(settings::NUM_SERVER_THREADS == 0) {
        // Run on max possible threads
        LOG_INFO("Using all available threads");
        app.port(settings::SERVER_PORT).multithreaded().run();
    } else {
        // Limit on the number of threads
        LOG_INFO("Using " << settings::NUM_SERVER_THREADS << " threads");
        app.port(settings::SERVER_PORT).concurrency(settings::NUM_SERVER_THREADS).run();
    }

    return 0;
}
