#pragma once
#include <curl/curl.h>
#include <regex>

#include "hnsw/hnswlib.h"
#include "settings.hpp"
#include "id_mapper.hpp"
#include "vector_storage.hpp"
#include "../sparse/sparse_storage.hpp"
#include "rand_utils.hpp"
#include "index_meta.hpp"
#include "msgpack_ndd.hpp"
#include "quant_vector.hpp"
#include "wal.hpp"
#include "../quant/dispatch.hpp"
#include "../utils/archive_utils.hpp"
#include <memory>
#include <deque>
#include <unordered_map>
#include <list>
#include <algorithm>
#include <mutex>
#include <chrono>
#include <filesystem>
#include <thread>
#include <atomic>
#include <optional>
#include <random>
#include <type_traits>
#include <future>

#define MAX_BACKUP_NAME_LENGTH 200

struct IndexConfig {
    size_t dim;
    size_t sparse_dim = 0;  // 0 means dense-only
    size_t max_elements = settings::MAX_ELEMENTS;
    std::string space_type_str;
    size_t M = settings::DEFAULT_M;
    size_t ef_construction = settings::DEFAULT_EF_CONSTRUCT;
    ndd::quant::QuantizationLevel quant_level =
            ndd::quant::QuantizationLevel::INT8;  // Default to INT8 quantization
    const int32_t checksum;
};

struct IndexInfo {
    size_t total_elements;
    size_t dimension;
    size_t sparse_dim;
    std::string space_type_str;
    ndd::quant::QuantizationLevel
            quant_level;  // Quantization level (8, 15, 16, 32) - replaces use_fp16
    int32_t checksum;
    size_t M;
    size_t ef_con;
};

struct CacheEntry {
    std::string index_id;
    size_t sparse_dim = 0;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> alg;
    std::shared_ptr<IDMapper> id_mapper;
    std::shared_ptr<VectorStorage> vector_storage;
    std::unique_ptr<ndd::SparseVectorStorage> sparse_storage;
    std::chrono::system_clock::time_point last_access;
    std::chrono::system_clock::time_point last_saved_at;
    std::chrono::system_clock::time_point updated_at;
    // Flag to indicate if the index has been updated
    bool updated{false};
    // Number of searches performed on this index. For a search with k=10 it will be 10
    size_t searchCount{0};
    // Per-index operation mutex for coordinating addVectors, saveIndex, deleteVectors
    std::mutex operation_mutex;

    // Default constructor required for map
    CacheEntry() :
        last_access(std::chrono::system_clock::now()) {}

    CacheEntry(std::string index_id_,
               size_t sparse_dim_,
               std::unique_ptr<hnswlib::HierarchicalNSW<float>> alg_,
               std::shared_ptr<IDMapper> mapper_,
               std::shared_ptr<VectorStorage> storage_,
               std::unique_ptr<ndd::SparseVectorStorage> sparse_storage_,
               std::chrono::system_clock::time_point access_time_) {
        LOG_INFO("Creating CacheEntry for index: " << index_id_);

        // Validate all components
        if(!alg_) {
            LOG_ERROR("Algorithm is null");
            throw std::runtime_error("Algorithm is null");
        }
        if(!mapper_) {
            LOG_ERROR("ID Mapper is null");
            throw std::runtime_error("ID Mapper is null");
        }
        if(!storage_) {
            LOG_ERROR("Vector Storage is null");
            throw std::runtime_error("Vector Storage is null");
        }

        LOG_INFO("Assigning index_id: " << index_id_);
        index_id = std::move(index_id_);
        sparse_dim = sparse_dim_;

        id_mapper = std::move(mapper_);

        vector_storage = std::move(storage_);

        sparse_storage = std::move(sparse_storage_);

        last_access = access_time_;

        LOG_INFO("Moving algorithm instance");
        alg = std::move(alg_);

        last_saved_at = std::chrono::system_clock::now();

        LOG_INFO("CacheEntry construction completed");
    }

    void markUpdated() {
        updated = true;
        updated_at = std::chrono::system_clock::now();
    }
    void resetSearchCount() { searchCount = 0; }
    // Delete copy constructor and assignment
    CacheEntry(const CacheEntry&) = delete;
    CacheEntry& operator=(const CacheEntry&) = delete;

    // Disable move operations since mutex is not movable
    CacheEntry(CacheEntry&&) = delete;
    CacheEntry& operator=(CacheEntry&&) = delete;
};

struct PersistenceConfig {
    size_t save_every_n_updates{settings::SAVE_EVERY_N_UPDATES};
    std::chrono::minutes save_interval{settings::SAVE_EVERY_N_MINUTES};
    bool save_on_shutdown{true};
};

class IndexManager {
private:
    std::deque<std::string> indices_list_;
    std::unordered_map<std::string, CacheEntry> indices_;
    std::shared_mutex indices_mutex_;
    std::string data_dir_;
    // This is for locking the LRU
    std::shared_mutex active_indices_mutex_;
    PersistenceConfig persistence_config_;
    std::atomic<bool> shutdown_requested_{false};
    std::condition_variable persistence_cv_;
    std::unique_ptr<MetadataManager> metadata_manager_;
    // Autosave methods
    std::thread autosave_thread_;
    std::atomic<bool> running_{true};
    // Write-ahead log for each index
    std::unordered_map<std::string, std::unique_ptr<WriteAheadLog>> wal_logs_;

    // New methods to handle WAL
    WriteAheadLog* getOrCreateWAL(const std::string& index_id) {
        auto it = wal_logs_.find(index_id);
        if(it != wal_logs_.end()) {
            return it->second.get();
        }

        std::string wal_dir = data_dir_ + "/" + index_id;
        auto wal = std::make_unique<WriteAheadLog>(wal_dir);
        auto wal_ptr = wal.get();
        wal_logs_[index_id] = std::move(wal);
        return wal_ptr;
    }

    void clearWAL(const std::string& index_id) {
        auto it = wal_logs_.find(index_id);
        if(it != wal_logs_.end()) {
            it->second->clear();
        }
    }

    // Helper method for WAL recovery
    void recoverFromWAL(const std::string& index_id) {
        auto& entry = getIndexEntry(index_id);
        WriteAheadLog* wal = getOrCreateWAL(index_id);

        // Check if WAL has entries needing recovery
        if(wal->hasEntries()) {
            LOG_INFO("WAL recovery needed for index " << index_id);

            auto wal_entries = wal->readEntries();
            LOG_INFO("Read " << wal_entries.size() << " entries from WAL");

            // Process all entries in the exact order they were recorded
            std::vector<idInt> failed_vector_add_ids;

            for(const auto& wal_entry : wal_entries) {
                try {
                    if(wal_entry.op_type == WALOperationType::VECTOR_ADD) {
                        // Check if vector exists in storage before recovering
                        auto vector_bytes = entry.vector_storage->get_vector(wal_entry.numeric_id);
                        if(!vector_bytes.empty()) {
                            entry.alg->addPoint<true>(vector_bytes.data(), wal_entry.numeric_id);
                        } else {
                            // Vector doesn't exist - this VECTOR_ADD failed
                            failed_vector_add_ids.push_back(wal_entry.numeric_id);
                            LOG_DEBUG("VECTOR_ADD failed for ID " << wal_entry.numeric_id
                                                                  << " - adding to deleted_ids");
                        }
                    } else if(wal_entry.op_type == WALOperationType::VECTOR_UPDATE) {
                        // Recover vector update
                        auto vector_bytes = entry.vector_storage->get_vector(wal_entry.numeric_id);
                        if(!vector_bytes.empty()) {
                            entry.alg->addPoint<false>(vector_bytes.data(), wal_entry.numeric_id);
                        }
                    } else if(wal_entry.op_type == WALOperationType::VECTOR_DELETE) {
                        // For deletions, just mark the vector as deleted
                        entry.alg->markDelete(wal_entry.numeric_id);
                    }
                } catch(const std::exception& e) {
                    if(wal_entry.op_type == WALOperationType::VECTOR_ADD) {
                        // If VECTOR_ADD recovery failed, add ID to failed list
                        failed_vector_add_ids.push_back(wal_entry.numeric_id);
                        LOG_DEBUG("VECTOR_ADD recovery failed for ID " << wal_entry.numeric_id
                                                                       << ": " << e.what());
                    } else {
                        LOG_DEBUG("Failed to recover operation for vector " << wal_entry.numeric_id
                                                                            << ": " << e.what());
                    }
                }
            }

            // Add failed VECTOR_ADD IDs back to deleted_ids for reuse
            if(!failed_vector_add_ids.empty()) {
                entry.id_mapper->reclaim_failed_ids(failed_vector_add_ids);
                LOG_INFO("Reclaimed " << failed_vector_add_ids.size()
                                      << " failed VECTOR_ADD IDs for reuse");
            }

            // Mark as updated to trigger a save
            entry.markUpdated();
            // Explicitly save the index after recovery
            LOG_DEBUG("Saving index after WAL recovery: " << index_id);
            // Save index will also clear the WAL and save bloom filter
            // FIX: Call saveIndexInternal instead of saveIndex to avoid circular lock
            saveIndexInternal(entry);
        }
    }

    // The thread will call this method
    void autosaveLoop() {
        LOG_INFO("Autosave thread started");
        while(running_) {
            // Sleep for 5 minutes
            std::this_thread::sleep_for(std::chrono::minutes(5));

            // Check if we're still running
            if(!running_) {
                break;
            }
            LOG_INFO("Autosave check running");
            checkAndSaveIndices();
        }
        LOG_INFO("Autosave thread stopped");
    }

    // Check and save indices based on update time
    void checkAndSaveIndices() {
        std::vector<std::string> indices_to_save;
        auto now = std::chrono::system_clock::now();

        // First identify indices to save (without holding main mutex for too long)
        {
            for(auto& [index_id, entry] : indices_) {
                if(entry.updated) {
                    auto time_since_update = now - entry.updated_at;
                    // Save if more than 60 minutes since update
                    if(time_since_update > std::chrono::minutes(60)) {
                        indices_to_save.push_back(index_id);
                    }
                }
            }
        }

        // Now save each index individually
        for(const auto& index_id : indices_to_save) {
            auto it = indices_.find(index_id);
            if(it != indices_.end() && it->second.updated) {
                LOG_DEBUG("Auto-saving index (60-minute threshold): " << index_id);
                saveIndex(index_id);
            }
        }
    }

    // Get index entry with proper lock management - does NOT hold locks after return
    CacheEntry& getIndexEntry(const std::string& index_id) {
        // First try to find the index without write lock
        {
            //std::shared_lock<std::shared_mutex> read_lock(indices_mutex_);
            auto it = indices_.find(index_id);
            if(it != indices_.end()) {
                return it->second;
            }
        }

        // Index not found, need to load it with write lock
        {
            std::unique_lock<std::shared_mutex> write_lock(indices_mutex_);
            auto it = indices_.find(index_id);
            if(it == indices_.end()) {
                loadIndex(index_id);  // modifies indices_
                evictIfNeeded();      // Clean eviction only
            }
            it = indices_.find(index_id);
            if(it == indices_.end()) {
                throw std::runtime_error("[ERROR] Failed to load index");
            }
            // Return reference - write_lock will be released when this scope ends
            return it->second;
        }
    }

    void saveIndex(const std::string& index_id) {
        LOG_DEBUG("saveIndex called for index=" + index_id);

        // Get the index entry (thread-safe)
        auto& entry = getIndexEntry(index_id);

        // Use per-index operation mutex to prevent concurrent operations
        std::lock_guard<std::mutex> operation_lock(entry.operation_mutex);

        // Call internal implementation
        saveIndexInternal(entry);
    }

private:
    // Internal saveIndex implementation that doesn't call getIndexEntry
    // Used by functions that already have the entry and mutex
    void saveIndexInternal(CacheEntry& entry) {
        // Double check if the index is still updated
        if(!entry.updated) {
            return;
        }
        LOG_DEBUG("Saving index " << entry.index_id);
        // Auto-resize check
        size_t remainingCapacity = entry.alg->getRemainingCapacity();
        LOG_DEBUG("Remaining capacity for index " << entry.index_id << ": " << remainingCapacity);
        size_t maxElements = entry.alg->getMaxElements();
        LOG_DEBUG("Max elements for index " << entry.index_id << ": " << maxElements);

        // If remaining capacity is less than 50k, resize by adding 100k
        if(remainingCapacity < settings::MAX_ELEMENTS_INCREMENT_TRIGGER) {
            size_t newMaxElements = maxElements + settings::MAX_ELEMENTS_INCREMENT;
            LOG_DEBUG("Auto-resizing index " << entry.index_id << " from " << maxElements << " to "
                                             << newMaxElements << " elements");

            try {
                entry.alg->resizeIndex(newMaxElements);
            } catch(const std::exception& e) {
                LOG_DEBUG("Failed to auto-resize index: " << e.what());
                // Continue with saving even if resize fails
            }
        }

        std::string index_path = data_dir_ + "/" + entry.index_id + "/main.idx";
        std::string temp_path = index_path + ".tmp";

        entry.alg->saveIndex(temp_path);
        std::filesystem::rename(temp_path, index_path);

        // Clear the WAL
        clearWAL(entry.index_id);

        // Update element count in metadata
        if(!metadata_manager_->updateElementCount(entry.index_id, entry.alg->getElementsCount())) {
            std::cerr << "Warning: Failed to update element count in metadata for "
                      << entry.index_id << std::endl;
        }
        entry.updated = false;
    }

public:
    // Evict the last index if the total size exceeds the limit
    void evictIfNeeded() {
        // Go through indices and get the total size. If it exceeds the limit, evict the last one
        size_t total_size = 0;
        for(auto& [index_id, entry] : indices_) {
            if(entry.alg) {
                total_size += entry.alg->getApproxSizeGB();
            }
        }
        if(total_size > settings::MAX_MEMORY_GB) {
            // Make sure that there is at least one index in memory and we use only 80% of the total
            // size
            while((total_size > 0.80 * settings::MAX_MEMORY_GB) && (indices_list_.size() > 1)) {
                // Pop from the back of the active indices list
                std::string to_evict = indices_list_.back();
                indices_list_.pop_back();
                auto it = indices_.find(to_evict);
                if(it != indices_.end()) {
                    total_size -= it->second.alg->getApproxSizeGB();

                    // Only evict if the index is not dirty (hasn't been updated)
                    if(it->second.updated) {
                        LOG_WARN("Cannot evict dirty index " << to_evict
                                                             << " - needs saving first");
                        // Put it back at the front to try other indices
                        indices_list_.push_front(to_evict);
                        continue;
                    }

                    LOG_INFO("Evicting clean index " << to_evict);
                    indices_.erase(it);
                }
            }
        }
    }

    // Function to be called by cron job instead of running as a thread
    bool autoSave() {
        std::vector<std::string> indices_to_save;

        for(const auto& [index_id, entry] : indices_) {
            if(entry.updated) {  // Check the flag in CacheEntry
                indices_to_save.push_back(index_id);
            }
        }

        // Now save each index that needs saving
        bool saved_any = false;
        for(const auto& index_id : indices_to_save) {
            LOG_DEBUG("Auto-saving index: " << index_id);

            // Check if index still exists and needs saving (thread-safe)
            bool should_save = false;
            {
                std::shared_lock<std::shared_mutex> read_lock(indices_mutex_);
                auto it = indices_.find(index_id);
                should_save = (it != indices_.end() && it->second.updated);
            }

            if(should_save) {
                LOG_DEBUG("Autosaving index: " << index_id);
                saveIndex(index_id);

                // Reset the flag after saving (thread-safe)
                {
                    std::shared_lock<std::shared_mutex> read_lock(indices_mutex_);
                    auto it = indices_.find(index_id);
                    if(it != indices_.end()) {
                        it->second.updated = false;
                    }
                }
                saved_any = true;
            }
        }

        return saved_any;
    }

    std::string getUserPath(const std::string& username) { return data_dir_ + "/" + username; }

    std::string getIndexPath(const std::string& username, const std::string& index_name) {
        return getUserPath(username) + "/" + index_name;
    }

public:
    IndexManager(size_t max_indices,
                 const std::string& data_dir,
                 const PersistenceConfig& persistence_config = PersistenceConfig{}) :
        data_dir_(data_dir),
        persistence_config_(persistence_config) {
        std::filesystem::create_directories(data_dir);
        // Create backups directory for default system user
        std::filesystem::create_directories(data_dir + "/backups");
        metadata_manager_ = std::make_unique<MetadataManager>(data_dir);
        // Start the autosave thread
        autosave_thread_ = std::thread(&IndexManager::autosaveLoop, this);
    }

    ~IndexManager() {
        // Signal autosave thread to stop
        running_ = false;

        // Don't wait for autosave thread to exit. This allows quick restart
        if(autosave_thread_.joinable()) {
            autosave_thread_.detach();
        }
        if(persistence_config_.save_on_shutdown) {
            shutdown_requested_ = true;
            persistence_cv_.notify_all();
            LOG_DEBUG("Saving indices during shutdown");
            for(const auto& pair : indices_) {
                try {
                    // Only save indices that have been updated since last save
                    if(pair.second.updated) {
                        LOG_DEBUG("Saving updated index " << pair.first << " during shutdown");
                        saveIndex(pair.first);
                    }
                } catch(const std::exception& e) {
                    std::cerr << "Failed to save index " << pair.first
                              << " during shutdown: " << e.what() << std::endl;
                }
            }
            LOG_DEBUG("Shutdown complete");
        }
        // Clear WAL logs
        wal_logs_.clear();
    }

    // Reset the index file. It does not affect the LMDB or metadata.
    // This is used when the index is corrupted or needs to be reset.
    bool resetIndex(const std::string& index_id, const IndexConfig& config) {
        std::string base_path = data_dir_ + "/" + index_id;
        std::string index_file = base_path + "/main.idx";
        LOG_DEBUG(index_file);
        std::string recover_file = base_path + "/recover.txt";
        LOG_DEBUG(recover_file);

        // 1. Fail if directory doesn't exist
        if(!std::filesystem::exists(base_path)) {
            LOG_ERROR("Index directory does not exist: " << base_path);
            return false;
        }

        // 2. Fail if index file already exists
        if(std::filesystem::exists(index_file)) {
            LOG_ERROR("Index file already exists: " << index_file);
            return false;
        }

        // 3. Create and save empty HNSW index
        auto space_type = hnswlib::getSpaceType(config.space_type_str);
        ndd::quant::QuantizationLevel quant_level = config.quant_level;

        hnswlib::HierarchicalNSW<float> hnsw(config.max_elements,
                                             space_type,
                                             config.dim,
                                             config.M,
                                             config.ef_construction,
                                             settings::RANDOM_SEED,
                                             quant_level,
                                             config.checksum);
        hnsw.saveIndex(index_file);

        // 4. Write recover.txt with "0:0"
        std::ofstream fout(recover_file);
        fout << "0:0\n";
        fout.close();

        LOG_INFO("Index reset complete and saved: " << index_id);
        return true;
    }

    // Helper method to validate backup names
    std::pair<bool, std::string> validateBackupName(const std::string& backup_name) const {
        if(backup_name.empty()) {
            return std::make_pair(false, "Backup name cannot be empty");
        }

        // Check length limit (most filesystems limit to 255 chars)
        if(backup_name.length() > MAX_BACKUP_NAME_LENGTH) {
            return std::make_pair(false,
                                  "Backup name too long (max "
                                          + std::to_string(MAX_BACKUP_NAME_LENGTH)
                                          + " characters)");
        }

        // Use regex to check for alphanumeric, underscores, and hyphens
        static const std::regex backup_name_regex("^[a-zA-Z0-9_-]+$");
        if(!std::regex_match(backup_name, backup_name_regex)) {
            return std::make_pair(false,
                                  "Invalid backup name: only alphanumeric, underscores, "
                                  "and hyphens allowed");
        }

        return std::make_pair(true, "");
    }

    // Backup methods
    std::pair<bool, std::string> createBackup(const std::string& index_id,
                                              const std::string& backup_name) {
        // 1. Validate backup name
        std::pair<bool, std::string> result = validateBackupName(backup_name);
        if(!result.first) {
            return result;
        }

        // 2. Parse user and index name
        std::string user_id, index_name;
        size_t pos = index_id.find('/');
        if(pos != std::string::npos) {
            user_id = index_id.substr(0, pos);
            index_name = index_id.substr(pos + 1);
        } else {
            return {false, "Invalid index ID format"};
        }

        // 3. Get index entry and lock
        auto& entry = getIndexEntry(index_id);
        std::lock_guard<std::mutex> operation_lock(entry.operation_mutex);

        // 4. Force save
        saveIndexInternal(entry);

        // 5. Prepare paths - simplified for single-user system
        std::string backup_dir_root = data_dir_ + "/backups";
        std::string backup_dir = backup_dir_root + "/" + backup_name;
        std::string backup_tar = backup_dir_root + "/" + backup_name + ".tar.gz";
        std::string source_dir = data_dir_ + "/" + index_id;

        if(std::filesystem::exists(backup_tar)) {
            return {false, "Backup already exists: " + backup_name};
        }

        // 6. Copy files to temporary directory
        std::filesystem::create_directories(backup_dir);
        std::filesystem::copy(source_dir, backup_dir, std::filesystem::copy_options::recursive);

        // 7. Calculate uncompressed size and write metadata.json
        size_t uncompressed_size = 0;
        for(const auto& file : std::filesystem::recursive_directory_iterator(backup_dir)) {
            if(!std::filesystem::is_directory(file)) {
                uncompressed_size += std::filesystem::file_size(file);
            }
        }

        auto meta = metadata_manager_->getMetadata(index_id);
        if(meta) {
            nlohmann::json j;
            j["original_index"] = index_name;
            j["timestamp"] = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            j["size_mb"] = uncompressed_size / MB;
            j["params"] = {{"M", meta->M},
                           {"ef_construction", meta->ef_con},
                           {"dim", meta->dimension},
                           {"sparse_dim", meta->sparse_dim},
                           {"space_type", meta->space_type_str},
                           {"quant_level", static_cast<int>(meta->quant_level)},
                           {"total_elements", meta->total_elements},
                           {"checksum", meta->checksum}};

            std::ofstream meta_file(backup_dir + "/metadata.json");
            meta_file << j.dump(4);
        }

        // 7. Create tar.gz archive from the backup directory using libarchive
        std::string error_msg;
        if(!ndd::ArchiveUtils::createTarGz(backup_dir, backup_tar, error_msg)) {
            // Clean up on failure
            std::filesystem::remove_all(backup_dir);
            return {false, "Failed to create compressed backup archive: " + error_msg};
        }

        // 8. Remove the temporary uncompressed directory
        std::filesystem::remove_all(backup_dir);

        LOG_INFO("Created compressed backup: " << backup_tar);
        return {true, ""};
    }

    std::vector<std::string> listBackups() {
        std::vector<std::string> backups;
        std::string backup_dir = data_dir_ + "/backups";

        if(!std::filesystem::exists(backup_dir)) {
            return backups;
        }

        for(const auto& entry : std::filesystem::directory_iterator(backup_dir)) {
            if(entry.is_regular_file() && entry.path().extension() == ".gz") {
                std::string filename = entry.path().filename().string();

                // Check if it's a .tar.gz file
                if(filename.size() > 7 && filename.substr(filename.size() - 7) == ".tar.gz") {
                    // Remove .tar.gz extension to get backup name
                    std::string backup_name = filename.substr(0, filename.size() - 7);
                    backups.push_back(backup_name);
                }
            }
        }
        return backups;
    }

    std::pair<bool, std::string> restoreBackup(const std::string& backup_name,
                                               const std::string& target_index_name) {
        // 1. Validate backup name
        std::pair<bool, std::string> result = validateBackupName(backup_name);
        if(!result.first) {
            return result;
        }

        // Use default username for single-user system
        std::string user_id = settings::DEFAULT_USERNAME;
        std::string backup_dir_root = data_dir_ + "/backups";
        std::string backup_tar = backup_dir_root + "/" + backup_name + ".tar.gz";
        std::string backup_extract_dir = backup_dir_root + "/" + backup_name;
        std::string target_index_id = user_id + "/" + target_index_name;
        std::string target_dir = data_dir_ + "/" + target_index_id;

        // 2. Validation - check for tar.gz file
        if(!std::filesystem::exists(backup_tar)) {
            return {false, "Backup not found: " + backup_name};
        }
        if(metadata_manager_->getMetadata(target_index_id).has_value()) {
            return {false, "Target index already exists"};
        }

        // 3. Extract tar.gz to temporary directory using libarchive
        std::string error_msg;
        if(!ndd::ArchiveUtils::extractTarGz(backup_tar, backup_extract_dir, error_msg)) {
            return {false, "Failed to extract backup archive: " + error_msg};
        }

        // check if any folder is present in backup_extract_dir
        std::vector<std::string> folders;
        for(const auto& entry : std::filesystem::directory_iterator(backup_extract_dir)) {
            if(entry.is_directory()) {
                folders.push_back(entry.path().string());
            }
        }

        if(folders.size() != 1) {
            std::filesystem::remove_all(backup_extract_dir);
            return {false, "Backup extraction failed - directory not found"};
        }

        std::string backup_dir = folders[0];

        try {
            // 3. Read metadata
            std::ifstream f(backup_dir + "/metadata.json");
            if(!f.good()) {
                std::filesystem::remove_all(backup_extract_dir);
                return {false, "Backup metadata missing"};
            }
            nlohmann::json meta_json = nlohmann::json::parse(f);

            // 4. Copy files
            std::filesystem::create_directories(target_dir);
            std::filesystem::copy(backup_dir,
                                  target_dir,
                                  std::filesystem::copy_options::recursive
                                          | std::filesystem::copy_options::overwrite_existing);

            // Remove metadata.json from the restored index folder as it's not needed there
            std::filesystem::remove(target_dir + "/metadata.json");

            // 5. Register index
            IndexMetadata new_meta;
            new_meta.name = target_index_name;
            new_meta.dimension = meta_json["params"]["dim"];
            new_meta.sparse_dim = meta_json["params"].value("sparse_dim", 0ul);
            new_meta.M = meta_json["params"]["M"];
            new_meta.ef_con = meta_json["params"]["ef_construction"];
            new_meta.space_type_str = meta_json["params"]["space_type"];
            new_meta.quant_level = static_cast<ndd::quant::QuantizationLevel>(
                    meta_json["params"]["quant_level"].get<int>());
            new_meta.created_at = std::chrono::system_clock::now();
            new_meta.total_elements = meta_json["params"].value("total_elements", 0ul);
            new_meta.checksum = meta_json["params"].value("checksum", -1);

            metadata_manager_->storeMetadata(target_index_id, new_meta);

            // 6. Clean up extracted temporary directory
            std::filesystem::remove_all(backup_extract_dir);

            // 7. Load index
            loadIndex(target_index_id);

            LOG_INFO("Restored backup from compressed archive: " << backup_tar);
            return {true, ""};
        } catch(const std::exception& e) {
            // Clean up on failure
            std::filesystem::remove_all(backup_extract_dir);
            return {false, "Failed to restore backup: " + std::string(e.what())};
        }
    }

    std::pair<bool, std::string> deleteBackup(const std::string& backup_name) {
        // Validate backup name
        std::pair<bool, std::string> result = validateBackupName(backup_name);
        if(!result.first) {
            return result;
        }

        std::string backup_tar = data_dir_ + "/backups/" + backup_name + ".tar.gz";
        if(std::filesystem::exists(backup_tar)) {
            std::filesystem::remove(backup_tar);
            LOG_INFO("Deleted compressed backup: " << backup_tar);
            return {true, ""};
        } else {
            return {false, "Backup not found"};
        }
    }

    bool createIndex(const std::string& index_id,
                     const IndexConfig& config,
                     UserType user_type = UserType::Admin,
                     size_t size_in_millions = 0) {
        // Get usernmae and index name from index_id
        auto pos = index_id.find('/');
        if(pos == std::string::npos) {
            throw std::runtime_error("Invalid index ID");
        }
        std::string username = index_id.substr(0, pos);
        std::string index_name = index_id.substr(pos + 1);
        // Check if index already exists in metadata
        auto existing_indices = metadata_manager_->listUserIndexes(username);
        for(const auto& existing : existing_indices) {
            if(existing.first == index_name) {
                throw std::runtime_error("Index with this name already exists for this user");
            }
        }

        // Validate max_elements against user limits (unless admin with custom size)
        size_t max_vectors_allowed = getMaxVectorsPerIndex(user_type);
        if(user_type != UserType::Admin || size_in_millions == 0) {
            if(config.max_elements > max_vectors_allowed) {
                throw std::runtime_error("Index size " + std::to_string(config.max_elements)
                                         + " exceeds limit of "
                                         + std::to_string(max_vectors_allowed) + " vectors for "
                                         + userTypeToString(user_type) + " users");
            }
        }

        // Check file system without lock
        std::string index_path = data_dir_ + "/" + index_id + "/main.idx";
        if(std::filesystem::exists(index_path)) {
            throw std::runtime_error("Index already exists");
        }

        // Evict if needed (clean indices only)
        {
            std::unique_lock<std::shared_mutex> temp_lock(indices_mutex_);
            evictIfNeeded();
        }

        hnswlib::SpaceType space_type = hnswlib::getSpaceType(config.space_type_str);
        std::string lmdb_dir = data_dir_ + "/" + index_id + "/ids";
        std::string vector_storage_dir = data_dir_ + "/" + index_id + "/vectors";

        //create the directory and initialize sequence for IDMapper
        LOG_INFO("Creating IDMapper for index "
                 << index_id << " with user type: " << userTypeToString(user_type));

        // IDMapper now uses tier-based fixed bloom filter sizing based on user_type
        auto id_mapper = std::make_shared<IDMapper>(lmdb_dir, true, user_type);

        std::filesystem::create_directories(vector_storage_dir);

        // Create HNSW directly with all necessary parameters
        ndd::quant::QuantizationLevel quant_level = config.quant_level;
        auto vector_storage =
                std::make_shared<VectorStorage>(vector_storage_dir, config.dim, config.quant_level);

        // Initialize Sparse Storage if needed
        std::unique_ptr<ndd::SparseVectorStorage> sparse_storage = nullptr;
        if(config.sparse_dim > 0) {
            std::string sparse_storage_dir = data_dir_ + "/" + index_id + "/sparse";
            sparse_storage = std::make_unique<ndd::SparseVectorStorage>(sparse_storage_dir);
            if(!sparse_storage->initialize()) {
                throw std::runtime_error("Failed to initialize sparse storage");
            }
        }

        auto alg = std::make_unique<hnswlib::HierarchicalNSW<float>>(config.max_elements,
                                                                     space_type,
                                                                     config.dim,
                                                                     config.M,
                                                                     config.ef_construction,
                                                                     settings::RANDOM_SEED,
                                                                     quant_level,
                                                                     config.checksum);

        alg->setVectorFetcher([vs = vector_storage](ndd::idInt label, uint8_t* buffer) {
            return vs->get_vector(label, buffer);
        });

        // Create WAL during index creation
        getOrCreateWAL(index_id);

        // Add to indices with minimal lock scope
        {
            std::unique_lock<std::shared_mutex> lock(indices_mutex_);
            // Emplace directly with constructor arguments to avoid move
            auto [it, inserted] =
                    indices_.emplace(std::piecewise_construct,
                                     std::forward_as_tuple(index_id),
                                     std::forward_as_tuple(index_id,
                                                           config.sparse_dim,
                                                           std::move(alg),
                                                           id_mapper,
                                                           vector_storage,
                                                           std::move(sparse_storage),
                                                           std::chrono::system_clock::now()));
            it->second.markUpdated();
            indices_list_.push_front(index_id);
        }

        // Create and store index metadata
        IndexMetadata metadata_entry;
        metadata_entry.name = index_name;
        metadata_entry.dimension = config.dim;
        metadata_entry.sparse_dim = config.sparse_dim;
        metadata_entry.space_type_str = config.space_type_str;
        metadata_entry.quant_level = config.quant_level;
        metadata_entry.checksum = config.checksum;
        metadata_entry.total_elements = 0;
        metadata_entry.M = config.M;
        metadata_entry.ef_con = config.ef_construction;
        metadata_entry.created_at = std::chrono::system_clock::now();

        if(!metadata_manager_->storeMetadata(index_id, metadata_entry)) {
            throw std::runtime_error("Failed to store index metadata");
        }

        LOG_INFO("Saving newly created index " << index_id);
        // Index is marked as updated so it needs to be saved immediately for crash recovery
        saveIndex(index_id);
        return true;
    }

    std::vector<std::pair<std::string, IndexMetadata>>
    listUserIndexes(const std::string& username) {
        // Use the metadata manager directly to get the list of indexes
        return metadata_manager_->listUserIndexes(username);
    }
    std::vector<std::pair<std::string, IndexMetadata>> listAllIndexes() {
        // Use the metadata manager directly to get the list of indexes
        return metadata_manager_->listAllIndexes();
    }

    void loadIndex(const std::string& index_id) {
        std::string index_path = data_dir_ + "/" + index_id + "/main.idx";
        std::string lmdb_dir = data_dir_ + "/" + index_id + "/ids";
        std::string vector_storage_dir = data_dir_ + "/" + index_id + "/vectors";
        if(!std::filesystem::exists(index_path) || !std::filesystem::exists(lmdb_dir)
           || !std::filesystem::exists(vector_storage_dir)) {
            throw std::runtime_error("Required files missing for index: " + index_id);
        }

        // Load metadata to get sparse_dim
        auto metadata = metadata_manager_->getMetadata(index_id);
        size_t sparse_dim = 0;
        if(metadata) {
            sparse_dim = metadata->sparse_dim;
        }

        // Step 1: Load HNSW index (automatically adjusts cache based on element count and cache
        // percentage)
        std::unique_ptr<hnswlib::HierarchicalNSW<float>> alg;
        try {
            alg = std::make_unique<hnswlib::HierarchicalNSW<float>>(index_path, 0);

        } catch(const std::exception& e) {
            throw std::runtime_error("Cannot load index '" + index_id + "': " + e.what());
        }

        // Step 2: Create IDMapper and VectorStorage - IDMapper handles bloom filter initialization
        auto id_mapper = std::make_shared<IDMapper>(lmdb_dir, false);
        auto vector_storage = std::make_shared<VectorStorage>(
                vector_storage_dir, alg->getDimension(), alg->getQuantLevel());

        // Initialize Sparse Storage if sparse_dim > 0
        std::unique_ptr<ndd::SparseVectorStorage> sparse_storage;
        if(sparse_dim > 0) {
            std::string sparse_db_path = data_dir_ + "/" + index_id + "/sparse.db";
            sparse_storage = std::make_unique<ndd::SparseVectorStorage>(sparse_db_path);
            if(!sparse_storage->initialize()) {
                throw std::runtime_error("Failed to initialize sparse storage for index: "
                                         + index_id);
            }
        }

        // Set up vector fetcher
        alg->setVectorFetcher([vs = vector_storage](ndd::idInt label, uint8_t* buffer) {
            return vs->get_vector(label, buffer);
        });

        LOG_DEBUG("Loaded index: " << index_id);
        LOG_DEBUG("Created space for index: " << index_id);

        // Step 3: Update cache entry so that index becomes available to other threads
        // Emplace directly with constructor arguments to avoid move
        auto [it, inserted] =
                indices_.emplace(std::piecewise_construct,
                                 std::forward_as_tuple(index_id),
                                 std::forward_as_tuple(index_id,
                                                       sparse_dim,
                                                       std::move(alg),
                                                       id_mapper,
                                                       vector_storage,
                                                       std::move(sparse_storage),
                                                       std::chrono::system_clock::now()));
        indices_list_.push_front(index_id);

        // Handle WAL recovery using the IndexManager's method
        recoverFromWAL(index_id);
    }

    // Reload index: save (if updated), evict from memory, and reload
    // Cache size is automatically checked and adjusted if < 5% of element count during reload
    bool reload(const std::string& index_id) {
        LOG_INFO("Starting reload for " << index_id);

        try {
            // Phase 1: Save index if it was updated
            {
                std::shared_lock<std::shared_mutex> lock(indices_mutex_);
                auto it = indices_.find(index_id);
                if(it != indices_.end() && it->second.updated) {
                    LOG_DEBUG("Saving updated index before reload: " << index_id);
                    saveIndex(index_id);
                }
            }

            // Phase 2: Evict from memory
            {
                std::unique_lock<std::shared_mutex> lock(indices_mutex_);
                auto it = indices_.find(index_id);
                if(it != indices_.end()) {
                    // Remove from LRU list
                    auto list_it = std::find(indices_list_.begin(), indices_list_.end(), index_id);
                    if(list_it != indices_list_.end()) {
                        indices_list_.erase(list_it);
                    }
                    indices_.erase(it);
                    LOG_INFO("Evicted " << index_id << " from cache");
                }
            }

            // Phase 3: Reload (cache adjustment happens automatically in loadIndex)
            loadIndex(index_id);

            // Phase 4: Report final state
            {
                std::shared_lock<std::shared_mutex> lock(indices_mutex_);
                auto it = indices_.find(index_id);
                if(it != indices_.end()) {
                    // Cache removed
                    LOG_INFO("Reloaded "
                             << index_id << ", bloom: fixed size"
                             << ", index elements: " << it->second.alg->getElementsCount());
                }
            }

            return true;
        } catch(const std::exception& e) {
            LOG_ERROR("Failed to reload " << index_id << ": " << e.what());
            return false;
        }
    }

    // Add this new function to reload just the algorithm part while preserving the CacheEntry
    void reloadIndex(const std::string& index_id) {
        auto it = indices_.find(index_id);
        if(it == indices_.end()) {
            return;  // Index not in cache
        }

        CacheEntry& entry = it->second;

        std::string index_path = data_dir_ + "/" + index_id + "/main.idx";

        // Create a new HNSW algorithm object from the saved file
        auto new_alg = std::make_unique<hnswlib::HierarchicalNSW<float>>(index_path, 0);

        // Set the vector fetcher to use our storage
        new_alg->setVectorFetcher([vs = entry.vector_storage](ndd::idInt label, uint8_t* buffer) {
            return vs->get_vector(label, buffer);
        });

        // Replace the algorithm in the existing entry
        entry.alg = std::move(new_alg);
    }

    template <typename VectorType>
    bool addVectors(const std::string& index_id, const std::vector<VectorType>& vectors) {
        try {
            // Get the index entry (loads if needed, handles all locking)
            auto& entry = getIndexEntry(index_id);

            // Use per-index operation mutex to prevent concurrent operations
            std::lock_guard<std::mutex> operation_lock(entry.operation_mutex);

            // Extract string IDs first
            LOG_DEBUG("Adding " << vectors.size() << " vectors to index " << index_id);
            if(vectors.empty()) {
                LOG_DEBUG("No vectors to add");
                return false;
            }

            // CRITICAL FIX: Pass WAL to create_ids_batch for atomic logging
            WriteAheadLog* wal = getOrCreateWAL(index_id);

            std::vector<std::string> str_ids;
            str_ids.reserve(vectors.size());
            for(const auto& vec : vectors) {
                str_ids.push_back(vec.id);
            }
            LOG_DEBUG("Extracted " << str_ids.size() << " string IDs from vectors");
            std::vector<std::pair<idInt, bool>> numeric_ids;
            // Get or create numeric IDs in batch - this returns ids.
            // If str_id already exists, it will return the old numeric ID
            if(entry.alg->getDeletedCount() > 0) {
                // There are deleted IDs, we need to reuse them
                numeric_ids = entry.id_mapper->create_ids_batch<true>(str_ids, wal);
            } else {
                // No deleted IDs, just create new ones
                numeric_ids = entry.id_mapper->create_ids_batch<false>(str_ids, wal);
            }
            LOG_DEBUG("Created " << numeric_ids.size() << " numeric IDs for string IDs");

            // Handle Sparse Vectors if storage is initialized
            if(entry.sparse_storage) {
                if constexpr(std::is_same_v<VectorType, ndd::HybridVectorObject>) {
                    std::vector<std::pair<ndd::idInt, ndd::SparseVector>> sparse_batch;
                    sparse_batch.reserve(vectors.size());

                    for(size_t i = 0; i < vectors.size(); ++i) {
                        const auto& vec = vectors[i];
                        if(!vec.sparse_ids.empty()) {
                            // Sort indices and values together
                            std::vector<std::pair<uint32_t, float>> pairs;
                            pairs.reserve(vec.sparse_ids.size());
                            for(size_t k = 0; k < vec.sparse_ids.size(); ++k) {
                                pairs.emplace_back(vec.sparse_ids[k], vec.sparse_values[k]);
                            }
                            std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
                                return a.first < b.first;
                            });

                            ndd::SparseVector sparse_vec;
                            sparse_vec.indices.reserve(pairs.size());
                            sparse_vec.values.reserve(pairs.size());
                            for(const auto& p : pairs) {
                                sparse_vec.indices.push_back(p.first);
                                sparse_vec.values.push_back(p.second);
                            }

                            sparse_batch.emplace_back(numeric_ids[i].first, std::move(sparse_vec));
                        }
                    }

                    if(!sparse_batch.empty()) {
                        entry.sparse_storage->store_vectors_batch(sparse_batch);
                    }
                }
            }

            // Convert all vectors to QuantVectorObject ONCE using efficient move constructor
            std::vector<QuantVectorObject> quantized_vectors;
            quantized_vectors.reserve(vectors.size());
            ndd::quant::QuantizationLevel quant_level = entry.alg->getQuantLevel();
            auto space = entry.alg->getSpace();
            const void* dist_params = space ? space->get_dist_func_param() : nullptr;

            LOG_DEBUG("Converting " << vectors.size() << " vectors to QuantVectorObject with level "
                                    << (int)quant_level);

            // Create a mutable copy for move operations
            std::vector<VectorType> mutable_vectors = vectors;

            for(auto& vec_obj : mutable_vectors) {
                // Use efficient move constructor with internal quantization
                quantized_vectors.emplace_back(std::move(vec_obj), quant_level, dist_params);
            }
            LOG_DEBUG("QuantVectorObject conversion completed with move semantics");

            // Store quantized vectors using optimized batch function (no double conversion!)
            std::vector<std::pair<idInt, QuantVectorObject>> storage_vectors;
            storage_vectors.reserve(quantized_vectors.size());
            for(size_t i = 0; i < quantized_vectors.size(); i++) {
                // Copy QuantVectorObject for storage (we need to keep original for HNSW)
                storage_vectors.emplace_back(numeric_ids[i].first, quantized_vectors[i]);
            }
            entry.vector_storage->store_vectors_batch(storage_vectors);
            LOG_DEBUG("Stored " << storage_vectors.size()
                                << " pre-quantized vectors in vector storage");

            // Add to write ahead log using IndexManager's method
            logInsertsAndUpdates(index_id, numeric_ids);

            // Add to HNSW index in parallel using pre-quantized data from QuantVectorObject
            size_t available_threads = settings::NUM_PARALLEL_INSERTS;
            const size_t num_threads = (available_threads < quantized_vectors.size())
                                               ? available_threads
                                               : quantized_vectors.size();
            std::vector<std::thread> threads;
            const size_t chunk_size =
                    (quantized_vectors.size() + num_threads - 1) / num_threads;  // Ceiling division

            threads.reserve(num_threads);
            for(size_t t = 0; t < num_threads; t++) {
                threads.emplace_back([&, t]() {
                    // Calculate start and end indices for this thread
                    size_t start_idx = t * chunk_size;
                    size_t end_idx = (start_idx + chunk_size < quantized_vectors.size())
                                             ? (start_idx + chunk_size)
                                             : quantized_vectors.size();

                    // Process assigned chunk of vectors
                    for(size_t i = start_idx; i < end_idx; i++) {
                        const auto& quant_vec_obj = quantized_vectors[i];

                        // Use pre-quantized data directly from QuantVectorObject - no conversion
                        // needed!
                        const uint8_t* vector_data = quant_vec_obj.quant_vector.data();

                        // Add to HNSW index using pre-quantized raw bytes
                        if(numeric_ids[i].second) {
                            // If it's a new ID, add it to the index
                            entry.alg->addPoint<true>(vector_data, numeric_ids[i].first);
                        } else {
                            // If it's an update, add it to the index
                            entry.alg->addPoint<false>(vector_data, numeric_ids[i].first);
                        }
                    }
                });
            }

            // Wait for all threads to complete
            for(auto& thread : threads) {
                thread.join();
            }

            entry.markUpdated();

            // Check if we need to save based on WAL entry count after logging
            if(wal->getEntryCount() >= persistence_config_.save_every_n_updates) {
                LOG_DEBUG("Saving index " << index_id << " after " << wal->getEntryCount()
                                          << " updates");
                saveIndexInternal(entry);
            }

            PRINT_LOG_TIME();
            return true;
        } catch(const std::exception& e) {
            std::cerr << "Batch insertion failed: " << e.what() << std::endl;
            return false;
        }
    }

    // Recover a corrupted index from vectorstore and keep adding to the index in batches
    bool recoverIndex(const std::string& index_id) {
        const size_t batch_size = settings::RECOVERY_BATCH_SIZE;
        std::string base_path = data_dir_ + "/" + index_id;
        std::string recover_file = base_path + "/recover.txt";

        if(!std::filesystem::exists(recover_file)) {
            LOG_ERROR("Recover file not found: " << recover_file);
            return false;
        }

        // Step 1: Read offset and busy flag
        std::ifstream fin(recover_file);
        std::string line;
        std::getline(fin, line);
        fin.close();

        auto colon = line.find(':');
        if(colon == std::string::npos) {
            LOG_ERROR("Invalid recover.txt format");
            return false;
        }

        size_t offset = std::stoull(line.substr(0, colon));
        int flag = std::stoi(line.substr(colon + 1));
        if(flag == 1) {
            LOG_INFO("Recovery already in progress for: " << index_id);
            return false;
        }

        // Step 2: Mark as busy
        {
            std::ofstream fout(recover_file);
            fout << offset << ":1\n";
        }

        // Step 3: Load entry and acquire operation mutex for thread safety
        auto& entry = getIndexEntry(index_id);

        // FIX: Use per-index operation mutex to prevent concurrent operations
        std::lock_guard<std::mutex> operation_lock(entry.operation_mutex);

        auto cursor = entry.vector_storage->getCursor();

        // Step 4: Collect next batch
        std::vector<std::pair<ndd::idInt, std::vector<uint8_t>>> batch;
        while(cursor.hasNext() && batch.size() < batch_size) {
            auto [label, vec_bytes] = cursor.next();
            if(label < offset) {
                continue;
            }
            batch.emplace_back(label, std::move(vec_bytes));
        }

        if(batch.empty()) {
            LOG_INFO("No more vectors to recover");
            std::ofstream fout(recover_file);
            fout << offset << ":0\n";  // just mark as not busy
            return true;
        }

        // Step 5: Insert in parallel like addVectors()
        size_t num_threads = std::min(settings::NUM_RECOVERY_THREADS, batch.size());
        std::atomic<size_t> next{0};
        std::vector<std::thread> threads;

        for(size_t t = 0; t < num_threads; ++t) {
            threads.emplace_back([&]() {
                size_t i;
                while((i = next.fetch_add(1)) < batch.size()) {
                    const auto& [label, vec_bytes] = batch[i];
                    if(!vec_bytes.empty()) {
                        entry.alg->addPoint<true>(vec_bytes.data(), label);
                    } else {
                        LOG_ERROR("Skipping label " << label << " due to empty vector");
                    }
                }
            });
        }

        for(auto& th : threads) {
            th.join();
        }

        LOG_INFO("Recovered " << batch.size() << " vectors to index: " << index_id);

        // Step 6: Save index
        // Mark the index as updated so that it will be saved
        entry.markUpdated();
        // FIX: Use internal save to avoid circular lock
        saveIndexInternal(entry);

        // Step 7: Update recover.txt to next offset
        std::ofstream fout(recover_file);
        fout << (offset + batch.size()) << ":0\n";

        return true;
    }

    std::optional<ndd::VectorObject> getVector(const std::string& index_id,
                                               const std::string& str_id) {
        try {
            auto& entry = getIndexEntry(index_id);
            ndd::idInt numeric_id = entry.id_mapper->get_id(str_id);
            if(numeric_id == 0) {
                return std::nullopt;
            }

            std::vector<uint8_t> vec_bytes = entry.vector_storage->get_vector(numeric_id);
            ndd::VectorMeta meta = entry.vector_storage->get_meta(numeric_id);

            ndd::VectorObject obj;
            obj.id = meta.id;
            obj.meta = meta.meta;
            obj.filter = meta.filter;
            obj.norm = meta.norm;

            // Convert raw bytes to float vector using unified dequantization function
            ndd::quant::QuantizationLevel quant_level = entry.alg->getQuantLevel();
            std::vector<float> float_data =
                    ndd::quant::get_quantizer_dispatch(quant_level)
                            .dequantize(vec_bytes.data(), entry.alg->getDimension());

            // Add the float data to the msgpack
            obj.vector = {float_data.begin(), float_data.end()};

            return obj;
        } catch(const std::exception& e) {
            std::cerr << "Error retrieving vector: " << e.what() << std::endl;
            return std::nullopt;
        }
    }

    // Delete vectors from id mapper, delete filter and mark as deleted in HNSW. Does not delete
    // meta, vector data Meta and vector data will be overwritten when the id is reused
    bool deleteVectorsByIds(CacheEntry& entry, const std::vector<ndd::idInt>& numeric_ids) {
        try {
            for(ndd::idInt numeric_id : numeric_ids) {
                auto meta = entry.vector_storage->get_meta(numeric_id);
                // Remove ID mapping by getting the string id from metadata
                auto stored_ids = entry.id_mapper->deletePoints({meta.id});
                if(stored_ids[0] != numeric_id) {
                    LOG_DEBUG("Error: Mismatch in stored ID and numeric ID "
                              << stored_ids[0] << " != " << numeric_id);
                    continue;
                }
                // Remove the filter
                entry.vector_storage->deleteFilter(numeric_id, meta.filter);
                // Mark as deleted in HNSW index
                entry.alg->markDelete(numeric_id);
            }
            // Add the list to write ahead log using IndexManager's method
            logDeletions(entry.index_id, numeric_ids);

            // Mark the index as updated
            entry.markUpdated();

            return true;
        } catch(const std::exception& e) {
            std::cerr << "Failed to delete vectors: " << e.what() << std::endl;
            return false;
        }
    }

    size_t deleteVectorsByFilter(const std::string& index_id, const nlohmann::json& filter_array) {
        try {
            auto& entry = getIndexEntry(index_id);

            // Use per-index operation mutex to prevent concurrent operations
            std::lock_guard<std::mutex> operation_lock(entry.operation_mutex);

            auto numeric_ids =
                    entry.vector_storage->filter_store_->getIdsMatchingFilter(filter_array);
            LOG_DEBUG("Filter matched " << numeric_ids.size() << " vectors");

            if(deleteVectorsByIds(entry, numeric_ids)) {
                // Check if we need to save based on WAL entry count after logging
                WriteAheadLog* wal = getOrCreateWAL(index_id);
                if(wal->getEntryCount() >= persistence_config_.save_every_n_updates) {
                    LOG_DEBUG("Saving index " << index_id << " after " << wal->getEntryCount()
                                              << " updates");
                    saveIndexInternal(entry);
                }
                return numeric_ids.size();
            } else {
                return 0;
            }
        } catch(const std::exception& e) {
            std::cerr << "Failed to delete vectors by filter: " << e.what() << std::endl;
            return 0;
        }
    }

    // Update filters for a batch of vectors
    size_t updateFilters(const std::string& index_id,
                         const std::vector<std::pair<std::string, std::string>>& updates) {
        try {
            auto& entry = getIndexEntry(index_id);
            std::lock_guard<std::mutex> operation_lock(entry.operation_mutex);

            size_t updated_count = 0;
            for(const auto& [str_id, new_filter] : updates) {
                ndd::idInt numeric_id = entry.id_mapper->get_id(str_id);
                if(numeric_id == 0) {
                    LOG_DEBUG("updateFilters: ID not found: " << str_id);
                    continue;
                }

                entry.vector_storage->updateFilter(numeric_id, new_filter);
                updated_count++;
            }

            if(updated_count > 0) {
                entry.markUpdated();
            }

            return updated_count;
        } catch(const std::exception& e) {
            std::cerr << "Failed to update filters: " << e.what() << std::endl;
            return 0;
        }
    }

    // Delete a single vector by string ID - vector data will not be deleted. The meta and filter
    // will be deleted and the vector will be marked as deleted in HNSW. The id will put in the
    // deleted_ids in id mapper and will be reused for new vectors
    bool deleteVector(const std::string& index_id, const std::string& str_id) {
        try {
            auto& entry = getIndexEntry(index_id);

            // Use per-index operation mutex to prevent concurrent operations
            std::lock_guard<std::mutex> operation_lock(entry.operation_mutex);

            size_t numeric_id = entry.id_mapper->get_id(str_id);
            if(numeric_id == 0) {
                return false;
            }
            bool result = deleteVectorsByIds(entry, {static_cast<idInt>(numeric_id)});

            // Check if we need to save based on WAL entry count after logging
            if(result) {
                WriteAheadLog* wal = getOrCreateWAL(index_id);
                if(wal->getEntryCount() >= persistence_config_.save_every_n_updates) {
                    LOG_DEBUG("Saving index " << index_id << " after " << wal->getEntryCount()
                                              << " updates");
                    saveIndexInternal(entry);
                }
            }

            return result;
        } catch(const std::exception& e) {
            std::cerr << "Failed to delete vector: " << e.what() << std::endl;
            return false;
        }
    }

    std::optional<std::vector<ndd::VectorResult>> searchKNN(const std::string& index_id,
                                                            const std::vector<float>& query,
                                                            size_t k,
                                                            const nlohmann::json& filter_array,
                                                            bool include_vectors = false,
                                                            size_t ef = 0) {
        return searchKNN(index_id, query, {}, {}, k, filter_array, include_vectors, ef);
    }

    std::optional<std::vector<ndd::VectorResult>>
    searchKNN(const std::string& index_id,
              const std::vector<float>& query,
              const std::vector<uint32_t>& sparse_indices,
              const std::vector<float>& sparse_values,
              size_t k,
              const nlohmann::json& filter_array,
              bool include_vectors = false,
              size_t ef = 0) {
        try {
            auto& entry = getIndexEntry(index_id);
            entry.searchCount += k;

            // 1. Sparse Search (Async)
            std::future<std::vector<std::pair<ndd::idInt, float>>> sparse_future;
            if(entry.sparse_storage && !sparse_indices.empty()) {
                sparse_future = std::async(std::launch::async, [&]() {
                    ndd::SparseVector sparse_query;
                    // Sort indices and values together
                    std::vector<std::pair<uint32_t, float>> pairs;
                    pairs.reserve(sparse_indices.size());
                    for(size_t i = 0; i < sparse_indices.size(); ++i) {
                        pairs.emplace_back(sparse_indices[i], sparse_values[i]);
                    }
                    std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
                        return a.first < b.first;
                    });

                    sparse_query.indices.reserve(pairs.size());
                    sparse_query.values.reserve(pairs.size());
                    for(const auto& p : pairs) {
                        sparse_query.indices.push_back(p.first);
                        sparse_query.values.push_back(p.second);
                    }

                    return entry.sparse_storage->search(sparse_query, k);
                });
            }

            // 2. Dense Search (Main Thread)
            std::vector<std::pair<float, ndd::idInt>> dense_results;

            if(!query.empty()) {
                // Convert query to bytes using the wrapper method
                ndd::quant::QuantizationLevel quant_level = entry.alg->getQuantLevel();
                auto space = entry.alg->getSpace();
                std::vector<uint8_t> query_bytes =
                        ndd::quant::get_quantizer_dispatch(quant_level).quantize(query);
                // Always try post-filtering first (or direct search if no filter)
                // When there are filters, we need to search for more candidates to account for
                // filtering
                size_t search_k = filter_array.empty() ? k : std::max(ef, k * 2);
                dense_results = entry.alg->searchKnn(query_bytes.data(), search_k, ef);
            }

            // 3. Get Sparse Results (Join)
            std::vector<std::pair<ndd::idInt, float>> sparse_results;
            if(sparse_future.valid()) {
                sparse_results = sparse_future.get();
            }

            // 3. Combine Results
            std::vector<std::pair<float, ndd::idInt>> final_candidates;

            if(dense_results.empty() && sparse_results.empty()) {
                return std::vector<ndd::VectorResult>();
            } else if(sparse_results.empty()) {
                // Only dense results
                final_candidates.reserve(dense_results.size());
                for(const auto& p : dense_results) {
                    final_candidates.emplace_back(p.first, p.second);
                }
            } else if(dense_results.empty()) {
                // Only sparse results
                final_candidates.reserve(sparse_results.size());
                for(const auto& p : sparse_results) {
                    final_candidates.emplace_back(p.second, p.first);
                }
            } else {
                // Hybrid results - RRF
                std::unordered_map<ndd::idInt, float> combined_scores;
                const float k_rrf = 60.0f;

                for(size_t i = 0; i < dense_results.size(); ++i) {
                    combined_scores[dense_results[i].second] += 1.0f / (k_rrf + i + 1);
                }

                for(size_t i = 0; i < sparse_results.size(); ++i) {
                    combined_scores[sparse_results[i].first] += 1.0f / (k_rrf + i + 1);
                }

                final_candidates.reserve(combined_scores.size());
                for(const auto& [id, score] : combined_scores) {
                    final_candidates.emplace_back(score, id);
                }

                std::sort(final_candidates.begin(),
                          final_candidates.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
            }

            std::vector<ndd::VectorResult> results;
            results.reserve(final_candidates.size());
            LOG_DEBUG("Search results size: " << final_candidates.size());

            // Process and filter results
            size_t filtered_count = 0;
            for(const auto& p : final_candidates) {
                // Get metadata
                ndd::VectorMeta meta = entry.vector_storage->get_meta(p.second);

                // Apply filter
                if(!filter_array.empty()
                   && !entry.vector_storage->matches_filter(p.second, meta, filter_array)) {
                    continue;
                }

                ndd::VectorResult result;
                result.id = meta.id;
                result.filter = meta.filter;
                result.meta = meta.meta;
                result.similarity = p.first;

                result.norm = meta.norm;

                if(include_vectors) {
                    std::vector<uint8_t> vec_bytes = entry.vector_storage->get_vector(p.second);
                    if(!vec_bytes.empty()) {
                        ndd::quant::QuantizationLevel quant_level = entry.alg->getQuantLevel();
                        std::vector<float> float_data =
                                ndd::quant::get_quantizer_dispatch(quant_level)
                                        .dequantize(vec_bytes.data(), entry.alg->getDimension());
                        result.vector = {float_data.begin(), float_data.end()};
                    }
                }

                results.push_back(std::move(result));
                filtered_count++;

                // Early exit when we have enough results
                if(filtered_count >= k) {
                    break;
                }
            }

            // Check if post-filtering gave poor results and pre-filtering might help
            // Only for dense search for now as sparse pre-filtering is not implemented
            if(!filter_array.empty()
               && filtered_count < k * settings::PREFILTER_RESULT_RATIO_THRESHOLD && !query.empty()
               && sparse_results.empty()) {
                size_t filter_cardinality =
                        entry.vector_storage->filter_store_->countIdsMatchingFilter(filter_array);
                LOG_DEBUG("Post-filter gave poor results ("
                          << filtered_count << "/" << k
                          << "), checking pre-filter option. Cardinality: " << filter_cardinality);

                if(filter_cardinality < settings::PREFILTER_CARDINALITY_THRESHOLD) {
                    LOG_DEBUG("Using pre-filter approach due to poor post-filter results");

                    // Pre-filter: Get filtered IDs and do bruteforce search
                    auto filtered_ids =
                            entry.vector_storage->filter_store_->getIdsMatchingFilter(filter_array);
                    LOG_DEBUG("Pre-filter: got " << filtered_ids.size() << " filtered IDs");

                    if(!filtered_ids.empty()) {
                        // Convert size_t to size_t for batch retrieval
                        std::vector<ndd::idInt> numeric_ids(filtered_ids.begin(),
                                                            filtered_ids.end());

                        // Get vectors for filtered IDs
                        auto vector_batch = entry.vector_storage->get_vectors_batch(numeric_ids);
                        LOG_DEBUG("Pre-filter: retrieved " << vector_batch.size() << " vectors");

                        // Prepare subset for bruteforce search
                        std::vector<std::pair<idInt, std::vector<uint8_t>>> vector_subset;
                        vector_subset.reserve(vector_batch.size());

                        for(const auto& [numeric_id, vec_bytes] : vector_batch) {
                            vector_subset.emplace_back(static_cast<idInt>(numeric_id), vec_bytes);
                        }

                        // Perform bruteforce search on subset using HNSW's space interface
                        ndd::quant::QuantizationLevel quant_level = entry.alg->getQuantLevel();
                        auto space = entry.alg->getSpace();
                        std::vector<uint8_t> query_bytes =
                                ndd::quant::get_quantizer_dispatch(quant_level).quantize(query);
                        auto prefilter_results = hnswlib::searchKnnSubset<float>(
                                query_bytes.data(), vector_subset, k, space);

                        LOG_DEBUG("Pre-filter: bruteforce search returned "
                                  << prefilter_results.size() << " results");

                        // Convert results to VectorResult format
                        std::vector<ndd::VectorResult> prefilter_final_results;
                        prefilter_final_results.reserve(prefilter_results.size());

                        for(const auto& [distance, label] : prefilter_results) {
                            ndd::idInt numeric_id = static_cast<ndd::idInt>(label);
                            ndd::VectorMeta meta = entry.vector_storage->get_meta(numeric_id);

                            ndd::VectorResult result;
                            result.id = meta.id;
                            result.filter = meta.filter;
                            result.meta = meta.meta;

                            if(entry.alg->getSpaceType() == hnswlib::COSINE_SPACE
                               || entry.alg->getSpaceType() == hnswlib::IP_SPACE) {
                                result.similarity = 1.0f - distance;
                            } else {
                                result.similarity = distance;
                            }

                            result.norm = meta.norm;

                            if(include_vectors) {
                                // Find the vector bytes from our batch
                                auto it = std::find_if(vector_batch.begin(),
                                                       vector_batch.end(),
                                                       [numeric_id](const auto& pair) {
                                                           return pair.first == numeric_id;
                                                       });

                                if(it != vector_batch.end()) {
                                    const auto& vec_bytes = it->second;

                                    ndd::quant::QuantizationLevel quant_level =
                                            entry.alg->getQuantLevel();
                                    std::vector<float> float_data =
                                            ndd::quant::get_quantizer_dispatch(quant_level)
                                                    .dequantize(vec_bytes.data(),
                                                                entry.alg->getDimension());
                                    result.vector = {float_data.begin(), float_data.end()};
                                }
                            }

                            prefilter_final_results.push_back(std::move(result));
                        }

                        return prefilter_final_results;
                    }
                } else {
                    LOG_DEBUG("Filter cardinality too high for pre-filtering ("
                              << filter_cardinality
                              << " >= " << settings::PREFILTER_CARDINALITY_THRESHOLD
                              << "), returning post-filter results");
                }
            }

            // Ensure we don't return more than k results
            if(results.size() > k) {
                results.resize(k);
            }
            return results;
        } catch(const std::exception& e) {
            std::cerr << "Search error: " << e.what() << std::endl;
            return std::nullopt;
        }
    }

    bool deleteIndex(const std::string& index_id) {
        std::unique_lock<std::shared_mutex> write_lock(indices_mutex_);
        // Remove from in-memory structures if loaded
        auto it = indices_.find(index_id);
        if(it != indices_.end()) {
            auto indx_it = std::find(indices_list_.begin(), indices_list_.end(), index_id);
            if(indx_it != indices_list_.end()) {
                indices_list_.erase(indx_it);
            }
            indices_.erase(it);
        }

        // Delete metadata
        metadata_manager_->deleteMetadata(index_id);

        // Move to deleted directory instead of removing
        std::string index_dir = data_dir_ + "/" + index_id;
        std::string deleted_dir = data_dir_ + "/deleted";

        try {
            LOG_DEBUG("Deleting index: " << index_dir);
            if(std::filesystem::exists(index_dir)) {
                // Create deleted directory if it doesn't exist
                std::filesystem::create_directories(deleted_dir);

                // Parse username and index_name from index_id (format: username/index_name)
                size_t slash_pos = index_id.find('/');
                std::string username = index_id.substr(0, slash_pos);
                std::string index_name = index_id.substr(slash_pos + 1);

                // Generate backup name with random suffix
                std::string rand_suffix = random_generator::rand_alphanum(4);
                std::string backup_path =
                        deleted_dir + "/" + username + "_" + index_name + "_" + rand_suffix;

                // Move the directory
                std::filesystem::rename(index_dir, backup_path);
                return true;
            }
        } catch(const std::filesystem::filesystem_error& e) {
            std::cerr << "Failed to move index to deleted directory: " << e.what() << std::endl;
            return false;
        }

        return false;
    }

    std::optional<IndexInfo> getIndexInfo(const std::string& index_id) {
        auto& entry = getIndexEntry(index_id);
        IndexInfo indx = {entry.alg->getElementsCount(),
                          entry.alg->getDimension(),
                          entry.sparse_dim,
                          entry.alg->getSpaceTypeStr(),
                          entry.alg->getQuantLevel(),
                          entry.alg->getChecksum(),
                          entry.alg->getM(),
                          entry.alg->getEfConstruction()};
        return indx;
    }

    // Method to log vector additions with both numeric and string IDs
    void logInsertsAndUpdates(const std::string& index_id,
                              const std::vector<std::pair<idInt, bool>>& numeric_ids) {

        WriteAheadLog* wal = getOrCreateWAL(index_id);

        // Create WAL entries for each vector addition
        std::vector<WriteAheadLog::WALEntry> entries;
        entries.reserve(numeric_ids.size());

        for(size_t i = 0; i < numeric_ids.size(); i++) {
            if(numeric_ids[i].first) {
                entries.push_back({
                        WALOperationType::VECTOR_ADD,
                        numeric_ids[i].first,
                });
            } else {
                entries.push_back({
                        WALOperationType::VECTOR_UPDATE,
                        numeric_ids[i].first,
                });
            }
        }

        // Log the entries
        wal->log(entries);

        // FIX: Don't call saveIndex here to avoid circular lock
        // The calling function (addVectors) already holds operation_mutex
        // and will call save at appropriate time
    }

    // Method to log vector deletions (only numeric IDs needed)
    void logDeletions(const std::string& index_id, const std::vector<idInt>& numeric_ids) {
        WriteAheadLog* wal = getOrCreateWAL(index_id);

        // Create WAL entries for each vector deletion
        std::vector<WriteAheadLog::WALEntry> entries;
        entries.reserve(numeric_ids.size());

        for(size_t i = 0; i < numeric_ids.size(); i++) {
            entries.push_back({
                    WALOperationType::VECTOR_DELETE,
                    numeric_ids[i],
            });
        }

        // Log the entries
        wal->log(entries);

        // FIX: Don't call saveIndex here to avoid circular lock
        // The calling function already holds operation_mutex
        // and will call save at appropriate time
    }
};
