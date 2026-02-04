// wal.hpp
#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <mutex>
#include <atomic>
#include <filesystem>
#include <cstring>
#include <cerrno>
#include "../core/types.hpp"

enum class WALOperationType : uint8_t { VECTOR_ADD = 1, VECTOR_DELETE = 2, VECTOR_UPDATE = 3 };

class WriteAheadLog {
private:
    std::string log_path_;
    std::ofstream log_file_;
    std::mutex file_mutex_;
    std::atomic<bool> enabled_{true};
    std::atomic<size_t> entry_count_{0};

public:
    // WAL entry structure for operations
    struct WALEntry {
        WALOperationType op_type;
        ndd::idInt numeric_id;
    };

    WriteAheadLog(const std::string& index_dir) {
        log_path_ = index_dir + "/wal.bin";
        // Open in append mode
        log_file_.open(log_path_, std::ios::binary | std::ios::app);
        if(!log_file_) {
            std::string err_string;
            err_string = "Failed to open WAL file: " + log_path_
                         + " errno: " + std::to_string(errno) + " errcode: " + std::strerror(errno);

            LOG_ERROR(err_string);
            throw std::runtime_error(err_string);
        }
        // Check if WAL has existing entries (no need to count them)
        std::error_code ec;
        auto file_size = std::filesystem::file_size(log_path_, ec);
        if(!ec && file_size > 0) {
            // Set entry_count_ to 1 to indicate there are entries needing recovery
            // The exact count doesn't matter - we just need to know recovery is needed
            entry_count_ = 1;
        }
    }

    ~WriteAheadLog() { log_file_.close(); }

    // Check if WAL has entries that need recovery
    bool hasEntries() const { return entry_count_ > 0; }
    // Get the number of entries added since last clear
    size_t getEntryCount() const { return entry_count_.load(); }
    // Unified log function that handles a vector of entries
    void log(const std::vector<WALEntry>& entries) {
        if(!enabled_ || entries.empty()) {
            return;
        }

        std::lock_guard<std::mutex> lock(file_mutex_);

        for(const auto& entry : entries) {
            // Write operation type
            uint8_t op = static_cast<uint8_t>(entry.op_type);
            log_file_.write(reinterpret_cast<const char*>(&op), sizeof(op));

            // Write numeric ID (always included)
            log_file_.write(reinterpret_cast<const char*>(&entry.numeric_id),
                            sizeof(entry.numeric_id));
        }

        log_file_.flush();
        entry_count_ += entries.size();
    }

    // Convenience method for logging a single entry
    void log(const WALEntry& entry) { log(std::vector<WALEntry>{entry}); }

    // Read all entries from the WAL file
    std::vector<WALEntry> readEntries() {
        std::vector<WALEntry> entries;

        std::ifstream infile(log_path_, std::ios::binary);
        if(!infile) {
            return entries;  // Return empty if file can't be opened
        }

        while(true) {
            uint8_t op;
            ndd::idInt numeric_id;

            // Read operation type
            infile.read(reinterpret_cast<char*>(&op), sizeof(op));
            if(infile.eof()) {
                break;
            }

            // Read numeric ID
            infile.read(reinterpret_cast<char*>(&numeric_id), sizeof(numeric_id));
            if(infile.eof()) {
                break;
            }

            std::string string_id;
            // Only VECTOR_ADD has string_id
            if(static_cast<WALOperationType>(op) == WALOperationType::VECTOR_ADD) {
                // Read string length
                size_t str_len;
                infile.read(reinterpret_cast<char*>(&str_len), sizeof(str_len));
                if(infile.eof()) {
                    break;
                }

                // Read string content
                string_id.resize(str_len);
                infile.read(&string_id[0], str_len);
                if(infile.eof()) {
                    break;
                }
            }

            entries.push_back({static_cast<WALOperationType>(op), numeric_id});
        }

        return entries;
    }
    // Clear the WAL file
    void clear() {
        log_file_.close();
        std::filesystem::remove(log_path_);
        log_file_.open(log_path_, std::ios::binary | std::ios::app);
        entry_count_ = 0;
    }

    void disable() { enabled_ = false; }

    void enable() { enabled_ = true; }
};