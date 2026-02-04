#pragma once
#include <chrono>
#include <unordered_map>
#include <string>
#include <mutex>
#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>

// Debug logging macro
#ifdef ND_DEBUG
#    define LOG_DEBUG(msg)                                                                         \
        do {                                                                                       \
            std::stringstream ss;                                                                  \
            ss << msg;                                                                             \
            std::cerr << "[DEBUG] " << ss.str() << std::endl;                                      \
        } while(0)
#else
#    define LOG_DEBUG(msg)                                                                         \
        do {                                                                                       \
        } while(0)
#endif

// Forward declare the timing macros
#ifdef ND_DEBUG
#    define LOG_TIME(name) FunctionTimer timer##__LINE__(name)
#    define PRINT_LOG_TIME() FunctionTimer::printAndReset()
#else
#    define LOG_TIME(name)                                                                         \
        if constexpr(false) {                                                                      \
        }
#    define PRINT_LOG_TIME()                                                                       \
        if constexpr(false) {                                                                      \
        }
#endif

#ifdef ND_DEBUG
// Only define the class in debug builds
class FunctionTimer {
private:
    struct TimingStats {
        uint64_t total_time{0};  // Total time in microseconds
        uint64_t count{0};       // Number of calls
    };

    static std::unordered_map<std::string, TimingStats> stats;
    static std::mutex mutex;

public:
    const std::string name;
    std::chrono::high_resolution_clock::time_point start;

    FunctionTimer(const std::string& func_name) :
        name(func_name) {
        start = std::chrono::high_resolution_clock::now();
    }

    ~FunctionTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::lock_guard<std::mutex> lock(mutex);
        auto& stat = stats[name];
        stat.total_time += duration;
        stat.count++;
    }

    static void printAndReset() {
        std::lock_guard<std::mutex> lock(mutex);

        std::vector<std::pair<std::string, TimingStats>> sorted_stats;
        sorted_stats.reserve(stats.size());
        for(const auto& pair : stats) {
            sorted_stats.push_back(pair);
        }

        std::sort(sorted_stats.begin(), sorted_stats.end(), [](const auto& a, const auto& b) {
            return a.second.total_time > b.second.total_time;
        });

        std::cerr << "\n=== Function Timings ===\n";
        std::cerr << std::fixed << std::setprecision(3);
        std::cerr << std::setw(30) << std::left << "Function" << std::setw(15) << "Count"
                  << std::setw(15) << "Total(ms)" << std::setw(15) << "Avg(ms)\n";
        std::cerr << std::string(75, '-') << "\n";

        for(const auto& [name, stat] : sorted_stats) {
            double total_ms = stat.total_time / 1000.0;
            double avg_ms = stat.count > 0 ? total_ms / stat.count : 0;

            std::cerr << std::setw(30) << std::left << name << std::setw(15) << stat.count
                      << std::setw(15) << total_ms << std::setw(15) << avg_ms << "\n";
        }
        std::cerr << "=====================\n";
        stats.clear();
    }
};

// Define static members only in debug builds
inline std::unordered_map<std::string, FunctionTimer::TimingStats> FunctionTimer::stats;
inline std::mutex FunctionTimer::mutex;
#endif

#define LOG_STREAM(level, msg)                                                                     \
    do {                                                                                           \
        std::stringstream __log_ss__;                                                              \
        __log_ss__ << msg;                                                                         \
        std::cerr << "[" << level << "] " << __FILE__ << ":" << __LINE__ << " - "                  \
                  << __log_ss__.str() << std::endl;                                                \
    } while(0)

#define LOG_INFO(msg) LOG_STREAM("INFO", msg)
#define LOG_WARN(msg) LOG_STREAM("WARN", msg)
#define LOG_ERROR(msg) LOG_STREAM("ERROR", msg)