#pragma once
#include <string>
#include <filesystem>
#include <fstream>
#include <archive.h>
#include <archive_entry.h>

namespace ndd {

    class ArchiveUtils {
    public:
        // Create tar.gz archive from a directory
        static bool createTarGz(const std::filesystem::path& source_dir,
                                const std::filesystem::path& archive_path,
                                std::string& error_msg) {
            struct archive* a = archive_write_new();
            archive_write_add_filter_gzip(a);
            archive_write_set_format_pax_restricted(a);

            if(archive_write_open_filename(a, archive_path.string().c_str()) != ARCHIVE_OK) {
                error_msg = archive_error_string(a);
                archive_write_free(a);
                return false;
            }

            // Recursively add all files
            for(const auto& entry : std::filesystem::recursive_directory_iterator(source_dir)) {
                if(entry.is_regular_file()) {
                    struct archive_entry* e = archive_entry_new();

                    // Calculate relative path for archive
                    std::filesystem::path rel_path =
                            std::filesystem::relative(entry.path(), source_dir.parent_path());
                    archive_entry_set_pathname(e, rel_path.string().c_str());
                    archive_entry_set_size(e, std::filesystem::file_size(entry.path()));
                    archive_entry_set_filetype(e, AE_IFREG);
                    archive_entry_set_perm(e, 0644);

                    if(archive_write_header(a, e) != ARCHIVE_OK) {
                        error_msg = archive_error_string(a);
                        archive_entry_free(e);
                        archive_write_free(a);
                        return false;
                    }

                    // Write file content
                    std::ifstream file(entry.path(), std::ios::binary);
                    char buffer[8192];
                    while(file.read(buffer, sizeof(buffer)) || file.gcount() > 0) {
                        archive_write_data(a, buffer, file.gcount());
                    }
                    file.close();
                    archive_entry_free(e);
                }
            }

            archive_write_close(a);
            archive_write_free(a);
            return true;
        }

        // Extract tar.gz archive to a directory
        static bool extractTarGz(const std::filesystem::path& archive_path,
                                 const std::filesystem::path& dest_dir,
                                 std::string& error_msg) {
            struct archive* a = archive_read_new();
            struct archive* ext = archive_write_disk_new();
            struct archive_entry* entry;

            archive_read_support_format_all(a);
            archive_read_support_filter_all(a);
            archive_write_disk_set_options(ext, ARCHIVE_EXTRACT_TIME | ARCHIVE_EXTRACT_PERM);
            archive_write_disk_set_standard_lookup(ext);

            if(archive_read_open_filename(a, archive_path.string().c_str(), 10240) != ARCHIVE_OK) {
                error_msg = archive_error_string(a);
                archive_read_free(a);
                archive_write_free(ext);
                return false;
            }

            while(archive_read_next_header(a, &entry) == ARCHIVE_OK) {
                std::filesystem::path full_path = dest_dir / archive_entry_pathname(entry);
                archive_entry_set_pathname(entry, full_path.string().c_str());

                if(archive_write_header(ext, entry) == ARCHIVE_OK) {
                    const void* buff;
                    size_t size;
                    la_int64_t offset;

                    while(archive_read_data_block(a, &buff, &size, &offset) == ARCHIVE_OK) {
                        archive_write_data_block(ext, buff, size, offset);
                    }
                }
                archive_write_finish_entry(ext);
            }

            archive_read_close(a);
            archive_read_free(a);
            archive_write_close(ext);
            archive_write_free(ext);
            return true;
        }
    };

}  // namespace ndd
