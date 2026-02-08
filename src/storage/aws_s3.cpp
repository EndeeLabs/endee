#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/s3/model/CreateMultipartUploadRequest.h>
#include <aws/s3/model/UploadPartRequest.h>
#include <aws/s3/model/CompleteMultipartUploadRequest.h>
#include <aws/s3/model/AbortMultipartUploadRequest.h>
#include <aws/s3/model/CompletedMultipartUpload.h>
#include <aws/s3/model/CompletedPart.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <ctime>

#include "aws_s3.hpp"

#define S3_BUCKET "shaleen-bucket"
#define S3_REGION "us-east-1"

const size_t MULTIPART_THRESHOLD = 100 * 1024 * 1024;  // 100MB
const size_t PART_SIZE = 64 * 1024 * 1024;             // 64MB per part

namespace fs = std::filesystem;

std::string formatSize(uintmax_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024 && unit < 4) {
        size /= 1024;
        unit++;
    }
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << units[unit];
    return oss.str();
}

double getElapsedMs(const timespec& start, const timespec& end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 + 
           (end.tv_nsec - start.tv_nsec) / 1000000.0;
}


bool uploadSmallFile(Aws::S3::S3Client& s3,
                     const Aws::String& bucket,
                     const std::string& localPath,
                     const Aws::String& s3Key) {
    
    Aws::S3::Model::PutObjectRequest request;
    request.SetBucket(bucket);
    request.SetKey(s3Key);
    request.SetChecksumAlgorithm(Aws::S3::Model::ChecksumAlgorithm::NOT_SET);

    auto input_data = Aws::MakeShared<Aws::FStream>(
        "PutObjectInputStream",
        localPath.c_str(),
        std::ios_base::in);

    if (!input_data->good()) {
        std::cerr << "Failed to open: " << localPath << std::endl;
        return false;
    }

    request.SetBody(input_data);
    auto outcome = s3.PutObject(request);
    
    return outcome.IsSuccess();
}

bool uploadLargeFile(Aws::S3::S3Client& s3,
                     const Aws::String& bucket,
                     const std::string& localPath,
                     const Aws::String& s3Key,
                     uintmax_t fileSize) {
    
    // 1. Initiate multipart upload
    Aws::S3::Model::CreateMultipartUploadRequest createRequest;
    createRequest.SetBucket(bucket);
    createRequest.SetKey(s3Key);

    auto createOutcome = s3.CreateMultipartUpload(createRequest);
    if (!createOutcome.IsSuccess()) {
        std::cerr << "Failed to initiate multipart upload: "
                  << createOutcome.GetError().GetMessage() << std::endl;
        return false;
    }

    Aws::String uploadId = createOutcome.GetResult().GetUploadId();
    Aws::Vector<Aws::S3::Model::CompletedPart> completedParts;

    std::ifstream file(localPath);
    if (!file.is_open()) {
        std::cerr << "Failed to open: " << localPath << std::endl;
        return false;
    }

    // 2. Upload parts
    int partNumber = 1;
    size_t uploadedBytes = 0;
    std::vector<char> buffer(PART_SIZE);

    while (uploadedBytes < fileSize) {
        size_t bytesToRead = std::min(PART_SIZE, static_cast<size_t>(fileSize - uploadedBytes));
        file.read(buffer.data(), bytesToRead);
        size_t bytesRead = file.gcount();

        auto partStream = Aws::MakeShared<Aws::StringStream>("UploadPartStream");
        partStream->write(buffer.data(), bytesRead);

        Aws::S3::Model::UploadPartRequest uploadPartRequest;
        uploadPartRequest.SetBucket(bucket);
        uploadPartRequest.SetKey(s3Key);
        uploadPartRequest.SetPartNumber(partNumber);
        uploadPartRequest.SetUploadId(uploadId);
        uploadPartRequest.SetBody(partStream);
        uploadPartRequest.SetContentLength(bytesRead);

        auto uploadPartOutcome = s3.UploadPart(uploadPartRequest);
        if (!uploadPartOutcome.IsSuccess()) {
            std::cerr << "Failed to upload part " << partNumber << ": "
                      << uploadPartOutcome.GetError().GetMessage() << std::endl;
            
            // Abort the multipart upload
            Aws::S3::Model::AbortMultipartUploadRequest abortRequest;
            abortRequest.SetBucket(bucket);
            abortRequest.SetKey(s3Key);
            abortRequest.SetUploadId(uploadId);
            s3.AbortMultipartUpload(abortRequest);
            return false;
        }

        Aws::S3::Model::CompletedPart completedPart;
        completedPart.SetPartNumber(partNumber);
        completedPart.SetETag(uploadPartOutcome.GetResult().GetETag());
        completedParts.push_back(completedPart);

        uploadedBytes += bytesRead;
        partNumber++;
        
        // Progress indicator
        std::cout << "\r  Uploading: " << (uploadedBytes * 100 / fileSize) << "%" << std::flush;
    }
    std::cout << "\r                    \r" << std::flush;  // Clear progress line

    file.close();

    // 3. Complete multipart upload
    Aws::S3::Model::CompletedMultipartUpload completedUpload;
    completedUpload.SetParts(completedParts);

    Aws::S3::Model::CompleteMultipartUploadRequest completeRequest;
    completeRequest.SetBucket(bucket);
    completeRequest.SetKey(s3Key);
    completeRequest.SetUploadId(uploadId);
    completeRequest.SetMultipartUpload(completedUpload);

    auto completeOutcome = s3.CompleteMultipartUpload(completeRequest);
    if (!completeOutcome.IsSuccess()) {
        std::cerr << "Failed to complete multipart upload: "
                  << completeOutcome.GetError().GetMessage() << std::endl;
        return false;
    }

    return true;
}

bool uploadFile(Aws::S3::S3Client& s3, 
                const Aws::String& bucket,
                const std::string& localPath, 
                const Aws::String& s3Key) {
    
    uintmax_t fileSize = fs::file_size(localPath);
    
    timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    bool success;
    if (fileSize < MULTIPART_THRESHOLD) {
        success = uploadSmallFile(s3, bucket, localPath, s3Key);
    } else {
        success = uploadLargeFile(s3, bucket, localPath, s3Key, fileSize);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsedMs = getElapsedMs(start, end);

    if (!success) {
        std::cerr << "Failed to upload: " << localPath << std::endl;
        return false;
    }

    std::cout << std::left << std::setw(40) << fs::path(localPath).filename().string()
              << std::right << std::setw(12) << formatSize(fileSize)
              << std::setw(12) << std::fixed << std::setprecision(2) << elapsedMs << "ms"
              << std::endl;

    return true;
}


/**
 * Uploads a single tarball to S3 bucket.
 * @param localPath - Path to the tarball file (e.g., "/home/user/backup.tar.gz")
 * @return true if upload succeeded, false otherwise
 */
bool upload_to_s3(const std::string& localPath) {

    bool ret = false;

    // Check if file exists
    if (!fs::exists(localPath)) {
        std::cerr << "File not found: " << localPath << std::endl;
        return ret;
    }

    Aws::SDKOptions options;
    Aws::InitAPI(options);

    {
        const Aws::String bucket = S3_BUCKET;

        Aws::Client::ClientConfiguration config;
        config.region = S3_REGION;

        Aws::S3::S3Client s3(config);

        // Use filename as S3 key (e.g., "backup.tar.gz")
        // Or you can add a prefix: "archives/" + filename
        std::string s3Key = "archives/" + fs::path(localPath).filename().string();

        std::cout << "Uploading: " << localPath << " -> s3://" << bucket << "/" << s3Key << std::endl;

        ret = uploadFile(s3, bucket, localPath, s3Key.c_str());

        if (ret) {
            std::cout << "Upload successful!" << std::endl;
        } else {
            std::cerr << "Upload failed!" << std::endl;
        }
    }

    Aws::ShutdownAPI(options);

    return ret;
}