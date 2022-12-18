#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "args_parser.h"
#include "file_utils.h"

namespace {

void GenerateDataset(const std::vector<std::string>& raw_files,
                     const std::string& override_output_prefix,
                     const std::string& classifier_path,
                     bool is_debug) {
    if (!override_output_prefix.empty() && !utils::IsDirectory(override_output_prefix)) {
        throw std::runtime_error(override_output_prefix + " is not a directory.");
    }

    std::vector<std::string> files;

    for (const auto& raw_file: raw_files) {
        if (utils::IsDirectory(raw_file)) {
            utils::ListFiles(raw_file, files);
        } else {
            files.push_back(raw_file);
        }
    }

    // always process images in the same order,
    // it would be easier to visually debug them
    std::sort(files.begin(), files.end());

    cv::CascadeClassifier face_cascade;
    face_cascade.load(classifier_path);

    for (const auto& file_path: files) {
        cv::Mat image = cv::imread(file_path, cv::IMREAD_COLOR);

        if (image.empty()) {
            // not an image, skipping
            continue;
        }

        cv::imshow(file_path, image);

        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(image, faces, 1.05, 6);

        for(size_t i = 0; i < faces.size(); i++) {
            cv::Mat cropped_image = image(faces[i]);

            const auto& report_image_name = 
                utils::GetFileName(file_path) + "_face_" + std::to_string(i)
                + utils::GetFileExtension(file_path);

            std::string output_image_path = report_image_name;

            // if output prefix 
            if (!override_output_prefix.empty()) {
                output_image_path = utils::Join({ 
                    utils::GetAbsolutePath(override_output_prefix), report_image_name });
            }

            cv::imwrite(output_image_path, cropped_image);

            cv::rectangle(image, faces[i], cv::Scalar(0, 0, 255), 6, 1, 0);
        }

        if (is_debug) {
            cv::imshow(file_path, image);
            cv::waitKey(0);
        }
    }

    cv::destroyAllWindows();
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        args::ArgsDict args = args::ParseArgs(argc, argv);

        if (args::DetectArgs(args, 
                { args::FLAG_TITLE_UNSPECIFIED, "-gd" } /* mandatory flags */,
                { "-d", "-o", "-c" } /* optional flags */)) {
            const auto& files = args::GetStringList(args, args::FLAG_TITLE_UNSPECIFIED);
            const auto& output_directory = args::GetString(args, "-o", "" /* default */);
            const auto& classifier_path = args::GetString(args, "-c", "" /* default */);
            const auto& is_debug = args::HasFlag(args, "-d");

            GenerateDataset(files, output_directory, classifier_path, is_debug);
        } else {
            std::cout << "Cannot find suitable command for the given flags." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    return 0;
}
