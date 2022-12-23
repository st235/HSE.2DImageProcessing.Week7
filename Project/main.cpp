#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/face.hpp>
#include <opencv2/opencv.hpp>

#include "args_parser.h"
#include "file_utils.h"
#include "bag_of_words.h"

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

void RecognizeFaceOnVideo(const std::string& video_file,
                          const std::string& classifier_path,
                          detection::BagOfWords& bag_of_words,
                          bool is_debug) {
    cv::VideoCapture capture(video_file);
    cv::Mat frame;

    cv::CascadeClassifier face_cascade;
    face_cascade.load(classifier_path);

    if(!capture.isOpened()) {
        throw std::runtime_error("Error when reading steam_avi");
    }

    while (true) {
        capture >> frame;
        if(frame.empty()) {
            break;
        }

        cv::Mat grey_scale;
        cv::cvtColor(frame, grey_scale, cv::COLOR_BGR2GRAY);
        grey_scale.convertTo(grey_scale, CV_8U);

        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(grey_scale, faces, 1.05, 6);

        for(size_t i = 0; i < faces.size(); i++) {
            cv::Mat cropped_image = grey_scale(faces[i]);
            int id = bag_of_words.predict(cropped_image);
            std::cout << "predicted id: " << id << std::endl;

            if (id == 0) {
                cv::rectangle(frame, faces[i], cv::Scalar(0, 255, 0), 6, 1, 0);
            } else {
                cv::rectangle(frame, faces[i], cv::Scalar(0, 0, 255), 6, 1, 0);
            }
        }

        cv::imshow("w", frame);
        cv::waitKey(20);
    }
    cv::waitKey(0);
}

void TrainModels(const std::string& dataset_root_folder,
                 const std::string& video_file,
                 const std::string& classifier_path,
                 bool is_debug) {
    detection::BagOfWords bag_of_words;

    std::vector<std::string> directories;
    utils::FlatListDirectories(dataset_root_folder, directories);

    std::vector<int> images_labels;
    std::vector<cv::Mat> images_descriptions;

    size_t image_id = 0;
    for (const auto& directory: directories) {
        std::vector<std::string> paths = utils::SplitPath(directory);
        std::string id = paths[paths.size() - 1];

        std::vector<std::string> face_files;
        utils::ListFiles(directory, face_files);

        for (const auto& face_file: face_files) {
            cv::Mat face = cv::imread(face_file);

            if (face.empty()) {
                // not an image, skipping
                continue;
            }

            images_labels.push_back(image_id);
            images_descriptions.push_back(face);

            std::cout << image_id << ", " << id << std::endl;
        }

        image_id += 1;
    }

    bag_of_words.fit(images_descriptions, images_labels);

    RecognizeFaceOnVideo(video_file, classifier_path, bag_of_words, is_debug);
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
        } else if (args::DetectArgs(args, 
                { args::FLAG_TITLE_UNSPECIFIED, "-tr" } /* mandatory flags */,
                { "-d", "-v", "-c" } /* optional flags */)) {
            const auto& dataset_root_folder = args::GetString(args, args::FLAG_TITLE_UNSPECIFIED);
            const auto& video_file = args::GetString(args, "-v", "" /* default */);
            const auto& classifier_path = args::GetString(args, "-c", "" /* default */);
            const auto& is_debug = args::HasFlag(args, "-d");

            TrainModels(dataset_root_folder, video_file, classifier_path, is_debug);
        } else {
            std::cout << "Cannot find suitable command for the given flags." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    return 0;
}
