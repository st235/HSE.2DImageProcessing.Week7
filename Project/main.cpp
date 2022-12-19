#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/face.hpp>
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

void TrainModel(const std::string& id,
                const std::vector<std::string>& faces_files,
                const std::string& model_output,
                bool is_debug) {
    cv::Ptr<cv::face::FaceRecognizer> recognizer = cv::face::LBPHFaceRecognizer::create();

    std::vector<cv::Mat> images;
    std::vector<int> labels;

    for (const auto& face_file: faces_files) {
        cv::Mat image = cv::imread(face_file, cv::IMREAD_COLOR);

        if (image.empty()) {
            continue;
        }

        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
        image.convertTo(image, CV_8U);

        images.push_back(image);
        labels.push_back(1);
    }

    const std::string output_model_path = utils::Join({ 
        utils::GetAbsolutePath(model_output), id + ".yml" });

    recognizer->train(images, labels);
    recognizer->write(output_model_path);
}

void TrainModels(const std::string& raw_file,
                 const std::string& model_output,
                 bool is_debug) {
    std::vector<std::string> directories;
    utils::FlatListDirectories(raw_file, directories);

    for (const auto& directory: directories) {
        std::vector<std::string> paths = utils::SplitPath(directory);
        std::string id = paths[paths.size() - 1];

        std::vector<std::string> face_files;
        utils::ListFiles(directory, face_files);

        TrainModel(id, face_files, model_output, is_debug);
    }
}

void RecognizeFaceOnVideo(const std::string& video_file,
                          const std::string& classifier_path,
                          const std::string& model,
                          bool is_debug) {
    cv::VideoCapture capture(video_file);
    cv::Mat frame;

    cv::Ptr<cv::face::FaceRecognizer> recognizer = cv::face::LBPHFaceRecognizer::create();
    recognizer->read(model);

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
            int id = recognizer->predict(cropped_image);

            if (id == 1) {
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

void RecognizeFaceOnImage(const std::vector<std::string>& raw_files,
                          const std::string& classifier_path,
                          const std::string& model,
                          bool is_debug) {
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

    cv::Ptr<cv::face::FaceRecognizer> recognizer = cv::face::LBPHFaceRecognizer::create();
    recognizer->read(model);

    cv::CascadeClassifier face_cascade;
    face_cascade.load(classifier_path);

    for (const auto& file_path: files) {
        cv::Mat image = cv::imread(file_path, cv::IMREAD_COLOR);

        if (image.empty()) {
            // not an image, skipping
            continue;
        }

        cv::Mat grey_scale;
        cv::cvtColor(image, grey_scale, cv::COLOR_BGR2GRAY);
        grey_scale.convertTo(grey_scale, CV_8U);

        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(grey_scale, faces, 1.05, 6);

        for(size_t i = 0; i < faces.size(); i++) {
            cv::Mat cropped_image = grey_scale(faces[i]);
            int id = recognizer->predict(cropped_image);

            if (id == 0) {
                cv::rectangle(image, faces[i], cv::Scalar(0, 255, 0), 6, 1, 0);
            } else {
                cv::rectangle(image, faces[i], cv::Scalar(0, 0, 255), 6, 1, 0);
            }
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
        } else if (args::DetectArgs(args, 
                { args::FLAG_TITLE_UNSPECIFIED, "-tr" } /* mandatory flags */,
                { "-d", "-o" } /* optional flags */)) {
            const auto& folder = args::GetString(args, args::FLAG_TITLE_UNSPECIFIED);
            const auto& output_directory = args::GetString(args, "-o", "" /* default */);
            const auto& is_debug = args::HasFlag(args, "-d");

            TrainModels(folder, output_directory, is_debug);
        } else if (args::DetectArgs(args, 
                { args::FLAG_TITLE_UNSPECIFIED, "-ri" } /* mandatory flags */,
                { "-c", "-m", "-d" } /* optional flags */)) {
            const auto& folder = args::GetStringList(args, args::FLAG_TITLE_UNSPECIFIED);
            const auto& classifier_path = args::GetString(args, "-c", "" /* default */);
            const auto& model_path = args::GetString(args, "-m", "" /* default */);
            const auto& is_debug = args::HasFlag(args, "-d");

            RecognizeFaceOnImage(folder, classifier_path, model_path, is_debug);
        } else if (args::DetectArgs(args, 
                { args::FLAG_TITLE_UNSPECIFIED, "-rv" } /* mandatory flags */,
                { "-c", "-m", "-d" } /* optional flags */)) {
            const auto& video_file = args::GetString(args, args::FLAG_TITLE_UNSPECIFIED);
            const auto& classifier_path = args::GetString(args, "-c", "" /* default */);
            const auto& model_path = args::GetString(args, "-m", "" /* default */);
            const auto& is_debug = args::HasFlag(args, "-d");

            RecognizeFaceOnVideo(video_file, classifier_path, model_path, is_debug);
        } else {
            std::cout << "Cannot find suitable command for the given flags." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    return 0;
}
