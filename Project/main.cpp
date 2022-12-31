#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/ml.hpp>
#include <opencv2/face.hpp>
#include <opencv2/opencv.hpp>

#include "bow_recognition_model.h"
#include "hog_recognition_model.h"
#include "dnn_recognition_model.h"

#include "args_parser.h"
#include "annotations_tracker.h"
#include "face_detection_model.h"
#include "face_tracking_model.h"
#include "face_utils.h"
#include "file_utils.h"
#include "labels_resolver.h"
#include "metrics_tracker.h"
#include "metrics_utils.h"
#include "opencv_face_detection_model.h"
#include "strings.h"
#include "video_player.h"
#include "rect.h"

namespace {

void GenerateDataset(const std::vector<std::string>& raw_files,
                     const std::string& override_output_prefix,
                     bool is_debug) {
    if (!override_output_prefix.empty() && !utils::IsDirectory(override_output_prefix)) {
        throw std::runtime_error(override_output_prefix + " is not a directory.");
    }

    std::vector<std::string> files = utils::ListAllFiles(raw_files, { ".png", ".jpg", ".jpeg", ".webp" });
    for (const auto& file_path: files) {
        cv::Mat image = cv::imread(file_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            // not an image, skipping
            continue;
        }

        std::unique_ptr<detection::FaceDetectionModel> face_detection =
                std::make_unique<detection::OpenCVFaceDetectionModel>();

        std::vector<detection::Face> faces = face_detection->extractFaces(image);

        if (is_debug) {
            detection::drawFaces(image, faces);
        }

        for(size_t i = 0; i < faces.size(); i++) {
            auto& face = faces[i].image;

            const auto& report_image_name = 
                utils::GetFileName(file_path) + "_face_" + std::to_string(i)
                + utils::GetFileExtension(file_path);

            std::string output_image_path = report_image_name;

            // if output prefix 
            if (!override_output_prefix.empty()) {
                output_image_path = utils::Join({ 
                    utils::GetAbsolutePath(override_output_prefix), report_image_name });
            }

            cv::imwrite(output_image_path, face);
        }

        if (faces.empty()) {
            std::cout << "Skipping: " << file_path
                      << "no faces detected" << std::endl;
        }

        if (is_debug) {
            cv::imshow(file_path, image);
            cv::waitKey(0);
        }
    }

    cv::destroyAllWindows();
}

void TrainModel(const std::string& dataset_root_folder,
                const std::string& output_model_file,
                const std::string& output_label_file) {
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::LINEAR);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 1e4, 1e-4));

    std::unique_ptr<detection::FaceRecognitionModel> recognizer =
            std::make_unique<detection::DnnRecognitionModel>();

    detection::LabelsResolver labels_resolver;

    std::vector<std::string> directories;
    utils::FlatListDirectories(dataset_root_folder, directories);

    std::vector<int> images_labels;
    std::vector<cv::Mat> images_descriptions;

    for (const auto& directory: directories) {
        std::vector<std::string> paths = utils::SplitPath(directory);
        std::string image_id = paths[paths.size() - 1];

        std::vector<std::string> face_files;
        utils::ListFiles(directory, face_files);

        for (const auto& face_file: face_files) {
            cv::Mat face = cv::imread(face_file);

            if (face.empty()) {
                // not an image, skipping
                continue;
            }

            cv::cvtColor(face, face, cv::COLOR_BGR2GRAY);

            images_labels.push_back(labels_resolver.obtainIdByLabel(image_id));
            images_descriptions.push_back(face);
        }
    }

    recognizer->train(images_descriptions, images_labels);
    recognizer->write(output_model_file);
    labels_resolver.write(output_label_file);
}

void ShowConfig(const std::vector<std::string>& raw_files) {
    std::vector<std::string> files = utils::ListAllFiles(raw_files, { ".mp4" });

    for (const auto& file: files) {
        std::unique_ptr<detection::AnnotationsTracker> annotations_tracker =
                detection::AnnotationsTracker::LoadForVideo(file);
        detection::VideoPlayer video_player(file, 10 /* playback_group_size */);
        cv::Mat frame;

        if(!video_player.isOpened()) {
            throw std::runtime_error("Cannot open " + file);
        }

        std::cout << file << ", frames:" << video_player.framesCount() << std::endl;

        while (video_player.hasNextFrame()) {
            const auto& frame_id = video_player.currentFrame();
            const auto& playback_state = video_player.nextFrame(frame);

            int window_delay = 5;
            bool hasInfo = annotations_tracker->hasInfo(frame_id);

            if (hasInfo) {
                window_delay = 1000;
                const auto& frame_info = annotations_tracker->describeFrame(frame_id);
                detection::drawFaces(frame, frame_info.face_origins(), frame_info.labels());
            }


            cv::imshow(file, frame);
            cv::waitKey(window_delay);
        }
    }

    cv::waitKey(0);
    cv::destroyAllWindows();
}

void ProcessVideoFiles(const std::vector<std::string>& raw_files,
                       const std::string& input_model_file,
                       const std::string& input_label_file,
                       bool test_against_annotations,
                       bool is_debug) {
    std::vector<std::string> files = utils::ListAllFiles(raw_files, { ".mp4" });

    detection::ConfusionMatrix overall_detection_metrics;
    detection::ConfusionMatrix overall_known_recognition_metrics;
    detection::ConfusionMatrix overall_unknown_recognition_metrics;

    detection::FaceTrackingModel face_tracking(detection::FaceTrackingModel::Model::KCF);

    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    std::unique_ptr<detection::FaceRecognitionModel> recognizer =
            std::make_unique<detection::DnnRecognitionModel>();
    recognizer->read(input_model_file);

    detection::LabelsResolver labels_resolver;
    labels_resolver.read(input_label_file);

    for (const auto& file: files) {
        detection::MetricsTracker metrics_tracker(labels_resolver.getLabels());
        std::unique_ptr<detection::AnnotationsTracker> annotations_tracker =
                detection::AnnotationsTracker::LoadForVideo(file);
        detection::VideoPlayer video_player(file, 10 /* playback_group_size */);
        cv::Mat frame;

        if(!video_player.isOpened()) {
            throw std::runtime_error("Cannot open " + file);
        }

        std::cout << file << ", frames:" << video_player.framesCount() << std::endl;

        std::vector<std::string> labels;
        std::vector<detection::Rect> detected_faces_origins;

        while (video_player.hasNextFrame()) {
            const auto& frame_id = video_player.currentFrame();
            bool has_annotations = annotations_tracker->hasInfo(frame_id);

            const auto& playback_state = video_player.nextFrame(frame);

            if (playback_state == detection::VideoPlayer::PlaybackGroupState::STARTING_NEW_GROUP) {
                labels.clear();
                detected_faces_origins.clear();

                std::unique_ptr<detection::FaceDetectionModel> face_detection =
                        std::make_unique<detection::OpenCVFaceDetectionModel>();

                std::vector<detection::Face> faces = face_detection->extractFaces(frame);

                for(size_t i = 0; i < faces.size(); i++) {
                    cv::Mat face = faces[i].image;
                    int id = recognizer->predict(face);
                    labels.push_back(labels_resolver.obtainLabelById(id));
                    detected_faces_origins.push_back(faces[i].origin);
                }

                face_tracking.reset_tracking(frame, labels, detected_faces_origins);
                detection::drawFaces(frame, faces, labels);
            } else {
                detected_faces_origins.clear();

                face_tracking.track(frame, labels, detected_faces_origins);
                detection::drawFaces(frame, detected_faces_origins, labels);
            }

            int window_delay = 5;

            if (test_against_annotations && has_annotations) {
                const auto& frame_info = annotations_tracker->describeFrame(frame_id);
                metrics_tracker.keepTrackOf(frame_info, labels, detected_faces_origins);

                if (is_debug) {
                    window_delay = 1000;
                    cv::putText(frame, std::AsString(frame_id), cv::Point(0, 40), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255, 0, 0), 2, cv::LINE_8);
                    detection::drawFaces(frame, frame_info.face_origins(), frame_info.labels(), cv::Scalar(255, 0, 0));
                }
            }

            cv::imshow(file, frame);
            cv::waitKey(window_delay);
        }

        const auto& detection_metrics = metrics_tracker.overallDetectionMetrics();
        overall_detection_metrics += detection_metrics;

        if (test_against_annotations) {
            const auto &known_recognition_metrics = metrics_tracker.overallKnownRecognitionMetrics();
            overall_known_recognition_metrics += known_recognition_metrics;
            const auto &unknown_recognition_metrics = metrics_tracker.overallUnknownRecognitionMetrics();
            overall_unknown_recognition_metrics += unknown_recognition_metrics;

            detection::PrintConfusionMatrix("detection",
                                            detection_metrics,
                                            detection::PI_TPR | detection::PI_FNR | detection::PI_FPR);
            detection::PrintConfusionMatrix("recognition for known subjects",
                                            known_recognition_metrics,
                                            detection::PI_ACCURACY);
            detection::PrintConfusionMatrix("recognition for unknown subjects",
                                            unknown_recognition_metrics,
                                            detection::PI_TPR | detection::PI_FNR | detection::PI_FPR);
            std::cout << std::endl;
        }
    }

    // if there is the only video there is no reason
    // to show overall statistics
    if (test_against_annotations && files.size() > 1) {
        detection::PrintConfusionMatrix("overall detection",
                                        overall_detection_metrics,
                                        detection::PI_TPR | detection::PI_FNR | detection::PI_FPR);
        detection::PrintConfusionMatrix("overall recognition for known subjects",
                                        overall_known_recognition_metrics,
                                        detection::PI_ACCURACY);
        detection::PrintConfusionMatrix("overall recognition for unknown subjects",
                                        overall_unknown_recognition_metrics,
                                        detection::PI_TPR | detection::PI_FNR | detection::PI_FPR);
    }

    cv::waitKey(0);
    cv::destroyAllWindows();
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        args::ArgsDict args = args::ParseArgs(argc, argv);

        if (args::DetectArgs(args, 
                { args::FLAG_TITLE_UNSPECIFIED, "--dataset" } /* mandatory flags */,
                { "-d", "-o" } /* optional flags */)) {
            const auto& files = args::GetStringList(args, args::FLAG_TITLE_UNSPECIFIED);

            const auto& output_directory = args::GetString(args, "-o", "" /* default */);
            const auto& is_debug = args::HasFlag(args, "-d");

            GenerateDataset(files,
                            output_directory, is_debug);
        } else if (args::DetectArgs(args,
                                    { args::FLAG_TITLE_UNSPECIFIED, "--config" } /* mandatory flags */,
                                    { } /* optional flags */)) {
            const auto& files = args::GetStringList(args, args::FLAG_TITLE_UNSPECIFIED);
            ShowConfig(files);
        } else if (args::DetectArgs(args,
                { args::FLAG_TITLE_UNSPECIFIED, "--train", "-om", "-ol" } /* mandatory flags */,
                { } /* optional flags */)) {
            const auto& dataset_root_folder = args::GetString(args, args::FLAG_TITLE_UNSPECIFIED);
            const auto& output_model_file = args::GetString(args, "-om");
            const auto& output_label_file = args::GetString(args, "-ol");

            TrainModel(dataset_root_folder,
                       output_model_file, output_label_file);
        } else if (args::DetectArgs(args,
                                    { args::FLAG_TITLE_UNSPECIFIED, "--process", "-il", "-im" } /* mandatory flags */,
                                    { "-t", "-d" } /* optional flags */)) {
            const auto& files = args::GetStringList(args, args::FLAG_TITLE_UNSPECIFIED);
            const auto& input_model_file = args::GetString(args, "-im");
            const auto& input_label_file = args::GetString(args, "-il");

            const auto& should_test_against_annotations = args::HasFlag(args, "-t");
            const auto& is_debug = args::HasFlag(args, "-d");

            ProcessVideoFiles(files,
                              input_model_file, input_label_file,
                              should_test_against_annotations,
                              is_debug);
        } else {
            std::cout << "Cannot find suitable command for the given flags." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    return 0;
}
