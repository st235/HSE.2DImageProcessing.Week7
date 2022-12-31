#include "dnn_recognition_model.h"

#include <algorithm>
// Render face landmarks (for win overlays)
// dlib::render_face_detections
#include <dlib/image_processing/render_face_detections.h>

#include "dlib_utils.h"

// Gui
#include <dlib/gui_widgets.h>

namespace detection {

void ShowImage(const dlib::array2d<dlib::rgb_pixel>& image) {
    dlib::image_window win;
    win.set_image(image);

    // using of title creates
    // memory leak at least on MacOS
    // win.set_title(title)

    win.wait_until_closed();
}


std::vector<double> DnnRecognitionModel::extractFeatures(const cv::Mat& mat) const {
//    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

    dlib::array2d<dlib::rgb_pixel> image = AsRGBOpenCVMatrix(mat);
    dlib::pyramid_up(image);

    std::vector<dlib::matrix<dlib::rgb_pixel>> face_images;
//    std::vector<dlib::rectangle> face_rectangles = detector(image);
    dlib::rectangle face_rectangle(0, 0, image.nc(), image.nr());

    std::vector<dlib::full_object_detection> faces_landmarks;
    dlib::full_object_detection landmarks = _shape_predictor(image, face_rectangle);
    faces_landmarks.push_back(landmarks);

    dlib::matrix<dlib::rgb_pixel> face_image;
    dlib::extract_image_chip(image, dlib::get_face_chip_details(landmarks, 150, 0.25), face_image);

    face_images.push_back(std::move(face_image));

//    dlib::image_window win;
//    win.set_image(image);
//    win.add_overlay(face_rectangles);
//    win.add_overlay(dlib::render_face_detections(faces_landmarks));
//    win.wait_until_closed();

    face_recognition_dnn_model model = _face_recognition_dnn_model;
    std::vector<dlib::matrix<float, 0, 1>> face_descriptors = model(face_images);

    std::vector<double> vector;
    for (size_t di = 0; di < face_descriptors.size(); di++) {
        auto& face_descriptor = face_descriptors[di];

        for (auto iterator = face_descriptor.begin(); iterator < face_descriptor.end(); iterator++) {
            vector.push_back(static_cast<double>(*iterator));
        }
    }

    return vector;
}

DnnRecognitionModel::DnnRecognitionModel(const std::string& landmarks_model_file,
                                         const std::string& dnn_model_file,
                                         cv::Ptr<cv::ml::StatModel> model):
    _shape_predictor(),
    _face_recognition_dnn_model(),
    _model(model) {
    dlib::deserialize(landmarks_model_file) >> _shape_predictor;
    dlib::deserialize(dnn_model_file) >> _face_recognition_dnn_model;
}

DnnRecognitionModel::DnnRecognitionModel(const DnnRecognitionModel& that) = default;

DnnRecognitionModel& DnnRecognitionModel::operator=(const DnnRecognitionModel& that) = default;

void DnnRecognitionModel::write(const std::string& file) {
    cv::Ptr<cv::FileStorage> file_storage = cv::makePtr<cv::FileStorage>(file, cv::FileStorage::WRITE);
    _model->write(file_storage, "_model");
}

void DnnRecognitionModel::read(const std::string& file) {
    cv::FileStorage file_storage(file, cv::FileStorage::READ);
    _model->read(file_storage["_model"]);
}

void DnnRecognitionModel::train(std::vector<cv::Mat>& images,
                                std::vector<int>& images_labels) {
    if (images.size() != images_labels.size()) {
        throw std::runtime_error("Images size is not equal to labels size");
    }

    cv::Mat data;
    cv::Mat train_labels;

    for (size_t i = 0; i < images.size(); i++) {
        cv::Mat image = images[i];
        int label = images_labels[i];

        std::vector<double> features = extractFeatures(image);

        cv::Mat row = cv::Mat::zeros(1, 128, CV_32F);

        // we were not able to detect anything in this
        // frame
        if (features.size() != 128) {
            continue;
        }

        for (size_t i = 0; i < features.size(); i++) {
            row.at<float>(0, i) = static_cast<float>(features[i]);
        }

        data.push_back(row);
        train_labels.push_back(cv::Mat(1, 1, CV_32S, label));
    }

    cv::Ptr<cv::ml::TrainData> train_data =
            cv::ml::TrainData::create(data, cv::ml::ROW_SAMPLE, train_labels);

    _model->train(train_data);
}

int DnnRecognitionModel::predict(cv::Mat& image) const {
    cv::Mat train_data;

    std::vector<double> features = extractFeatures(image);

    cv::Mat data = cv::Mat::zeros(1, 128, CV_32F);

    if (features.size() == 128) {
        for (size_t i = 0; i < 128; i++) {
            data.at<float>(0, i) = static_cast<float>(features[i]);
        }
    }

    train_data.push_back(data);
    return static_cast<int>(_model->predict(train_data));
}

} // namespace detection
