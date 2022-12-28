#include "hog_recognition_model.h"

namespace detection {

cv::Mat HogRecognitionModel::extractFeatures(cv::Mat image) const {
    cv::resize(image, image, cv::Size(320, 320), cv::INTER_AREA);

    cv::HOGDescriptor hog;
    std::vector<float> descriptors;
    std::vector<cv::Point> locations;

    hog.compute(image, descriptors, cv::Size(32, 32), cv::Size(0, 0), locations);

    cv::Mat out_descriptors(1, descriptors.size(), CV_32FC1);

    for(size_t i = 0; i < descriptors.size(); i++) {
        out_descriptors.at<float>(0, i) = descriptors[i];
    }

    return out_descriptors;
}

HogRecognitionModel::HogRecognitionModel(cv::Ptr<cv::ml::StatModel> model):
        _model(model) {
    // empty on purpose
}

HogRecognitionModel::HogRecognitionModel(const HogRecognitionModel& that) {
    this->_model = that._model;
}

HogRecognitionModel& HogRecognitionModel::operator=(const HogRecognitionModel& that) {
    if (this != &that) {
        this->_model = that._model;
    }

    return *this;
}

void HogRecognitionModel::write(const std::string& file) {
    cv::Ptr<cv::FileStorage> file_storage = cv::makePtr<cv::FileStorage>(file, cv::FileStorage::WRITE);
    _model->write(file_storage, "_model");
}

void HogRecognitionModel::read(const std::string& file) {
    cv::FileStorage file_storage(file, cv::FileStorage::READ);
    _model->read(file_storage["_model"]);
}

void HogRecognitionModel::train(std::vector<cv::Mat>& images,
                                std::vector<int>& images_labels) {
    if (images.size() != images_labels.size()) {
        throw std::runtime_error("Images size is not equal to labels size");
    }

    cv::Mat train_images_descriptors;
    cv::Mat train_images_labels;

    for (size_t i = 0; i < images.size(); i++) {
        cv::Mat image = images[i];
        int label = images_labels[i];
        cv::Mat descriptors = extractFeatures(image);

        train_images_descriptors.push_back(descriptors);
        train_images_labels.push_back(cv::Mat(1, 1, CV_32S, label));
    }

    cv::Ptr<cv::ml::TrainData> train_data =
            cv::ml::TrainData::create(train_images_descriptors, cv::ml::ROW_SAMPLE, train_images_labels);

    _model->train(train_data);
}

int HogRecognitionModel::predict(cv::Mat& image) const {
    cv::Mat descriptors = extractFeatures(image);
    return static_cast<int>(_model->predict(descriptors));
}

} // namespace detection
