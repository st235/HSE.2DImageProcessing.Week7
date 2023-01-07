#include "hog_recognition_model.h"

namespace detection {

cv::Mat HogRecognitionModel::extractFeatures(cv::Mat image) const {
    cv::resize(image, image, cv::Size(128, 128), cv::INTER_AREA);

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

HogRecognitionModel::HogRecognitionModel(uint32_t max_neighbours,
                                         double max_neighbours_distance):
    _max_neighbours(max_neighbours),
    _max_neighbours_distance(max_neighbours_distance),
    _knearest(cv::ml::KNearest::create()) {
    _knearest->setDefaultK(_max_neighbours);
    _knearest->setIsClassifier(true);
}

HogRecognitionModel::HogRecognitionModel(const HogRecognitionModel& that):
    _max_neighbours(that._max_neighbours),
    _max_neighbours_distance(that._max_neighbours_distance),
    _knearest(that._knearest) {
    // empty on purpose
}

HogRecognitionModel& HogRecognitionModel::operator=(const HogRecognitionModel& that) {
    if (this != &that) {
        this->_max_neighbours = that._max_neighbours;
        this->_max_neighbours_distance = that._max_neighbours_distance;
        this->_knearest = that._knearest;
    }

    return *this;
}

void HogRecognitionModel::write(const std::string& file) {
    cv::Ptr<cv::FileStorage> file_storage = cv::makePtr<cv::FileStorage>(file, cv::FileStorage::WRITE);

    file_storage->write("_max_neighbours", static_cast<int>(_max_neighbours));
    file_storage->write("_max_neighbours_distance", _max_neighbours_distance);

    _knearest->write(file_storage, "_knearest");
}

void HogRecognitionModel::read(const std::string& file) {
    cv::FileStorage file_storage(file, cv::FileStorage::READ);

    int max_neighbours;
    file_storage["_max_neighbours"] >> max_neighbours;
    _max_neighbours = static_cast<uint32_t>(max_neighbours);

    file_storage["_max_neighbours_distance"] >> _max_neighbours_distance;

    _knearest->read(file_storage["_knearest"]);
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

    _knearest->train(train_data);
}

int HogRecognitionModel::predict(cv::Mat& image) const {
    cv::Mat train_data = extractFeatures(image);

    cv::Mat out_results,
            out_neighbors,
            out_distances;
    float classification_result = _knearest->findNearest(train_data, _knearest->getDefaultK(), out_results, out_neighbors, out_distances);

    double distance = 0;
    for (size_t i = 0; i < out_distances.cols; i++) {
        distance += out_distances.at<float>(0, i);
    }
    // distance is on scale from [0, 1]
    distance /= out_distances.cols;

    if (distance > _max_neighbours_distance) {
        return FaceRecognitionModel::LABEL_UNKNOWN;
    }

    return static_cast<int>(classification_result);
}

} // namespace detection
