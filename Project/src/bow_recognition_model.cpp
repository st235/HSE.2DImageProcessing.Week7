#include "bow_recognition_model.h"

namespace {

cv::Mat MergeDescriptors(const std::vector<cv::Mat>& images_descriptors) {
    cv::Mat descriptors;

    for (const auto& image_descripors: images_descriptors) {
        descriptors.push_back(image_descripors);
    }

    return descriptors;
}

} // namespace

namespace detection {

void BowRecognitionModel::extractFeatures(cv::Mat image,
                                 std::vector<cv::KeyPoint>& out_keypoints,
                                 cv::Mat& out_descriptors) const {
    cv::Ptr<cv::SIFT> sift_transformer = cv::SIFT::create();
    sift_transformer->detectAndCompute(image, cv::Mat(), out_keypoints, out_descriptors);
}

cv::Mat BowRecognitionModel::buildVocabulary(std::vector<cv::Mat>& images,
                                             std::vector<cv::Mat>& out_images_descriptors) const {
    for (auto& image: images) {
        cv::Mat image_descriptors;
        std::vector<cv::KeyPoint> image_keypoints;
        extractFeatures(image, image_keypoints, image_descriptors);

        out_images_descriptors.push_back(image_descriptors);
    }

    auto term_criteria = cv::TermCriteria(
        cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 
        1000 /* iteration_number */, 1e-4);

    cv::Mat out_vocabulary;
    cv::Mat cluster_labels;
    cv::kmeans(MergeDescriptors(out_images_descriptors), _clusters_count, 
               cluster_labels, term_criteria, 5 /* attempts */, 
               cv::KMEANS_PP_CENTERS, out_vocabulary);

    return out_vocabulary;
}

cv::Mat BowRecognitionModel::buildHistogram(const cv::Mat& descriptors,
                                            const cv::Mat& vocabulary) const {
    cv::BFMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors, vocabulary, matches);

    cv::Mat out_histogram = cv::Mat::zeros(1, _clusters_count, CV_32F);

    for (const auto& match: matches) {
        out_histogram.at<float>(0, match.trainIdx) += 1;
    }

    return out_histogram;
}

void BowRecognitionModel::buildHistograms(cv::Mat& vocabulary,
                                          std::vector<cv::Mat> images_descriptors,
                                          std::vector<int> images_labels,
                                          cv::Mat& out_images_histogram,
                                          cv::Mat& out_images_labels) const {
    for (size_t i = 0; i < images_descriptors.size(); i++) {
        auto& image_descriptors = images_descriptors[i];
        auto& image_label = images_labels[i];

        cv::Mat image_histogram = buildHistogram(image_descriptors, vocabulary);

        out_images_histogram.push_back(image_histogram);
        out_images_labels.push_back(cv::Mat(1, 1, CV_32S, image_label));
    }
}

BowRecognitionModel::BowRecognitionModel(size_t clusters_count,
                                         cv::Ptr<cv::ml::StatModel> model):
    _clusters_count(clusters_count),
    _vocabulary(),
    _images_labels(),
    _images_histograms(),
    _model(model) {
    // empty on purpose
}

BowRecognitionModel::BowRecognitionModel(const BowRecognitionModel& that) {
    this->_clusters_count = that._clusters_count;
    this->_vocabulary = that._vocabulary;
    this->_images_labels = that._images_labels;
    this->_images_histograms = that._images_histograms;
    this->_model = that._model;
}

BowRecognitionModel& BowRecognitionModel::operator=(const BowRecognitionModel& that) {
    if (this != &that) {
        this->_clusters_count = that._clusters_count;
        this->_vocabulary = that._vocabulary;
        this->_images_labels = that._images_labels;
        this->_images_histograms = that._images_histograms;
        this->_model = that._model;
    }

    return *this;
}

void BowRecognitionModel::write(const std::string& file) {
    cv::Ptr<cv::FileStorage> file_storage = cv::makePtr<cv::FileStorage>(file, cv::FileStorage::WRITE);

    file_storage->write("_clusters_count", static_cast<int>(_clusters_count));
    file_storage->write("_vocabulary", _vocabulary);
    file_storage->write("_images_labels", _images_labels);
    file_storage->write("_images_histograms", _images_histograms);

    _model->write(file_storage, "_model");
}

void BowRecognitionModel::read(const std::string& file) {
    cv::FileStorage file_storage(file, cv::FileStorage::READ);

    int clusters_count;
    file_storage["_clusters_count"] >> clusters_count;
    _clusters_count = static_cast<uint32_t>(clusters_count);

    file_storage["_vocabulary"] >> _vocabulary;
    file_storage["_images_labels"] >> _images_labels;
    file_storage["_images_histograms"] >> _images_histograms;

    _model->read(file_storage["_model"]);
}

void BowRecognitionModel::train(std::vector<cv::Mat>& images,
                                std::vector<int>& images_labels) {
    if (images.size() != images_labels.size()) {
        throw std::runtime_error("Images size is not equal to labels size");
    }

    std::vector<cv::Mat> images_descriptors;

    _vocabulary = buildVocabulary(images, images_descriptors);
    buildHistograms(_vocabulary, 
                    images_descriptors, images_labels,
                    // internal state
                    _images_histograms, _images_labels);

    cv::Ptr<cv::ml::TrainData> train_data =
        cv::ml::TrainData::create(_images_histograms, cv::ml::ROW_SAMPLE, _images_labels);

    _model->train(train_data);
}

int BowRecognitionModel::predict(cv::Mat& image) const {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    extractFeatures(image, keypoints, descriptors);

    cv::Mat image_histogram = buildHistogram(descriptors, _vocabulary);
    return static_cast<int>(_model->predict(image_histogram));
}

} // namespace detection
