#include "bag_of_words.h"

#include <iostream>

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

void BagOfWords::extractFeatures(cv::Mat image,
                                 std::vector<cv::KeyPoint>& out_keypoints,
                                 cv::Mat& out_descriptors) {
    cv::Ptr<cv::SIFT> sift_transformer = cv::SIFT::create();
    sift_transformer->detectAndCompute(image, cv::Mat(), out_keypoints, out_descriptors);
}

cv::Mat BagOfWords::buildVocabulary(std::vector<cv::Mat>& images,
                                    std::vector<cv::Mat>& out_images_descriptors) {
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

cv::Mat BagOfWords::buildHistogram(cv::Mat& descriptors,
                                   cv::Mat& vocabulary) {
    cv::BFMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors, vocabulary, matches);

    cv::Mat out_histogram = cv::Mat::zeros(1, _clusters_count, CV_32F);

    for (const auto& match: matches) {
        out_histogram.at<float>(0, match.trainIdx) += 1;
    }

    return out_histogram;
}

void BagOfWords::buildHistograms(cv::Mat& vocabulary,
                                 std::vector<cv::Mat> images_descriptors,
                                 std::vector<int> images_labels,
                                 cv::Mat& out_images_histogram,
                                 cv::Mat& out_images_labels) {
    for (size_t i = 0; i < images_descriptors.size(); i++) {
        auto& image_descriptors = images_descriptors[i];
        auto& image_label = images_labels[i];

        cv::Mat image_histogram = buildHistogram(image_descriptors, vocabulary);

        out_images_histogram.push_back(image_histogram);
        out_images_labels.push_back(cv::Mat(1, 1, CV_32F, static_cast<float>(image_label)));
    }
}

BagOfWords::BagOfWords(size_t clusters_count,
                       cv::Ptr<cv::ml::StatModel> model):
    _clusters_count(clusters_count),
    _vocabulary(),
    _images_labels(),
    _images_histograms(),
    _model(model) {
    // empty on purpose
}

BagOfWords::BagOfWords(const BagOfWords& that) {
    this->_clusters_count = that._clusters_count;
    this->_vocabulary = that._vocabulary;
    this->_images_labels = that._images_labels;
    this->_images_histograms = that._images_histograms;
    this->_model = that._model;
}

BagOfWords& BagOfWords::operator=(const BagOfWords& that) {
    if (this != &that) {
        this->_clusters_count = that._clusters_count;
        this->_vocabulary = that._vocabulary;
        this->_images_labels = that._images_labels;
        this->_images_histograms = that._images_histograms;
        this->_model = that._model;
    }

    return *this;
}

void BagOfWords::fit(std::vector<cv::Mat>& images,
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

int BagOfWords::predict(cv::Mat& image) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    extractFeatures(image, keypoints, descriptors);

    cv::Mat image_histogram = buildHistogram(descriptors, _vocabulary);

    return static_cast<int>(_model->predict(image_histogram));
}

} // namespace detection
