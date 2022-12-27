#ifndef BAG_OF_WORDS_H
#define BAG_OF_WORDS_H

#include <vector>

#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>

#include "face_recognition_model.h"

namespace detection {

constexpr size_t DEFAULT_CLUSTERS_SIZE = 800;

/**
 * Recognition model based on Bag of Words algorithm.
 * This is a meta recognition algorithm. The steps are:
 * 1. Extract features: using SIFT, KAZE
 * 2. Build vocabulary: using any clustering algorithm,
 * and considering clusters centers as a word from vocabulary
 * 3. Build histograms: frequency of words from vocabulary
 * for our images will help to identify unique features for
 * the given set of images
 * 4. Train some underlying model based on histograms
 * as feature vectors. Default model is KNN.
*/
class BowRecognitionModel: public FaceRecognitionModel {
private:
  size_t _clusters_count;

  cv::Mat _vocabulary;
  cv::Mat _images_labels;
  cv::Mat _images_histograms;

  cv::Ptr<cv::ml::StatModel> _model;

  void extractFeatures(cv::Mat image,
                       std::vector<cv::KeyPoint>& out_keypoints,
                       cv::Mat& out_descriptors) const;

  cv::Mat buildVocabulary(std::vector<cv::Mat>& images,
                          std::vector<cv::Mat>& out_images_descriptors) const;

  cv::Mat buildHistogram(const cv::Mat& descriptors,
                         const cv::Mat& vocabulary) const;

  void buildHistograms(cv::Mat& vocabulary,
                       std::vector<cv::Mat> images_descriptors,
                       std::vector<int> images_labels,
                       cv::Mat& out_images_histogram,
                       cv::Mat& out_images_labels) const;

public:
  explicit BowRecognitionModel(size_t clusters_count = DEFAULT_CLUSTERS_SIZE,
                               cv::Ptr<cv::ml::StatModel> model = cv::ml::KNearest::create());
    BowRecognitionModel(const BowRecognitionModel& that);
    BowRecognitionModel& operator=(const BowRecognitionModel& that);

  void write(const std::string& file) override;
  void read(const std::string& file) override;

  void train(std::vector<cv::Mat>& images,
             std::vector<int>& images_labels) override;
  int predict(cv::Mat& image) const override;

  ~BowRecognitionModel() = default;
};

} // namespace detection

#endif // BAG_OF_WORDS_H
