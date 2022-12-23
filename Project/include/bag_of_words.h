#ifndef BAG_OF_WORDS_H
#define BAG_OF_WORDS_H

#include <vector>

#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>

namespace detection {

constexpr size_t DEFAULT_CLUSTERS_SIZE = 800;

/**
 * Algorithm steps:
 * 1. Extract features: using SIFT, KAZE
 * 2. Build vocabulary: using any clustering algorithm,
 * and considering clusters centers as a word from vocabulary
 * 3. Build historgrams: freqency of words from vocabulary
 * for our images will help to identify unique features for
 * the given set of images
*/
class BagOfWords {
private:
  size_t _clusters_count;

  cv::Mat _vocabulary;
  cv::Mat _images_labels;
  cv::Mat _images_histograms;

  cv::Ptr<cv::ml::StatModel> _model;

  void extractFeatures(cv::Mat image,
                       std::vector<cv::KeyPoint>& out_keypoints,
                       cv::Mat& out_descriptors);

  cv::Mat buildVocabulary(std::vector<cv::Mat>& images,
                          std::vector<cv::Mat>& out_images_descriptors);

  cv::Mat buildHistogram(cv::Mat& descriptors,
                         cv::Mat& vocabulary);

  void buildHistograms(cv::Mat& vocabulary,
                       std::vector<cv::Mat> images_descriptors,
                       std::vector<int> images_labels,
                       cv::Mat& out_images_histogram,
                       cv::Mat& out_images_labels);

public:
  explicit BagOfWords(size_t clusters_count = DEFAULT_CLUSTERS_SIZE,
                      cv::Ptr<cv::ml::StatModel> model = cv::ml::KNearest::create());
  BagOfWords(const BagOfWords& that);
  BagOfWords& operator=(const BagOfWords& that);

  void fit(std::vector<cv::Mat>& images,
           std::vector<int>& images_labels);

  int predict(cv::Mat& image);
};

} // namespace detection

#endif // BAG_OF_WORDS_H
