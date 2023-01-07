#ifndef HOG_RECOGNITION_MODEL_H
#define HOG_RECOGNITION_MODEL_H

#include "face_recognition_model.h"

namespace {

const uint32_t DEFAULT_MAX_NEIGHBOURS = 50;
const double MAX_NEIGHBOURS_DISTANCE = 250;

} // namespace

namespace detection {

class HogRecognitionModel: public FaceRecognitionModel {
private:
  uint32_t _max_neighbours;
  double _max_neighbours_distance;
  cv::Ptr<cv::ml::KNearest> _knearest;

  cv::Mat extractFeatures(cv::Mat image) const;

public:
  explicit HogRecognitionModel(uint32_t max_neighbours = DEFAULT_MAX_NEIGHBOURS,
                               double max_neighbours_distance = MAX_NEIGHBOURS_DISTANCE);
  HogRecognitionModel(const HogRecognitionModel& that);
  HogRecognitionModel& operator=(const HogRecognitionModel& that);

  void write(const std::string& file) override;
  void read(const std::string& file) override;

  void train(std::vector<cv::Mat>& images,
             std::vector<int>& images_labels) override;
  int predict(cv::Mat& image) const override;

  ~HogRecognitionModel() = default;
};

} // namespace detection

#endif //HOG_RECOGNITION_MODEL_H
