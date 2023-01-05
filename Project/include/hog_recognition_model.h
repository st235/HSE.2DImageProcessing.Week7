#ifndef HOG_RECOGNITION_MODEL_H
#define HOG_RECOGNITION_MODEL_H

#include "face_recognition_model.h"

namespace detection {

class HogRecognitionModel: public FaceRecognitionModel {
private:
  cv::Ptr<cv::ml::StatModel> _model;

  cv::Mat extractFeatures(cv::Mat image) const;

public:
  explicit HogRecognitionModel(cv::Ptr<cv::ml::StatModel> model = cv::ml::KNearest::create());
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
