#ifndef FACE_DETECTION_MODEL_H
#define FACE_DETECTION_MODEL_H

#include <string>
#include <vector>
#include <unordered_map>

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

#include "face_utils.h"
#include "rect.h"

namespace detection {

class FaceDetectionModel {
public:
  enum class Model {
      // the best balance between speed and accuracy
      KCF,
      // slow but accurate, slower than {@code KCF}
      MIL,
      // fast but makes a lot of errors
      CSRT,
      GOTURN
  };

  explicit FaceDetectionModel(FaceDetectionModel::Model model);
  FaceDetectionModel(const FaceDetectionModel& that);
  FaceDetectionModel& operator=(const FaceDetectionModel& that);

  void reset_tracking(cv::Mat& frame,
                      std::vector<std::string>& labels,
                      std::vector<Rect>& faces_origins);

  void track(cv::Mat& frame,
             std::vector<std::string>& labels,
             std::vector<Rect>& out_faces_origins);

  virtual ~FaceDetectionModel() = default;

private:
    FaceDetectionModel::Model _model;
    std::unordered_map<std::string, cv::Ptr<cv::Tracker>> _trackers;
};

} // namespace detection

#endif //FACE_DETECTION_MODEL_H
