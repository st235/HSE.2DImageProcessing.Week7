#ifndef FACE_TRACKING_MODEL_H
#define FACE_TRACKING_MODEL_H

#include <string>
#include <vector>
#include <unordered_map>

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

#include "face_utils.h"
#include "rect.h"

namespace detection {

class FaceTrackingModel {
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

  explicit FaceTrackingModel(FaceTrackingModel::Model model);
  FaceTrackingModel(const FaceTrackingModel& that);
  FaceTrackingModel& operator=(const FaceTrackingModel& that);

  void reset_tracking(cv::Mat& frame,
                      std::vector<std::string>& labels,
                      std::vector<Rect>& faces_origins);

  void track(cv::Mat& frame,
             std::vector<std::string>& labels,
             std::vector<Rect>& out_faces_origins);

  virtual ~FaceTrackingModel() = default;

private:
    FaceTrackingModel::Model _model;
    std::unordered_map<std::string, cv::Ptr<cv::Tracker>> _trackers;
};

} // namespace detection

#endif //FACE_TRACKING_MODEL_H
