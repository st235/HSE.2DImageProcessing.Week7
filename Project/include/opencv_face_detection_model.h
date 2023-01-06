#ifndef OPENCV_FACE_DETECTION_MODEL_H
#define OPENCV_FACE_DETECTION_MODEL_H

#include "face_detection_model.h"
#include "rect.h"

namespace {

const double DEFAULT_FACE_SCALE_FACTOR = 1.1;
const uint32_t DEFAULT_FACE_MIN_NEIGHBOURS = 6;
const double DEFAULT_EYES_SCALE_FACTOR = 1.1;
const uint32_t DEFAULT_EYES_MIN_NEIGHBOURS = 6;
const std::string DEFAULT_FACE_CASCADE_FILE_PATH = "haarcascade_frontalface_alt2.xml";
const std::string DEFAULT_RIGHT_EYE_CASCADE_FILE_PATH = "haarcascade_righteye_2splits.xml";
const std::string DEFAULT_LEFT_EYE_CASCADE_FILE_PATH = "haarcascade_lefteye_2splits.xml";

} // namespace

namespace detection {

class OpenCVFaceDetectionModel: public FaceDetectionModel {
private:
  double _face_scale_factor;
  uint32_t _face_min_neighbours;
  double _eyes_scale_factor;
  uint32_t _eyes_min_neighbours;

  cv::CascadeClassifier _face_cascade;
  cv::CascadeClassifier _right_eye_cascade;
  cv::CascadeClassifier _left_eye_cascade;

public:
 OpenCVFaceDetectionModel(double face_scale_factor = DEFAULT_FACE_SCALE_FACTOR,
                          uint32_t face_min_neighbours = DEFAULT_FACE_MIN_NEIGHBOURS,
                          double eyes_scale_factor = DEFAULT_EYES_SCALE_FACTOR,
                          uint32_t eyes_min_neighbours = DEFAULT_EYES_MIN_NEIGHBOURS,
                          const std::string& face_cascade_file_path = DEFAULT_FACE_CASCADE_FILE_PATH,
                          const std::string& right_eye_cascade_file_path = DEFAULT_RIGHT_EYE_CASCADE_FILE_PATH,
                          const std::string& left_eye_cascade_file_path = DEFAULT_LEFT_EYE_CASCADE_FILE_PATH);
 OpenCVFaceDetectionModel(const OpenCVFaceDetectionModel& that);
 OpenCVFaceDetectionModel& operator=(const OpenCVFaceDetectionModel& that);

 std::vector<Face> extractFaces(const Rect& viewport, cv::Mat& image) override;

 ~OpenCVFaceDetectionModel() = default;
};

} // namespace detection

#endif //OPENCV_FACE_DETECTION_MODEL_H
