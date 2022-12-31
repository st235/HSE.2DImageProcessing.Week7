#ifndef OPENCV_FACE_DETECTION_MODEL_H
#define OPENCV_FACE_DETECTION_MODEL_H

#include "face_detection_model.h"

namespace {

const std::string FACE_CASCADE_FILE_PATH = "haarcascade_frontalface_alt2.xml";
const std::string RIGHT_EYE_CASCADE_FILE_PATH = "haarcascade_righteye_2splits.xml";
const std::string LEFT_EYE_CASCADE_FILE_PATH = "haarcascade_lefteye_2splits.xml";

} // namespace

namespace detection {

class OpenCVFaceDetectionModel: public FaceDetectionModel {
private:
  cv::CascadeClassifier _face_cascade;
  cv::CascadeClassifier _right_eye_cascade;
  cv::CascadeClassifier _left_eye_cascade;

public:
 OpenCVFaceDetectionModel(const std::string& face_cascade_file_path = FACE_CASCADE_FILE_PATH,
                          const std::string& right_eye_cascade_file_path = RIGHT_EYE_CASCADE_FILE_PATH,
                          const std::string& left_eye_cascade_file_path = LEFT_EYE_CASCADE_FILE_PATH);
 OpenCVFaceDetectionModel(const OpenCVFaceDetectionModel& that);
 OpenCVFaceDetectionModel& operator=(const OpenCVFaceDetectionModel& that);

 std::vector<Face> extractFaces(cv::Mat& image) override;

 ~OpenCVFaceDetectionModel() = default;
};

} // namespace detection

#endif //OPENCV_FACE_DETECTION_MODEL_H
