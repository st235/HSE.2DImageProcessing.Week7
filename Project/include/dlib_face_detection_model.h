#ifndef DLIB_FACE_DETECTION_MODEL_H
#define DLIB_FACE_DETECTION_MODEL_H

#include <dlib/image_processing/frontal_face_detector.h>

#include "face_detection_model.h"

namespace detection {

class DLibFaceDetectionModel: public FaceDetectionModel {
private:
    dlib::frontal_face_detector _detector;

public:
    DLibFaceDetectionModel();
    DLibFaceDetectionModel(const DLibFaceDetectionModel& that);
    DLibFaceDetectionModel& operator=(const DLibFaceDetectionModel& that);

    std::vector<Face> extractFaces(const Rect& viewport, cv::Mat& image) override;

    ~DLibFaceDetectionModel() = default;
};    
    
} // namespace detection

#endif //DLIB_FACE_DETECTION_MODEL_H
