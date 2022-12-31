#ifndef DLIB_FACE_DETECTION_MODEL_H
#define DLIB_FACE_DETECTION_MODEL_H

#include <dlib/image_processing/frontal_face_detector.h>

namespace detection {

class DlibFaceDetectionModel: public FaceDetectionModel {
private:
    dlib::frontal_face_detector _detector;

public:
    DlibFaceDetectionModel();
    DlibFaceDetectionModel(const DlibFaceDetectionModel& that);
    DlibFaceDetectionModel& operator=(const DlibFaceDetectionModel& that);

    std::vector<Face> extractFaces(cv::Mat& image) override;

    ~DlibFaceDetectionModel() = default;
};    
    
} // namespace detection

#endif //DLIB_FACE_DETECTION_MODEL_H
