#include "dlib_face_detection_model.h"

#include "dlib_utils.h"

namespace detection {

DLibFaceDetectionModel::DLibFaceDetectionModel():
        _detector(dlib::get_frontal_face_detector()) {
    // empty on purpose
}

DLibFaceDetectionModel::DLibFaceDetectionModel(const DLibFaceDetectionModel& that):
        _detector(that._detector) {
    // empty on purpose
}

DLibFaceDetectionModel& DLibFaceDetectionModel::operator=(const DLibFaceDetectionModel& that) {
    if (this != &that) {
        this->_detector = that._detector;
    }

    return *this;
}

std::vector<Face> DLibFaceDetectionModel::extractFaces(cv::Mat& raw_image) {
    dlib::array2d<dlib::rgb_pixel> image = AsRGBOpenCVMatrix(raw_image);

    std::vector<Face> result_faces;
    std::vector<dlib::rectangle> faces = _detector(image);
    for(size_t i = 0; i < faces.size(); i++) {
        const auto& face = faces[i];
        cv::Rect face_rect(face.left(), face.top(), (face.right() - face.left()), (face.bottom() - face.top()));
        cv::Mat face_area = raw_image(face_rect);

        result_faces.push_back(Face(
                face_area,
                Rect::from(face_rect),
                Eyes()));
    }

    return result_faces;
}
    
} // namespace detection
