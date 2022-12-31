#include "dlib_face_detection_model.h"

#include "dlib_utils.h"

namespace detection {

DlibFaceDetectionModel::DlibFaceDetectionModel():
        _detector(dlib::get_frontal_face_detector()) {
    // empty on purpose
}

DlibFaceDetectionModel::DlibFaceDetectionModel(const DlibFaceDetectionModel& that):
        _detector(that._detector) {
    // empty on purpose
}

DlibFaceDetectionModel& DlibFaceDetectionModel::operator=(const DlibFaceDetectionModel& that) {
    if (this != &that) {
        this->_detector = that._detector;
    }

    return *this;
}

std::vector<Face> DlibFaceDetectionModel::extractFaces(cv::Mat& raw_image) {
    dlib::array2d<dlib::rgb_pixel> image = AsRGBOpenCVMatrix(raw_image);
    dlib::pyramid_up(image);

    std::vector<Face> result_faces;
    std::vector<cv::Rect> faces;
    _face_cascade.detectMultiScale(greyscale_image, faces, 1.1, 6);

    for(size_t i = 0; i < faces.size(); i++) {
        const auto& face = faces[i];
        cv::Rect face_rect(face.left(), face.top(), (face.right() - face.left()), (face.top() - face.bottom()));
        cv::Mat face_area = raw_image(face_rect);

        result_faces.push_back(Face(
                output_image,
                Rect::from(face_rect),
                Eyes()));
    }

    return result_faces;
}
    
} // namespace detection
