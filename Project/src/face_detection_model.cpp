#include "face_detection_model.h"

namespace {

cv::Ptr<cv::Tracker> CreateTracker(detection::FaceDetectionModel::Model model) {
    switch (model) {
        case detection::FaceDetectionModel::Model::KCF: return cv::TrackerKCF::create();
        case detection::FaceDetectionModel::Model::MIL: return cv::TrackerMIL::create();
        case detection::FaceDetectionModel::Model::CSRT: return cv::TrackerCSRT::create();
        case detection::FaceDetectionModel::Model::GOTURN: return cv::TrackerGOTURN::create();
    }
}

} // namespace

namespace detection {

FaceDetectionModel::FaceDetectionModel(FaceDetectionModel::Model model):
    _model(model),
    _trackers() {
    // empty on purpose
}

FaceDetectionModel::FaceDetectionModel(const FaceDetectionModel& that):
    _model(that._model),
    _trackers(that._trackers) {
    // empty on purpose
}

FaceDetectionModel& FaceDetectionModel::operator=(const FaceDetectionModel& that) {
    if (this != &that) {
        this->_model = that._model;
        this->_trackers = that._trackers;
    }

    return *this;
}

void FaceDetectionModel::reset_tracking(cv::Mat& frame,
                                        std::vector<std::string>& labels,
                                        std::vector<Rect>& faces_origins) {
    if (faces_origins.size() != labels.size()) {
        throw std::runtime_error("Faces and labels have different sizes.");
    }

    _trackers.clear();

    for (size_t i = 0; i < faces_origins.size(); i++) {
        const auto& face_origin = faces_origins[i];
        const auto& label = labels[i];

        cv::Ptr<cv::Tracker> tracker = CreateTracker(_model);
        tracker->init(frame, Rect::toCVRect(face_origin));

        _trackers[label] = tracker;
    }
}

void FaceDetectionModel::track(cv::Mat& frame,
                               std::vector<std::string>& labels,
                               std::vector<Rect>& out_faces_origins) {
    for (const auto& label: labels) {
        if (_trackers.find(label) == _trackers.end()) {
            out_faces_origins.push_back(Rect());
            continue;
        }

        cv::Rect out_face;
        _trackers[label]->update(frame, out_face);
        out_faces_origins.push_back(Rect::from(out_face));
    }
}

} // namespace detection
