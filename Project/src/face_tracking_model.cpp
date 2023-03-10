#include "face_tracking_model.h"

namespace {

cv::Ptr<cv::Tracker> CreateTracker(detection::FaceTrackingModel::Model model) {
    switch (model) {
        case detection::FaceTrackingModel::Model::KCF: return cv::TrackerKCF::create();
        case detection::FaceTrackingModel::Model::MIL: return cv::TrackerMIL::create();
        case detection::FaceTrackingModel::Model::CSRT: return cv::TrackerCSRT::create();
        case detection::FaceTrackingModel::Model::GOTURN: return cv::TrackerGOTURN::create();
    }
}

} // namespace

namespace detection {

FaceTrackingModel::FaceTrackingModel(FaceTrackingModel::Model model):
    _model(model),
    _trackers() {
    // empty on purpose
}

FaceTrackingModel::FaceTrackingModel(const FaceTrackingModel& that):
    _model(that._model),
    _trackers(that._trackers) {
    // empty on purpose
}

FaceTrackingModel& FaceTrackingModel::operator=(const FaceTrackingModel& that) {
    if (this != &that) {
        this->_model = that._model;
        this->_trackers = that._trackers;
    }

    return *this;
}

void FaceTrackingModel::resetTracking(cv::Mat& frame,
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

        _trackers.push_back(tracker);
    }
}

void FaceTrackingModel::track(cv::Mat& frame,
                              std::vector<std::string>& labels,
                              std::vector<Rect>& out_faces_origins) {
    if (labels.size() != _trackers.size()) {
        throw std::runtime_error("Labels and _trackers have different sizes.");
    }

    for (size_t i = 0; i < labels.size(); i++) {
        cv::Rect out_face;
        if (_trackers[i]->update(frame, out_face)) {
            out_faces_origins.push_back(Rect::from(out_face));
        } else {
            // tracking for the given object has
            // failed, let's put empty Rect in
            // this case
            out_faces_origins.push_back(Rect());
        }
    }
}

} // namespace detection
