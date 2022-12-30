#include "metrics_tracker.h"

#include <unordered_map>

namespace detection {

ConfusionMatrix::ConfusionMatrix():
    tp(0),
    tn(0),
    fp(0),
    fn(0) {
    // empty on purpose
}

ConfusionMatrix::ConfusionMatrix(uint32_t tp,
                                 uint32_t tn,
                                 uint32_t fp,
                                 uint32_t fn):
        tp(tp),
        tn(tn),
        fp(fp),
        fn(fn) {
    // empty on purpose
}

ConfusionMatrix::ConfusionMatrix(const ConfusionMatrix& that):
        tp(that.tp),
        tn(that.tn),
        fp(that.fp),
        fn(that.fn) {
    // empty on purpose
}

ConfusionMatrix& ConfusionMatrix::operator=(const ConfusionMatrix& that) {
    if (this != &that) {
        this->tp = that.tp;
        this->tn = that.tn;
        this->fp = that.fp;
        this->fn = that.fn;
    }

    return *this;
}

ConfusionMatrix ConfusionMatrix::merge(const ConfusionMatrix& that) {
    return ConfusionMatrix(tp + that.tp,
                           tn + that.tn,
                           fp + that.fp,
                           fn + that.fn);
}

ConfusionMatrix ConfusionMatrix::operator+(const ConfusionMatrix& that) {
    return merge(that);
}

ConfusionMatrix& ConfusionMatrix::operator+=(const ConfusionMatrix& that) {
    this->tp += that.tp;
    this->tn += that.tn;
    this->fp += that.fp;
    this->fn += that.fn;
    return *this;
}

MetricsTracker::MetricsTracker():
    _confusion_metric() {
    // empty on purpose
}

MetricsTracker::MetricsTracker(const MetricsTracker& that):
        _confusion_metric(that._confusion_metric) {
    // empty on purpose
}

MetricsTracker& MetricsTracker::operator=(const MetricsTracker& that) {
    if (this != &that) {
        this->_confusion_metric = that._confusion_metric;
    }

    return *this;
}

void MetricsTracker::keepTrackOf(const FrameInfo& frame_info,
                                         const std::vector<std::string>& labels,
                                         const std::vector<Rect>& face_origins) {
    std::unordered_map<std::string, Rect> references;

    // TODO(st235): handle 'unknown' cases
    std::vector<std::string> ref_labels = frame_info.labels();
    std::vector<Rect> ref_face_origins = frame_info.face_origins();

    for (size_t i = 0; i < frame_info.count(); i++) {
        std::string label = ref_labels[i];
        Rect face_origin = ref_face_origins[i];
        references[label] = face_origin;
    }

    for (size_t i = 0; i < labels.size(); i++) {
        std::string label = labels[i];
        Rect face_origin = face_origins[i];

//        bool has_ref =
//
//        if (Rect::iou(face_origin, ))
    }
}

} // namespace detection
