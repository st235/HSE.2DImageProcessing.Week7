#ifndef METRICS_TRACKER_H
#define METRICS_TRACKER_H

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

#include "annotations_tracker.h"
#include "rect.h"

namespace detection {

struct BinaryClassificationMatrix {
public:
    static const int32_t INF;

    int32_t tp;
    int32_t tn;
    int32_t fp;
    int32_t fn;

    BinaryClassificationMatrix();
    BinaryClassificationMatrix(int32_t tp,
                               int32_t tn,
                               int32_t fp,
                               int32_t fn);
    BinaryClassificationMatrix(const BinaryClassificationMatrix& that);
    BinaryClassificationMatrix& operator=(const BinaryClassificationMatrix& that);

    BinaryClassificationMatrix merge(const BinaryClassificationMatrix& that);

    double tpr() const;
    double fnr() const;
    double tnr() const;
    double fpr() const;
    double precision() const;
    double recall() const;
    double f1() const;
    double accuracy() const;

    bool empty() const;

    BinaryClassificationMatrix operator+(const BinaryClassificationMatrix& that);
    BinaryClassificationMatrix& operator+=(const BinaryClassificationMatrix& that);

    ~BinaryClassificationMatrix() = default;
};

struct MultiClassificationMatrix {
private:
  uint32_t _number_of_classes;
  uint32_t* _classification_matrix;

public:
  MultiClassificationMatrix(uint32_t number_of_classes);
  MultiClassificationMatrix(const MultiClassificationMatrix& that);
  MultiClassificationMatrix& operator=(const MultiClassificationMatrix& that);

  uint32_t classesSize() const;
  uint32_t metricAt(uint32_t i, uint32_t j) const;

  void track(uint32_t annotated_label_id,
             uint32_t detected_label_id);

  double accuracy() const;

  BinaryClassificationMatrix getBinaryMatrix(uint32_t label_id) const;
  MultiClassificationMatrix remove(uint32_t label_id) const;

  MultiClassificationMatrix operator+(const MultiClassificationMatrix& that);
  MultiClassificationMatrix& operator+=(const MultiClassificationMatrix& that);

  ~MultiClassificationMatrix();
};

class MetricsTracker {
private:
    std::vector<std::string> _labels;
    std::unordered_map<std::string, uint32_t> _labels_to_ids_lookup;
    std::unordered_map<uint32_t, BinaryClassificationMatrix> _detections_per_frame_lookup;
    std::unordered_map<uint32_t, MultiClassificationMatrix> _recognitions_per_frame_lookup;

    /**
     * Brute forces detected rectangles match
     * based on intersection over union.
     * Works for O(n^2) where n is a size of matched rectangles.
     * Should be fine to match
     */
    void trackDetection(const FrameInfo& frame_info,
                        const std::vector<Rect>& detected_face_origins);

    void trackRecognition(const FrameInfo& frame_info,
                          const std::vector<std::string>& labels,
                          const std::vector<Rect>& detected);

public:
    explicit MetricsTracker(std::vector<std::string> labels);
    MetricsTracker(const MetricsTracker& that);
    MetricsTracker& operator=(const MetricsTracker& that);

    void keepTrackOf(const FrameInfo& frame_info,
                     const std::vector<std::string>& detected_labels,
                     const std::vector<Rect>& detected_face_origins);

    BinaryClassificationMatrix overallDetectionMetrics() const;

    MultiClassificationMatrix overallKnownRecognitionMetrics() const;

    BinaryClassificationMatrix overallUnknownRecognitionMetrics() const;

    ~MetricsTracker() = default;
};

} // namespace detection

#endif //METRICS_TRACKER_H
