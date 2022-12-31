#ifndef METRICS_TRACKER_H
#define METRICS_TRACKER_H

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

#include "annotations_tracker.h"
#include "rect.h"

namespace detection {

struct ConfusionMatrix {
public:
  uint32_t tp;
  uint32_t tn;
  uint32_t fp;
  uint32_t fn;

  ConfusionMatrix();
  ConfusionMatrix(uint32_t tp,
                  uint32_t tn,
                  uint32_t fp,
                  uint32_t fn);
  ConfusionMatrix(const ConfusionMatrix& that);
  ConfusionMatrix& operator=(const ConfusionMatrix& that);

  ConfusionMatrix merge(const ConfusionMatrix& that);

  double tpr() const;
  double fnr() const;
  double tnr() const;
  double fpr() const;
  double precision() const;
  double recall() const;
  double f1() const;
  double accuracy() const;

  bool empty() const;

  ConfusionMatrix operator+(const ConfusionMatrix& that);
  ConfusionMatrix& operator+=(const ConfusionMatrix& that);

  ~ConfusionMatrix() = default;
};

class MetricsTracker {
private:
    std::vector<std::string> _labels;
    std::unordered_map<uint32_t, ConfusionMatrix> _detections_per_frame_lookup;
    std::unordered_map<uint32_t, ConfusionMatrix> _known_recognitions_per_frame_lookup;
    std::unordered_map<uint32_t, ConfusionMatrix> _unknown_recognitions_per_frame_lookup;

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

    ConfusionMatrix overallDetectionMetrics() const;

    ConfusionMatrix overallKnownRecognitionMetrics() const;

    ConfusionMatrix overallUnknownRecognitionMetrics() const;

    ConfusionMatrix overallRecognitionMetrics() const;

    ~MetricsTracker() = default;
};

} // namespace detection

#endif //METRICS_TRACKER_H
