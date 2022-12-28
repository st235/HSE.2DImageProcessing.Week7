#ifndef CONFUSION_MATRIX_TRACKER_H
#define CONFUSION_MATRIX_TRACKER_H

#include <cstdint>
#include <string>
#include <vector>

#include "annotations_tracker.h"
#include "rect.h"

namespace detection {

struct ConfusionMetric {
public:
  uint32_t tp;
  uint32_t tn;
  uint32_t fp;
  uint32_t fn;

  ConfusionMetric();
  ConfusionMetric(uint32_t tp,
                  uint32_t tn,
                  uint32_t fp,
                  uint32_t fn);
  ConfusionMetric(const ConfusionMetric& that);
  ConfusionMetric& operator=(const ConfusionMetric& that);

  ConfusionMetric merge(const ConfusionMetric& that);

  ConfusionMetric operator+(const ConfusionMetric& that);
  ConfusionMetric& operator+=(const ConfusionMetric& that);

  ~ConfusionMetric() = default;
};

class ConfusionMatrixTracker {
private:
    ConfusionMetric _confusion_metric;

public:
    ConfusionMatrixTracker();
    ConfusionMatrixTracker(const ConfusionMatrixTracker& that);
    ConfusionMatrixTracker& operator=(const ConfusionMatrixTracker& that);

    void keepTrackOf(const FrameInfo& frame_info,
                     const std::vector<std::string>& labels,
                     const std::vector<Rect>& face_origins);

    ~ConfusionMatrixTracker() = default;
};

} // namespace detection

#endif //CONFUSION_MATRIX_TRACKER_H
