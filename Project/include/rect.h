#ifndef RECT_H
#define RECT_H

#include <cstdint>

#include <opencv2/opencv.hpp>

namespace detection {

struct Rect {
  const uint32_t x;
  const uint32_t y;
  const uint32_t width;
  const uint32_t height;

  static Rect from(const cv::Rect& that);
  double iou(const Rect& one, const Rect& another);

  Rect();
  Rect(uint32_t x, uint32_t y, uint32_t width, uint32_t height);
  Rect(const Rect& that);
  // There is no reason to assign one rect to another,
  // especially, when all fields are const
  Rect& operator=(const Rect& that) = delete;

  bool intersects(const Rect& that) const;
  Rect intersection(const Rect& that) const;

  Rect escapeFromOldBasis(const Rect basis) const;

  uint64_t area() const;

  bool empty() const;

  ~Rect() = default;
};

} // namespace detection

#endif //RECT_H
