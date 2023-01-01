#include "rect.h"

#include <algorithm>

namespace {

bool Intersects(int32_t s1, int32_t f1,
                int32_t s2, int32_t f2) {
    return std::max(s1, s2) < std::min(f1, f2);
}

void Merge(int32_t s1, int32_t f1,
           int32_t s2, int32_t f2,
           int32_t& s_out, int32_t& f_out) {
    s_out = std::max(s1, s2);
    f_out = std::min(f1, f2);
}

} // namespace

namespace detection {

std::ostream& operator<<(std::ostream& os, const Rect& rect) {
    os << "Rect{x=" << rect.x << ",y=" << rect.y
       << ",width=" << rect.width << ",height=" << rect.height;
    return os;
}

Rect Rect::from(const cv::Rect& that) {
    if (that.empty()) {
        return Rect();
    }

    return Rect(static_cast<int32_t>(that.x),
                static_cast<int32_t>(that.y),
                static_cast<uint32_t>(that.width),
                static_cast<uint32_t>(that.height));
}

cv::Rect Rect::toCVRect(const Rect& that) {
    if (that.empty()) {
        return cv::Rect();
    }

    return cv::Rect(that.x, that.y, that.width, that.height);
}

double Rect::iou(const Rect& one, const Rect& another) {
    if (!one.intersects(another)) {
        return 0.0;
    }

    Rect intersection = one.intersection(another);

    double intersection_area = static_cast<double>(intersection.area());
    double union_area = static_cast<double>(one.area() + another.area()) - intersection_area;

    return intersection_area / union_area;
}

Rect::Rect():
        x(0),
        y(0),
        width(0),
        height(0) {
    // empty on purpose
}

Rect::Rect(int32_t x, int32_t y, uint32_t width, uint32_t height):
    x(x),
    y(y),
    width(width),
    height(height) {
    // empty on purpose
}

Rect::Rect(const Rect& that):
        x(that.x),
        y(that.y),
        width(that.width),
        height(that.height) {
    // empty on purpose
}

Rect& Rect::operator=(const Rect& that) {
    if (this != &that) {
        this->x = that.x;
        this->y = that.y;
        this->width = that.width;
        this->height = that.height;
    }

    return *this;
}

bool Rect::intersects(const Rect& that) const {
    // checking horizontal axis
    return Intersects(x, x + width, that.x, that.x + that.width) &&
        // checking vertical axis
        Intersects(y, y + height, that.y, that.y + that.height);
}

Rect Rect::intersection(const Rect& that) const {
    int32_t sx, fx;
    int32_t sy, fy;

    Merge(x, x + width, that.x, that.x + that.width, sx, fx);
    Merge(y, y + height, that.y, that.y + that.height, sy, fy);

    return Rect(sx, sy, fx - sx, fy - sy);
}

Rect Rect::escapeFromOldBasis(const Rect basis) const {
    return Rect(basis.x + x, basis.y + y, width, height);
}

uint64_t Rect::area() const {
    return width * height;
}

bool Rect::empty() const {
    return x == 0 && y == 0 && width == 0 && height == 0;
}

} // namespace detection
