#ifndef FACE_RECOGNITION_MODEL_H
#define FACE_RECOGNITION_MODEL_H

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

namespace detection {

/**
 * An abstract class for a face recognition model.
 * Any recognition model is an algorithm to extract
 * face features and/or classification model for features.
 */
class FaceRecognitionModel {
public:
  virtual void write(const std::string& file) = 0;
  virtual void read(const std::string& file) = 0;

  virtual void train(std::vector<cv::Mat>& images,
                     std::vector<int>& images_labels) = 0;
  virtual int predict(cv::Mat& image) const = 0;

  virtual ~FaceRecognitionModel() = default;
};

} // namespace detection

#endif //FACE_RECOGNITION_MODEL_H
