#ifndef DNN_RECOGNITION_MODEL_H
#define DNN_RECOGNITION_MODEL_H

#include <vector>
#include <string>

#include <dlib/dnn.h>
#include <dlib/clustering.h>
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>

#include "face_recognition_model.h"

namespace {

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N,BN,1,dlib::tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2,2,2,2,dlib::skip1<dlib::tag2<block<N,BN,2,dlib::tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<dlib::con<N,3,3,1,1,dlib::relu<BN<dlib::con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = dlib::relu<residual<block,N,dlib::affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block,N,dlib::affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using face_recognition_dnn_model = dlib::loss_metric<dlib::fc_no_bias<128,dlib::avg_pool_everything<
    alevel0<
        alevel1<
            alevel2<
                alevel3<
                    alevel4<
                        dlib::max_pool<3,3,2,2,dlib::relu<dlib::affine<dlib::con<32,7,7,2,2,dlib::input_rgb_image_sized<150>>>>>
                    >
                >
            >
        >
    >
>>>;

} // namespace

namespace detection {

/**
 * Deep neural network recognition model.
*/
class DnnRecognitionModel: public FaceRecognitionModel {
private:
  dlib::shape_predictor _shape_predictor;
  face_recognition_dnn_model _face_recognition_dnn_model;
  std::string _dnn_model_file;

  cv::Ptr<cv::ml::StatModel> _model;

  std::vector<double> extractFeatures(const cv::Mat& mat) const;

public:
  DnnRecognitionModel(const std::string& landmarks_model_file = "shape_predictor_68_face_landmarks.dat",
                      const std::string& dnn_model_file = "dlib_face_recognition_resnet_model_v1.dat",
                      cv::Ptr<cv::ml::StatModel> model = cv::ml::KNearest::create());
  DnnRecognitionModel(const DnnRecognitionModel& that);
  DnnRecognitionModel& operator=(const DnnRecognitionModel& that);

  void write(const std::string& file) override;
  void read(const std::string& file) override;

  void train(std::vector<cv::Mat>& images,
             std::vector<int>& images_labels) override;

  int predict(cv::Mat& image) const override;

  ~DnnRecognitionModel() = default;
};

} // namespace detection

#endif //DNN_RECOGNITION_MODEL_H
