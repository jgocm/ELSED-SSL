#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <ELSED.h>

namespace py = pybind11;
using namespace upm;

// Converts C++ descriptors to Numpy
inline py::tuple salient_segments_to_py(const upm::SalientSegments &ssegs) {
  py::array_t<float> scores({int(ssegs.size()), 1});
  py::array_t<float> labels({int(ssegs.size()), 1});
  py::array_t<float> segments({int(ssegs.size()), 4});
  py::array_t<float> grad_x({int(ssegs.size()), 3});
  py::array_t<float> grad_y({int(ssegs.size()), 3});
  float *p_scores = scores.mutable_data();
  float *p_labels = labels.mutable_data();
  float *p_segments = segments.mutable_data();
  float *p_grad_x = grad_x.mutable_data();
  float *p_grad_y = grad_y.mutable_data();
  for (int i = 0; i < ssegs.size(); i++) {
    p_scores[i] = ssegs[i].salience;
    p_labels[i] = ssegs[i].classification;
    p_segments[i * 4] = ssegs[i].segment[0];
    p_segments[i * 4 + 1] = ssegs[i].segment[1];
    p_segments[i * 4 + 2] = ssegs[i].segment[2];
    p_segments[i * 4 + 3] = ssegs[i].segment[3];
    p_grad_x[i * 3] = ssegs[i].g_BGRx[0];
    p_grad_x[i * 3 + 1] = ssegs[i].g_BGRx[1];
    p_grad_x[i * 3 + 2] = ssegs[i].g_BGRx[2];
    p_grad_y[i * 3] = ssegs[i].g_BGRy[0];
    p_grad_y[i * 3 + 1] = ssegs[i].g_BGRy[1];
    p_grad_y[i * 3 + 2] = ssegs[i].g_BGRy[2];  }
  return pybind11::make_tuple(segments, scores, labels, grad_x, grad_y);
}

py::tuple compute_elsed(const py::array &py_img,
                        float sigma = 1,
                        float gradientThreshold = 30,
                        int minLineLen = 15,
                        double lineFitErrThreshold = 0.2,
                        double pxToSegmentDistTh = 1.5,
                        double validationTh = 0.15,
                        bool validate = true,
                        bool treatJunctions = true,
                        double boundaryGradTh = 27.78332309,
                        double boundaryAngleTh = 51.21788891,
                        double boundaryMinLength = 95,
                        double markingGradTh = 32.17114637,
                        double markingAngleTh = 29.69457975,
                        double markingMinLength = 104
) {

  py::buffer_info info = py_img.request();

  cv::Mat img;  // Declare the img variable before the if statement

  if (info.ndim == 2) {
      // Grayscale image
      // std::cout << "The image is grayscale." << std::endl;
      img = cv::Mat(info.shape[0], info.shape[1], CV_8UC1, (uint8_t *) info.ptr);
  } else if (info.ndim == 3 && info.shape[2] == 3) {
      // BGR image
      // std::cout << "The image is BGR." << std::endl;
      img = cv::Mat(info.shape[0], info.shape[1], CV_8UC3, (uint8_t *) info.ptr);
  } else {
      std::cerr << "Unexpected image format!" << std::endl;
  }

  ELSEDParams params;

  params.sigma = sigma;
  params.ksize = cvRound(sigma * 3 * 2 + 1) | 1; // Automatic kernel size detection
  params.gradientThreshold = gradientThreshold;
  params.minLineLen = minLineLen;
  params.lineFitErrThreshold = lineFitErrThreshold;
  params.pxToSegmentDistTh = pxToSegmentDistTh;
  params.validationTh = validationTh;
  params.validate = validate;
  params.treatJunctions = treatJunctions;
  params.boundaryGradTh = boundaryGradTh;
  params.boundaryAngleTh = boundaryAngleTh;
  params.boundaryMinLength = boundaryMinLength;
  params.markingGradTh = markingGradTh;
  params.markingAngleTh = markingAngleTh;
  params.markingMinLength = markingMinLength;

  ELSED elsed(params);
  upm::SalientSegments salient_segs = elsed.detectSalient(img);

  return salient_segments_to_py(salient_segs);
}

int classify_segment_from_gradients(const py::array &grad_x,
                                    const py::array &grad_y,
                                    double segment_length,
                                    double boundaryGradTh = 27.78332309,
                                    double boundaryAngleTh = 51.21788891,
                                    double boundaryMinLength = 95,
                                    double markingGradTh = 32.17114637,
                                    double markingAngleTh = 29.69457975,
                                    double markingMinLength = 104
) {

  ELSEDParams params;

  params.boundaryGradTh = boundaryGradTh;
  params.boundaryAngleTh = boundaryAngleTh;
  params.boundaryMinLength = boundaryMinLength;
  params.markingGradTh = markingGradTh;
  params.markingAngleTh = markingAngleTh;
  params.markingMinLength = markingMinLength;

  py::buffer_info info_x = grad_x.request();
  cv::Mat g_BGRx(1, info_x.shape[0], CV_32F, info_x.ptr);

  py::buffer_info info_y = grad_y.request();
  cv::Mat g_BGRy(1, info_y.shape[0], CV_32F, info_y.ptr);

  int seg_classification = isFieldFeature(g_BGRx, g_BGRy, segment_length, params);

  return seg_classification;
}

// TODO: Make new function bind to compute the segment classification with the gradients as inputs
PYBIND11_MODULE(pyelsed, m) {
  m.def("detect", &compute_elsed, R"pbdoc(
        Computes ELSED: Enhanced Line SEgment Drawing in the input image.
    )pbdoc",
        py::arg("img"),
        py::arg("sigma") = 1,
        py::arg("gradientThreshold") = 30,
        py::arg("minLineLen") = 15,
        py::arg("lineFitErrThreshold") = 0.2,
        py::arg("pxToSegmentDistTh") = 1.5,
        py::arg("validationTh") = 0.15,
        py::arg("validate") = true,
        py::arg("treatJunctions") = true,
        py::arg("boundaryGradTh") = 27.78332309,
        py::arg("boundaryAngleTh") = 51.21788891,
        py::arg("boundaryMinLength") = 95,
        py::arg("markingGradTh") = 32.17114637,
        py::arg("markingAngleTh") = 29.69457975,
        py::arg("markingMinLength") = 104
  );
  m.def("classify", &classify_segment_from_gradients, R"pbdoc(
        Classifies a line segment based on its gradients, length and given thresholds.
    )pbdoc",
        py::arg("grad_x"),
        py::arg("grad_y"),
        py::arg("segment_length"),
        py::arg("boundaryGradTh") = 27.78332309,
        py::arg("boundaryAngleTh") = 51.21788891,
        py::arg("boundaryMinLength") = 95,
        py::arg("markingGradTh") = 32.17114637,
        py::arg("markingAngleTh") = 29.69457975,
        py::arg("markingMinLength") = 104
  );
}
