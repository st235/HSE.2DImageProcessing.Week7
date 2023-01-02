#ifndef METRICS_UTILS_H
#define METRICS_UTILS_H

#include <iostream>
#include <string>

#include "metrics_tracker.h"

namespace detection {

constexpr int PB_CONFUSION_SCORES =   1;
constexpr int PB_PRECISION        =   2;
constexpr int PB_TPR              =   4;
// recall is an alias for TPR,
// so they should have the same id
constexpr int PB_RECALL           =   4;
constexpr int PB_ACCURACY         =   8;
constexpr int PB_F1               =  16;
constexpr int PB_FNR              =  32;
constexpr int PB_TNR              =  64;
constexpr int PB_FPR              = 128;

constexpr int MB_CONFUSION_SCORES =   1;
constexpr int MB_ACCURACY         =   2;


static void PrintBinaryMatrix(const std::string& title,
                              const detection::BinaryClassificationMatrix& matrix,
                              const int flags = PB_CONFUSION_SCORES | PB_F1) {
    std::string indent = "    ";
    std::cout << title << ":" << std::endl;

    if (matrix.empty()) {
        std::cout << indent << "metrics have not been observed" << std::endl;
        return;
    }

    if ((flags & PB_CONFUSION_SCORES) != 0) {
        std::cout << indent
            << "tp=" << matrix.tp
            << ", tn=" << matrix.tn
            << ", fp=" << matrix.fp
            << ", fn=" << matrix.fn << std::endl;
    }

    if ((flags & PB_PRECISION) != 0) {
        std::cout << indent
                  << "precision=" << matrix.precision() << std::endl;
    }

    if ((flags & PB_RECALL) != 0 || (flags & PB_TPR) != 0) {
        std::cout << indent
                  << "recall(aka TPR)=" << matrix.recall() << std::endl;
    }

    if ((flags & PB_ACCURACY) != 0) {
        std::cout << indent
                  << "accuracy=" << matrix.accuracy() << std::endl;
    }

    if ((flags & PB_F1) != 0) {
        std::cout << indent
                  << "f1=" << matrix.f1() << std::endl;
    }

    if ((flags & PB_FNR) != 0) {
        std::cout << indent
                  << "FNR=" << matrix.fnr() << std::endl;
    }

    if ((flags & PB_TNR) != 0) {
        std::cout << indent
                  << "TNR=" << matrix.tnr() << std::endl;
    }

    if ((flags & PB_FPR) != 0) {
        std::cout << indent
                  << "FPR=" << matrix.fpr() << std::endl;
    }
}

static void PrintMulticlassMatrix(const std::string& title,
                                  const detection::MultiClassificationMatrix& matrix,
                                  const int flags = MB_CONFUSION_SCORES | MB_ACCURACY) {
    std::string indent = "    ";
    std::cout << title << ":" << std::endl;

    if ((flags & MB_CONFUSION_SCORES) != 0) {
        for (size_t i = 0; i < matrix.classesSize(); i++) {
            std::cout << indent;

            for (size_t j = 0; j < matrix.classesSize(); j++) {
                std::cout << std::setw(3) << matrix.metricAt(i, j) << " ";
            }
        }

        std::cout << std::endl;
    }

    if ((flags & MB_ACCURACY) != 0) {
        std::cout << indent
                  << "accuracy=" << matrix.accuracy() << std::endl;
    }
}

} // namespace detection

#endif //METRICS_UTILS_H
