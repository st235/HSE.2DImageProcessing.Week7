#ifndef METRICS_UTILS_H
#define METRICS_UTILS_H

#include <iostream>
#include <string>

#include "metrics_tracker.h"

namespace detection {

constexpr int PI_CONFUSION_SCORES =   1;
constexpr int PI_PRECISION        =   2;
constexpr int PI_TPR              =   4;
// recall is an alias for TPR,
// so they should have the same id
constexpr int PI_RECALL           =   4;
constexpr int PI_ACCURACY         =   8;
constexpr int PI_F1               =  16;
constexpr int PI_FNR              =  32;
constexpr int PI_TNR              =  64;
constexpr int PI_FPR              = 128;

static void PrintConfusionMatrix(const std::string& title,
                                 const detection::ConfusionMatrix& matrix,
                                 const int flags = PI_CONFUSION_SCORES | PI_F1) {
    std::string indent = "    ";
    std::cout << title << ":" << std::endl;

    if ((flags & PI_CONFUSION_SCORES) != 0) {
        std::cout << indent
            << "tp=" << matrix.tp
            << ", tn=" << matrix.tn
            << ", fp=" << matrix.fp
            << ", fn=" << matrix.fn << std::endl;
    }

    if ((flags & PI_PRECISION) != 0) {
        std::cout << indent
                  << "precision=" << matrix.precision() << std::endl;
    }

    if ((flags & PI_RECALL) != 0 || (flags & PI_TPR) != 0) {
        std::cout << indent
                  << "recall(aka TPR)=" << matrix.recall() << std::endl;
    }

    if ((flags & PI_ACCURACY) != 0) {
        std::cout << indent
                  << "accuracy=" << matrix.accuracy() << std::endl;
    }

    if ((flags & PI_F1) != 0) {
        std::cout << indent
                  << "f1=" << matrix.f1() << std::endl;
    }

    if ((flags & PI_FNR) != 0) {
        std::cout << indent
                  << "FNR=" << matrix.fnr() << std::endl;
    }

    if ((flags & PI_TNR) != 0) {
        std::cout << indent
                  << "TNR=" << matrix.tnr() << std::endl;
    }

    if ((flags & PI_FPR) != 0) {
        std::cout << indent
                  << "FPR=" << matrix.fpr() << std::endl;
    }
}

} // namespace detection

#endif //METRICS_UTILS_H
