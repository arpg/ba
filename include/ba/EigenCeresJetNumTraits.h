// Copyright (c) George Washington University, all rights reserved.  See the
// accompanying LICENSE file for more information.

#pragma once

#include <ceres/jet.h>
#include <Eigen/Core>

namespace Eigen {

template<>
struct NumTraits<ceres::Jet<double,6> >
: NumTraits<double> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
    typedef ceres::Jet<double,6> Real;
    typedef ceres::Jet<double,6> NonInteger;
    typedef ceres::Jet<double,6> Nested;
    enum {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 3,
        MulCost = 3
    };
};

}

namespace ceres {

template <typename T, int N> inline
ceres::Jet<T, N> fabs(const ceres::Jet<T, N>& f) {
    return abs(f);
}

//// TODO: Improve this definition
//template <typename T, int N> inline
//ceres::Jet<T, N> tan(const ceres::Jet<T, N>& f) {
//    return sin(f) / cos(f);
//}

//// TODO: Improve this definition
//template <typename T, int N> inline
//ceres::Jet<T, N> atan(const Jet<T, N>& g) {
//    const Jet<T, N> f(1);
//    return atan2(g,f);
//}

}
