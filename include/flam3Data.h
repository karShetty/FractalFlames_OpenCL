#ifndef FRACTAL_FLAMES_FLAM3DATA_H
#define FRACTAL_FLAMES_FLAM3DATA_H

#include <string>
#include <random>
#include <cmath>
#include <vector>
#include <initializer_list>
#include <iostream>
#include "flam3.h"

enum Symmetry {
    no_symmetry,
    symmetry_180,
    symmetry_120,
    symmetry_90,
    symmetry_72,
    symmetry_60,
    symmetry_x,
    symmetry_y
};

enum variationType {
    linear,
    sinusoidal,
    spherical,
    swirl,
    horseshoe,
    popcorn,
    pdj,
    heart,
    julia
};

//SHOULD HAVE THOUGHT OF THIS SHITY PROBLEM EARLIER
//ORDER NEEDS TO BE MAINTAINED
typedef float flam3::* flam3Ptr;
#define FLMA3_NO_OF_ITEMS 25


struct BasicCoeff {
    float a, b, c, d, e, f;

    static BasicCoeff create(float a, float b, float c, float d, float e, float f) {
        return {a, b, c, d, e, f};
    }
};


//Very Basic;
class flam3FunctionSet {
    std::vector<std::vector<flam3>> functionSet;
    flam3 *finalTansform;

public:
    flam3FunctionSet();
    void createNewFunctionSet();
    void addVariation(BasicCoeff coeff, float weight, float color,
                      std::initializer_list<std::pair<variationType, std::initializer_list<float>>> varCoeff);
    void addSymmetryAndNormalize(Symmetry symmetry);
    float normalize(Symmetry symmetry);
    flam3 *getFunctionSet(unsigned int idx);
    flam3 *getFinalTransform();
    std::vector<flam3> getFunctionSetVector(unsigned int idx);
    unsigned int getFunctionSetLength(unsigned int idx);
    std::vector<flam3>
    getInterpolatedFunction(const std::vector<flam3> &lhs, const std::vector<flam3> &rhs, float ratio);
};


#endif //FRACTAL_FLAMES_FLAM3DATA_H
