#include "flam3Data.h"

//ToDo: Think of a better approach to iterate over struct
flam3Ptr __at[] = {&flam3::weight,
                   &flam3::a,
                   &flam3::b,
                   &flam3::c,
                   &flam3::d,
                   &flam3::e,
                   &flam3::f,
                   &flam3::c_i,
                   &flam3::linear,
                   &flam3::sinusoidal,
                   &flam3::spherical,
                   &flam3::swirl,
                   &flam3::horseshoe,
                   &flam3::popcorn,
                   &flam3::pdj,
                   &flam3::pdj_a,
                   &flam3::pdj_b,
                   &flam3::pdj_c,
                   &flam3::pdj_d,
                   &flam3::heart,
                   &flam3::julia,
                   &flam3::julia_omega, //Needs to be between 0 to 3.14
                   &flam3::alpha,
                   &flam3::beta,
                   &flam3::gamma,
                   &flam3::delta,
                   &flam3::epsilon,
                   &flam3::xsi};


flam3FunctionSet::flam3FunctionSet() {
    finalTansform = new flam3{};
}


void flam3FunctionSet::createNewFunctionSet() {
    functionSet.emplace_back(std::vector<flam3>());
}

void flam3FunctionSet::addVariation(BasicCoeff coeff, float weight, float color,
                                    std::initializer_list<std::pair<variationType, std::initializer_list<float>>> varCoeff) {
    if (functionSet.size() != 0) {
        std::vector<flam3> &currentFunctionSet = functionSet.back();
        flam3 currentSet = {};

        currentSet.a = coeff.a;
        currentSet.b = coeff.b;
        currentSet.c = coeff.c;
        currentSet.d = coeff.d;
        currentSet.e = coeff.e;
        currentSet.f = coeff.e;

        currentSet.weight = weight;
        currentSet.c_i = color;

        for (auto list : varCoeff) {
            switch (list.first) {
                case linear:
                    currentSet.linear = *list.second.begin();
                    break;
                case sinusoidal:
                    currentSet.sinusoidal = *list.second.begin();
                    break;
                case spherical:
                    currentSet.spherical = *list.second.begin();
                    break;
                case swirl:
                    currentSet.swirl = *list.second.begin();
                    break;
                case horseshoe:
                    currentSet.horseshoe = *list.second.begin();
                    break;
                case popcorn:
                    currentSet.popcorn = *list.second.begin();
                    break;
                case pdj:
                    currentSet.pdj = *list.second.begin();
                    currentSet.pdj_a = *(list.second.begin() + 1);
                    currentSet.pdj_b = *(list.second.begin() + 2);
                    currentSet.pdj_c = *(list.second.begin() + 3);
                    currentSet.pdj_d = *(list.second.begin() + 4);
                    break;
                case heart:
                    currentSet.heart = *list.second.begin();
                case julia:
                    currentSet.julia = *list.second.begin();
                    currentSet.julia_omega = *(list.second.begin() + 1);
                default:
                    break;
            }
        }
        currentFunctionSet.emplace_back(currentSet);
    } else {
        throw std::runtime_error("No Function set created");
    }
}

void flam3FunctionSet::addSymmetryAndNormalize(Symmetry symmetry) {
    if (functionSet.size() != 0) {
        std::vector<flam3> &currentFunctionSet = functionSet.back();
        //Normalize frst
        float orderd_weights = normalize(symmetry);

        if (symmetry == symmetry_x || symmetry == symmetry_y) {
            flam3 currentSt = {};
            currentSt.a = symmetry == symmetry_x ? 1.f : -1.f;
            currentSt.b = 0.f;
            currentSt.c = 0.f;
            currentSt.d = 0.f;
            currentSt.e = symmetry == symmetry_x ? -1.f : 1.f;
            currentSt.f = 0.f;
            currentSt.linear = 1.f;
            currentSt.weight = 1.f;
            currentSt.isSymmetric = true;
            currentFunctionSet.emplace_back(currentSt);
        } else {
            if (symmetry != no_symmetry) {
                flam3 currentSt = {};
                float theta = 0;
                float sym_weight = 1.f / (2.f * symmetry);
                for (unsigned int symIdx = 0; symIdx < symmetry; ++symIdx) {
                    theta += 2.f * 3.147f / (symmetry + 1);
                    currentSt.a = cos(theta);
                    currentSt.b = -sin(theta);
                    currentSt.c = 0.f;
                    currentSt.d = sin(theta);
                    currentSt.e = cos(theta);
                    currentSt.f = 0.f;
                    currentSt.linear = 1.f;
                    orderd_weights += sym_weight;
                    currentSt.weight = orderd_weights;
                    currentSt.isSymmetric = true;
                    currentFunctionSet.emplace_back(currentSt);
                }
            }
        }
    } else {
        throw std::runtime_error("No Function set created");
    }
}

float flam3FunctionSet::normalize(Symmetry symmetry = no_symmetry) {
    if (functionSet.size() != 0) {
        std::vector<flam3> &currentFunctionSet = functionSet.back();
        //Normalize frst
        float weighted_sum = 0.f;
        for (auto varList : currentFunctionSet) {
            weighted_sum += varList.weight;
        }

        if (symmetry != no_symmetry)
            weighted_sum *= 2;

        float orderd_weights = 0.f;
        for (auto &varList : currentFunctionSet) {
            varList.weight /= weighted_sum;
            float currentWeight = varList.weight;
            varList.weight += orderd_weights;
            orderd_weights += currentWeight;
        }
        return orderd_weights;
    } else {
        throw std::runtime_error("No Function set created");
    }

}

std::vector<flam3> flam3FunctionSet::getFunctionSetVector(unsigned int idx) {
    if (idx < functionSet.size()) {
        return functionSet.at(idx);
    } else {
        throw std::runtime_error("Index greater than no. of function sets");
    }
}

std::vector<flam3>
flam3FunctionSet::getInterpolatedFunction(const std::vector<flam3> &lhs, const std::vector<flam3> &rhs, float ratio) {
    if (ratio > 1.f)
        ratio = 1.f;
    if (ratio < 0)
        ratio = 0;
    float ratio1_t = 1 - ratio;

    if (lhs.data() != NULL && rhs.data() != NULL && lhs.data() != rhs.data() && lhs.size() == rhs.size()) {
        std::vector<flam3> interpolatedFn(lhs.size());
        for (unsigned int idx = 0; idx < lhs.size(); ++idx) {
            for (unsigned int i = 0; i < FLMA3_NO_OF_ITEMS; ++i)
                interpolatedFn[idx].*__at[i] = ratio1_t * lhs[idx].*__at[i] + ratio * rhs[idx].*__at[i];
            if (lhs[idx].isSymmetric && rhs[idx].isSymmetric)
                interpolatedFn[idx].isSymmetric = true;
            else if(lhs[idx].isSymmetric)
                interpolatedFn[idx].c_i = lhs[idx].c_i;
            else
                interpolatedFn[idx].c_i = rhs[idx].c_i;
            //ToDo: add the else case
        }
        return interpolatedFn;
    } else {
        throw std::runtime_error("Interpolated Function Not Set");
    }
}


