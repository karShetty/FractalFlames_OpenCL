#ifndef FRACTAL_FLAMES_FRACTAL_OCL_H
#define FRACTAL_FLAMES_FRACTAL_OCL_H

#define __CL_ENABLE_EXCEPTIONS
#include <CL\cl.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <memory>
#include <algorithm>
#include <random>
#include <array>
#include<numeric>
#include <CL\cl.h>
//#include <CL/cl_platform.h>
#include "ocl_errors.h"
#include "flam3Data.h"




class Fractal_OCL{
public:
    Fractal_OCL(std::vector<std::string> kernel_filepaths);
    Fractal_OCL();
    bool initRandom(const unsigned int width, const unsigned int height, const unsigned int no_of_points, const unsigned int superSampling = 1, const unsigned int seed_ = 0);
    bool initFractal();
    bool resetFractalMemory(void *d_image);
    bool iterateFractal();
    bool setFunctionList(flam3* flam3FnList, const unsigned int fnListSize);
    bool setColorPallete(std::unique_ptr<unsigned char []> pallete, unsigned int _colorPalleteSize); //ToDO: add params
    bool renderFractal(void *d_image);

private:
    void initOCL(std::vector<std::string> kernel_filepaths);
    static std::string OCL_error(const char* msg, cl::Error const& clerr);

    unsigned int platformID;
    unsigned int deviceID;

    unsigned int seed;
    unsigned int width;
    unsigned int height;
    unsigned int noOfPoints;
    unsigned int superSampling;
    unsigned int imageSizeSS;
    unsigned int imageSize;
    unsigned int noOfFunctions;
    unsigned int colorPalleteSize;

    cl::Program program;
    cl::Context context;
    cl::CommandQueue queue;

    cl::Kernel randomPointColorGenerator;
    cl::Kernel resetMemoryColorHist;
    cl::Kernel iteratorKernel;
    cl::Kernel renderKernel;
    cl::Kernel resetMemoryImage;
    cl::Kernel superSample;

    std::shared_ptr<cl::Buffer> d_states;
    std::shared_ptr<cl::Buffer> d_ring;
    std::shared_ptr<cl::Buffer> d_pos_color;
    std::shared_ptr<cl::Buffer> d_color_hist; //ToDo:Rename to d_hist_color as x is hist and y is color
    std::shared_ptr<cl::Buffer> d_color_histSS; //ToDo:Rename to d_hist_colorSS as x is hist and y is color
    std::shared_ptr<cl::Buffer> d_flam3FnList;
    std::shared_ptr<cl::Buffer> d_color_pallete;

    std::vector<cl_float3> h_pos_color;
    std::vector<cl_float2> h_color_hist; //ToDo:Rename to h_hist_color as x is hist and y is color

};

#endif //FRACTAL_FLAMES_FRACTAL_OCL_H
