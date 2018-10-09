#include "Fractal_OCL.h"

Fractal_OCL::Fractal_OCL(std::vector<std::string> kernel_filepaths) {
    std::srand(5);
    platformID = 2;
    deviceID = 0;
    initOCL(kernel_filepaths);
}

//ToDo: platforID and deviceID should be selected automatically
Fractal_OCL::Fractal_OCL() {
    std::vector<std::string> kernel_filepaths;
    kernel_filepaths.push_back("../src/kernels/kernels.cc");
    std::srand(5);
    platformID = 2;
    deviceID = 0;
    initOCL(kernel_filepaths);
}


bool Fractal_OCL::initRandom(const unsigned int width_,
                             const unsigned int height_,
                             const unsigned int noOfPoints_,
                             const unsigned int superSampling_,
                             const unsigned int seed_) {
    if (width_ != width || height_ != height || noOfPoints_ != noOfPoints || superSampling_ != superSampling || seed_ != seed) {
        width = width_;
        height = height_;
        noOfPoints = noOfPoints_;

        //ToDo: don't use fixed size
        superSampling = SUPER_SAMPLING;
        seed = seed_;
        imageSizeSS = width * superSampling * height * superSampling;
        imageSize = width * height;
        try {
            d_states = std::make_shared<cl::Buffer>(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * noOfPoints);
            cl::Kernel randomWarpInit(program, "randomWarpInit");
            randomWarpInit.setArg(0, seed);
            randomWarpInit.setArg(1, *d_states);
            randomWarpInit.setArg(2, noOfPoints);
            queue.enqueueNDRangeKernel(randomWarpInit,
                                       cl::NullRange,
                                       cl::NDRange(noOfPoints),
                                       cl::NDRange(BLOCK_SIZE));


            std::mt19937 rng(std::random_device{}());
            std::array<unsigned int, BLOCK_SIZE> h_single_ring;
            std::array<unsigned int, BLOCK_SIZE * RANDOM_DATA_ACCESS_SIZE> h_ring;

            std::iota(h_single_ring.begin(), h_single_ring.end(), 0);
            auto itr_begin = h_ring.begin();

            for (size_t i = 0; i < RANDOM_DATA_ACCESS_SIZE; ++i) {
                std::shuffle(h_single_ring.begin(), h_single_ring.end(), rng);
                std::copy(h_single_ring.begin(), h_single_ring.end(), itr_begin);
                itr_begin += BLOCK_SIZE;
            }

            d_ring = std::make_shared<cl::Buffer>(context, CL_MEM_WRITE_ONLY,
                                                  sizeof(cl_uint) * BLOCK_SIZE * RANDOM_DATA_ACCESS_SIZE);

            queue.enqueueWriteBuffer(*d_ring, CL_TRUE, 0, sizeof(cl_uint) * BLOCK_SIZE * RANDOM_DATA_ACCESS_SIZE,
                                     h_ring.data());
            queue.finish();
        }
        catch (cl::Error const &cerr) {
            std::cerr << OCLError("Fractal_OCL::initRandom  ", cerr);
        }
    }
    return true;
}


bool Fractal_OCL::initFractal() {
    try {
        d_pos_color = std::make_shared<cl::Buffer>(context, CL_MEM_READ_WRITE, sizeof(cl_float3) * noOfPoints);
        h_pos_color = std::vector<cl_float3>(noOfPoints);

        randomPointColorGenerator = cl::Kernel (program, "randomPointColorGenerator");
        randomPointColorGenerator.setArg(0, *d_pos_color);
        randomPointColorGenerator.setArg(1, *d_states);

        queue.enqueueNDRangeKernel(randomPointColorGenerator,
                                   cl::NullRange,
                                   cl::NDRange(noOfPoints),
                                   cl::NDRange(BLOCK_SIZE));

        d_color_hist = std::make_shared<cl::Buffer>(context, CL_MEM_READ_WRITE, sizeof(cl_float2) * imageSize);
        d_color_histSS = std::make_shared<cl::Buffer>(context, CL_MEM_READ_WRITE, sizeof(cl_float2) * imageSizeSS);

        resetMemoryColorHist = cl::Kernel (program, "resetMemoryColorHist");
        resetMemoryColorHist.setArg(0, *d_color_histSS);
        resetMemoryColorHist.setArg(1, imageSizeSS);

        d_flam3FnList =  std::make_shared<cl::Buffer>(context, CL_MEM_READ_WRITE, sizeof(flam3) * MAX_FUNCTIONS);

        iteratorKernel = cl::Kernel (program, "iteratorKernel");
        iteratorKernel.setArg(0, *d_color_histSS);
        iteratorKernel.setArg(1, *d_pos_color);
        iteratorKernel.setArg(2, *d_ring);
        iteratorKernel.setArg(3, *d_states);
        iteratorKernel.setArg(4, *d_flam3FnList);
        iteratorKernel.setArg(5, noOfFunctions);

        resetMemoryImage = cl::Kernel (program, "resetMemoryImage");
        resetMemoryImage.setArg(1, imageSizeSS);
        queue.finish();

        superSample = cl::Kernel(program, "superSample");

        renderKernel = cl::Kernel (program, "renderKernel");
        renderKernel.setArg(5, imageSize);
    }
    catch (cl::Error const &cerr) {
        std::cerr << OCLError("Fractal_OCL::initFractalMemory  ", cerr);
        return false;
    }
    return true;
}

bool Fractal_OCL::setFunctionList(flam3 *flam3FnList, const unsigned int fnListSize) {
    try {
        queue.enqueueWriteBuffer(*d_flam3FnList, CL_TRUE, 0, sizeof(flam3) * fnListSize, flam3FnList);
        noOfFunctions = fnListSize;
        iteratorKernel.setArg(5, noOfFunctions);
        queue.finish();
    }
    catch (cl::Error const &cerr) {
        std::cerr << OCLError("Fractal_OCL::setFunctionList  ", cerr);
        return false;
    }
    return true;
}


bool Fractal_OCL::iterateFractal() {
    try {
        queue.enqueueNDRangeKernel(iteratorKernel,
                                   cl::NullRange,
                                   cl::NDRange(noOfPoints),
                                   cl::NDRange(BLOCK_SIZE));
        queue.finish();
    }
    catch (cl::Error const &cerr) {
        std::cerr << OCLError("Fractal_OCL::iterateFractal  ", cerr);
        return false;
    }
    return true;
}



bool Fractal_OCL::resetFractalMemory(void *d_image) {
    try {

        queue.enqueueNDRangeKernel(resetMemoryColorHist,
                                   cl::NullRange,
                                   cl::NDRange(imageSizeSS),
                                   cl::NDRange(BLOCK_SIZE));
        queue.finish();

        cl_mem mem =  clCreateBuffer(context(),CL_MEM_WRITE_ONLY|CL_MEM_USE_HOST_PTR,
                                     imageSize*sizeof(cl_uchar4), d_image, NULL);
        clSetKernelArg(resetMemoryImage(), 0, sizeof(mem) , &mem);
        queue.enqueueNDRangeKernel(resetMemoryImage,
                                   cl::NullRange,
                                   cl::NDRange(imageSize),
                                   cl::NDRange(BLOCK_SIZE ));
        queue.finish();
        clReleaseMemObject(mem);
    }
    catch (cl::Error const &cerr) {
        std::cerr << OCLError("Fractal_OCL::resetFractalMemory  ", cerr);
    }
	return true;
}



void Fractal_OCL::initOCL(std::vector<std::string> kernel_filepaths) {
    try{
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty())
            throw std::runtime_error("No platforms found");
        if (platformID > platforms.size())
            throw std::runtime_error("Invalid platformID");

        cl::Platform platform = platforms[platformID];


        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty())
            throw std::runtime_error("No devices found");;
        if (deviceID > devices.size())
            throw std::runtime_error("Invalid deviceID");
        cl::Device device = devices[deviceID];

        context = cl::Context(device);
        cl::Program::Sources source;

        std::vector<std::string> kernelCodes;
        for(auto kernel_filepath : kernel_filepaths){
            std::ifstream kernelOpenCLFile(kernel_filepath);
            kernelCodes.emplace_back(std::istreambuf_iterator<char>(kernelOpenCLFile), (std::istreambuf_iterator<char>()));
            source.push_back(std::make_pair(kernelCodes.back().c_str(), kernelCodes.back().length()));
        }

        program = cl::Program(context, source);
        try {
            program.build("-I ../include/");
        }
        catch (cl::Error const &cerr) {
            if (cerr.err() == CL_BUILD_PROGRAM_FAILURE) {
                cl_build_status status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device);
                std::string name = device.getInfo<CL_DEVICE_NAME>();
                std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                std::cerr << "Build log for " << name << ":" << std::endl << buildlog << std::endl;
            }
            else {
                throw cerr;
            }
        }
        queue = cl::CommandQueue(context, device);
    }
    catch (cl::Error const &cerr){
        std::cerr << OCLError("Fractal_OCL::initOCL  ", cerr);
    }

}


bool Fractal_OCL::setColorPallete(std::unique_ptr<unsigned char []>  pallete, unsigned int _colorPalleteSize) {
    colorPalleteSize = _colorPalleteSize;
    try{
        d_color_pallete = std::make_shared<cl::Buffer>(context, CL_MEM_READ_WRITE, sizeof(cl_uchar4) * colorPalleteSize);
        queue.enqueueWriteBuffer(*d_color_pallete, CL_TRUE, 0, sizeof(cl_uchar4) * colorPalleteSize, pallete.get());
        queue.finish();
        renderKernel.setArg(4, colorPalleteSize);
    }
    catch (cl::Error const &cerr){
        std::cerr << OCLError("Fractal_OCL::setColorPallete  ", cerr);
        return false;
    }
    return true;
}


bool Fractal_OCL::renderFractal(void *d_image){
    try{
        superSample.setArg(0, *d_color_histSS);
        superSample.setArg(1, *d_color_hist);
        queue.enqueueNDRangeKernel(superSample,
                                   cl::NullRange,
                                   cl::NDRange(width, height),
                                   cl::NDRange(BLOCK_SIZE, 1));
        queue.finish();


        h_color_hist = std::vector<cl_float2>(imageSize);

        //ToDo: Find max using reduction,,, no copying to host memory
        queue.enqueueReadBuffer(*d_color_hist,
                                CL_FALSE,
                                0,
                                sizeof(cl_float2) * imageSize,
                                h_color_hist.data());

        queue.finish();
        float max_hist = 0.;
        for(auto x : h_color_hist){
            if (x.s[0] > max_hist)
                max_hist = x.s[0];
        }
        float maxHistInv = 1./ std::log10(max_hist + 1);

        cl_mem mem =  clCreateBuffer(context(),CL_MEM_WRITE_ONLY|CL_MEM_USE_HOST_PTR,
                                     imageSize*sizeof(cl_uchar4), d_image, NULL);
        renderKernel.setArg(0, *d_color_hist);

        //ToDo: Check why setArg creates problem
        clSetKernelArg(renderKernel(), 1, sizeof(mem) , &mem);
        renderKernel.setArg(2, *d_color_pallete);
        renderKernel.setArg(3, maxHistInv);

        queue.enqueueNDRangeKernel(renderKernel,
                                   cl::NullRange,
                                   cl::NDRange(imageSize),
                                   cl::NDRange(BLOCK_SIZE));
        queue.finish();
        clReleaseMemObject(mem);

    }
    catch (cl::Error const &cerr){
        std::cerr << OCLError("Fractal_OCL::renderFractal  ", cerr);
        return false;
    }
    return true;
}