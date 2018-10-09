#ifndef FRACTAL_FLAMES_FLAM3_H
#define FRACTAL_FLAMES_FLAM3_H

//ToDo: Send values, to kernel not as constants
#define BLOCK_SIZE  128
#define RANDOM_DATA_ACCESS_SIZE 1
#define SUPER_SAMPLING 2
#define SKIP_ITERATIONS 15
#define MAX_ITERATIONS 5000
#define MAX_FUNCTIONS 128
#define X_RES 640
#define Y_RES 512

//Represent Each Function, (i.e include Variation)
//if variation is above 0; it is included
//weight of all functions sums up to 1; is used as a probability measure

struct flam3{
    //Probability Measure
    float weight; //Currently 0-1; Maybe change to 0-100 later

    //Function Co-efficients
    float a;
    float b;
    float c;
    float d;
    float e;
    float f;

    float c_i;

    //Variational Co-efficents, each representing each variation
    float linear;
    float sinusoidal;
    float spherical;
    float swirl;
    float horseshoe;
    float popcorn;
    float pdj;
    float pdj_a;
    float pdj_b;
    float pdj_c;
    float pdj_d;
    float heart;
    float julia;
    float julia_omega;

    //ToDo: Add remaining varitions

    //Post - Transform Param
    float alpha;
    float beta;
    float gamma;
    float delta;
    float epsilon;
    float xsi;
    bool isSymmetric;

    //ToDo: Add other parameters as required
};

#endif //FRACTAL_FLAMES_FLAM3_H
