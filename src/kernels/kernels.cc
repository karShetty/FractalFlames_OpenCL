#include "flam3.h"

typedef struct flam3 flam3;

//Ranom number generation
//http://www.reedbeta.com/blog/quick-and-easy-gpu-random-numbers-in-d3d11/
__kernel void randomWarpInit(unsigned int seed, __global unsigned int* state, const unsigned int noOfStates){
    const unsigned int gID =  get_global_id(0);
    if (gID < noOfStates){
        unsigned int seed_ = seed + gID;
        seed_ = (seed_ ^ 61) ^ (seed_ >> 16);
        seed_ *= 9;
        seed_ = seed_ ^ (seed_ >> 4);
        seed_ *= 0x27d4eb2d;
        seed_ = seed_ ^ (seed_ >> 15);
        state[gID] = seed_;
    }
}

//Random no :[0,1)
float rand_uniform(unsigned int *state){
    const float f_inv = (1.f / 4294967296.0f);
    unsigned int rng_state = *state;
    rng_state ^= (rng_state << 13);
    rng_state ^= (rng_state >> 17);
    rng_state ^= (rng_state << 5);
    *state = rng_state;
    return rng_state * f_inv;
}


uchar4 tex1D( __global uchar4 *color_index, float idx, int N) {
size_t gID = get_global_id(0);
    int i   = (int)(idx);
    int j   = i + 1;
    float a = (idx - i);
    if(i<0) { i=0; }
    if(j<0) { j=0; }
    if( i>=N ){ i=N-1; }
    if( j>=N ){ j=N-1; }
    uchar4 T1 = color_index[i];
    uchar4 T2= color_index[j];
    return (uchar4)(
    a * T1.x + (1 - a) * T2.x,
    a * T1.y + (1 - a) * T2.y,
    a * T1.z + (1 - a) * T2.z,
    a * T1.w + (1 - a) * T2.w
    );
}


void updatePositon(__global const flam3 *flam3Fn, float2* pos){
    float2 posNew, pos_;
    posNew.x = flam3Fn->a * (*pos).x + flam3Fn->b * (*pos).y + flam3Fn->c;
    posNew.y = flam3Fn->d * (*pos).x + flam3Fn->e * (*pos).y + flam3Fn->f;
    pos_.x = 0;
    pos_.y = 0;
    if(flam3Fn->linear != 0.f){
        pos_.x += flam3Fn->linear * posNew.x;
        pos_.y += flam3Fn->linear * posNew.y;
    }
    if(flam3Fn->swirl != 0.f){
        float r2 = (posNew.x * posNew.x + posNew.y * posNew.y);
        pos_.x += flam3Fn->swirl * ( posNew.x * sin(r2)  - posNew.y * cos(r2));
        pos_.y += flam3Fn->swirl * ( posNew.x * cos(r2)  + posNew.y * sin(r2));
    }
    //ToDo: Fill other variations
    *pos = pos_;
}

//ToDo:replce name of d_color_hist* to d_hist_color*
__kernel void resetMemoryColorHist(__global float2 *d_color_hist, unsigned int imageSizeSS){
    unsigned int gID =  get_global_id(0);
    if(gID < imageSizeSS){
        d_color_hist[gID] = (float2)(0.f, 0.f);
    }
}


__kernel void resetMemoryImage(__global uchar4 *d_image, unsigned int noOfPoints){
    unsigned int gID =  get_global_id(0);
    uchar4 reset_d_image;
    reset_d_image = (uchar4)(0, 0, 0, 0);
    if(gID < noOfPoints){
        d_image[gID] = reset_d_image;
    }
}

//Initialise Init Point/Color
__kernel void randomPointColorGenerator(__global float3* d_pos_color, __global unsigned int* state){
    unsigned int gID =  get_global_id(0);
    unsigned int state_ = state[gID];
    //ToDO: Make all equations genereic
    //Assuming aspect ratio of 1.25:1
    d_pos_color[gID].x = rand_uniform(&state_) * 2.5f - 1.25f;
    d_pos_color[gID].y = rand_uniform(&state_) * 2.f - 1.f;
    //Assuming color pallete size of 5
    d_pos_color[gID].z = rand_uniform(&state_) * 4.f;
    state[gID] = state_;
}

//No. of points should be same as no of threads stared
__kernel void iteratorKernel(__global float2* d_color_hist,
                             __global float3* d_pos_color,
                             __global unsigned int* d_ring,
                             __global unsigned int* state,
                             __global const flam3* d_flam3FnList,
                             unsigned int noOfFunctions) {
    unsigned int gID =  get_global_id(0);
    unsigned  int tID =  get_local_id(0);
    unsigned int lRingTID = tID;
    float2 pos;
    float color;

    unsigned int warp_id = 	get_group_id(0) * BLOCK_SIZE/32 + tID/32;
    unsigned int state_warp = state[warp_id];

    __local float shared_weightList[BLOCK_SIZE];
    __local float3 shared_d_pos_color[BLOCK_SIZE];
    __local unsigned int shared_d_ring[BLOCK_SIZE * RANDOM_DATA_ACCESS_SIZE];

    if(tID < noOfFunctions){
        shared_weightList[tID] = d_flam3FnList[tID].weight;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    shared_d_pos_color[tID] = d_pos_color[gID];

    for(int i=0; i< RANDOM_DATA_ACCESS_SIZE; ++i){
        shared_d_ring[BLOCK_SIZE *i + tID] = d_ring[BLOCK_SIZE*i + tID];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
	
    //Select a random point between -1 to 1 Different for different threads
    unsigned int rID = tID;
    unsigned int rIDx = shared_d_ring[tID];
    unsigned int lRingID = 0;

    float3 local_pos_color = shared_d_pos_color[rIDx];
    pos.x = local_pos_color.x;
    pos.y = local_pos_color.y;
    color = local_pos_color.z;

    for(unsigned int itr = 1; itr < MAX_ITERATIONS; ++itr){

	    //Select a random number, same for the entire warp - To prevent warp divergence
        float randWeight = rand_uniform(&state_warp);

		unsigned int fnListItr = 0;
        while(randWeight >= shared_weightList[fnListItr]){
            fnListItr++;
        }


        if(!d_flam3FnList[fnListItr].isSymmetric){
            updatePositon(&d_flam3FnList[fnListItr], &pos); //Function Iterator
            color = (color + d_flam3FnList[fnListItr].c_i) * 0.5f;
        }else{
			float2 pos_ = pos;
            pos.x = d_flam3FnList[fnListItr].a * pos_.x + d_flam3FnList[fnListItr].b * pos_.y;
            pos.y = d_flam3FnList[fnListItr].d * pos_.x + d_flam3FnList[fnListItr].e * pos_.y;
        }

        //ToDo: add post transofrm

        if(itr > SKIP_ITERATIONS){
            //ToDo: add Final Transform
            float2 final_pos = pos;
            float color_f;

            color_f = color;
            //Assuming Ratio 0f 1.25:1
            if( final_pos.x > -1.25f &&   final_pos.x < 1.25f  && final_pos.y > -1.f && final_pos.y < 1.f){
                //NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
                //ToDo: Change to modifiable values, currenty set to 1.25:1 screen size
                unsigned int x_img = ((final_pos.x - -1.25f) * (X_RES * SUPER_SAMPLING)) * 0.4f; /// (1.25f - -1.25f);
                unsigned int y_img = ((final_pos.y - -1.f) * (Y_RES * SUPER_SAMPLING)) * 0.5f; /// //(1.f - -1.f);
                d_color_hist[y_img * X_RES * SUPER_SAMPLING + x_img].x++;
                d_color_hist[y_img * X_RES * SUPER_SAMPLING + x_img].y = color_f;
            }
        }

        local_pos_color = (float3)( pos.x, pos.y, color);
        shared_d_pos_color[rIDx] = local_pos_color;
        lRingID = (lRingID + 1) & (RANDOM_DATA_ACCESS_SIZE -1);
        lRingTID =  ((lRingTID + 1)) & (BLOCK_SIZE - 1);
        rID = (lRingID * BLOCK_SIZE) + lRingTID;

        rIDx = shared_d_ring[rID];
        local_pos_color = shared_d_pos_color[rIDx];
        pos.x = local_pos_color.x;
        pos.y = local_pos_color.y;
        color = local_pos_color.z;
    }
}

//ToDO: Remove fixed values
__kernel void superSample(__global float2 *d_color_hist, __global float2 *d_color_histSS){
    size_t xID =  get_global_id(0);
    size_t yID =  get_global_id(1);
    size_t xID_SS = xID * SUPER_SAMPLING;
    size_t yID_SS = yID * SUPER_SAMPLING;
    float SS_inv = 1.f /(SUPER_SAMPLING * SUPER_SAMPLING);

    float2 itr_ss = (float2)(0.f, 0.f);
    if(xID < X_RES && yID < Y_RES){
        for(size_t x = xID_SS; x < (xID_SS + SUPER_SAMPLING); ++x){
            for(size_t y = yID_SS; y < (yID_SS+SUPER_SAMPLING); ++y){
                itr_ss.x += d_color_hist[x + y * X_RES*SUPER_SAMPLING].x;
                itr_ss.y += d_color_hist[x + y * X_RES*SUPER_SAMPLING].y;
            }
        }
        d_color_histSS[xID + yID *X_RES].x = itr_ss.x * SS_inv;
        d_color_histSS[xID + yID *X_RES].y = itr_ss.y * SS_inv;
    }
}


__kernel void renderKernel(__global float2* d_color_hist,
                           __global uchar4* d_image,
                           __global uchar4* d_color_pallete,
                           float max_histInv,
                           unsigned int colorPalleteSize,
                           unsigned int noOfPoints){
    size_t gID = get_global_id(0);

    float gamma = 1/2.3f;

    if(gID < noOfPoints){
        if(d_color_hist[gID].x != 0){
            uchar4 print_color = tex1D(d_color_pallete, d_color_hist[gID].y , colorPalleteSize);
            float alpha = log10(d_color_hist[gID].x+1.) * max_histInv;
            d_image[gID].x = (unsigned char)(pow(( print_color.x/255.f), (gamma)) *225.f * alpha );
            d_image[gID].y = (unsigned char)(pow(( print_color.y/255.f), (gamma)) *225.f* alpha );
            d_image[gID].z = (unsigned char)(pow(( print_color.z/255.f), (gamma)) *225.f * alpha);
            d_image[gID].w = (unsigned char)(pow(( 1.f), (gamma)) *225.f * alpha);
        }
    }
}

