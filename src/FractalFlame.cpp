#include <GL/glew.h>
#include <GL/glut.h>
#include "Fractal_OCL.h"
#include "flam3Data.h"
#include <iostream>

#define MAX_FRAMES 100


GLuint pbo = 0;
GLuint tex = 0;
//ToDo: W,H should be part of args
unsigned int W = X_RES;
unsigned int H = Y_RES;

//ToDo: Should be a part of flam3Functionset
unsigned int itr;
unsigned int fNitr1,fNitr2;
unsigned int fnList;
flam3FunctionSet fFunctionSet;
Fractal_OCL fOCL;

//ToDo: Color Pallete shoud be a part of flam3Functionset,
//Todo: create a 3d vector
unsigned char colorPallete[3][5][4] = {{{255,0,125,255},{0,255,0,255},{50,0,255,255},{254,255,0,255},{0,125,255,255}},
                                       {{255,30,0,255},{255,46,0,255},{255,66,0,255},{255,89,0,255},{255,102,0,255}},
                                       {{58,48,66,255},{219,157,71,255},{255,120,79,255},{255,255,156,255},{237,255,217,255}}};
const unsigned int colorPaletteSize = 5;

//ToDo: Modify this
std::unique_ptr<unsigned char []> palleteCombiner(unsigned char *colorPallete,const unsigned int colorPalleteSize, unsigned int f1, unsigned int f2, int itr){
    std::unique_ptr<unsigned char []> p(new unsigned char[colorPalleteSize * 4]);
    float res = (float)itr/(MAX_FRAMES-1);
    for(size_t i = 0; i < colorPalleteSize; ++i){
        for(size_t j = 0; j < 4; ++j){
            p[i*4 + j] = (unsigned char)(colorPallete[f1*colorPalleteSize*4+ i*4 +j] * res +  colorPallete[f2*colorPalleteSize*4+ i*4 +j] * (1-res));
        }
    }
    return p;
}

void render(){
    void *d_pbo = glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_READ_WRITE);
    if(itr < MAX_FRAMES){
        std::vector<flam3> interFn = fFunctionSet.getInterpolatedFunction(fFunctionSet.getFunctionSetVector(fNitr1),fFunctionSet.getFunctionSetVector(fNitr2),(float)itr/(MAX_FRAMES-1));
        fOCL.setFunctionList(interFn.data(), interFn.size());
        fOCL.resetFractalMemory(d_pbo);
//        std::unique_ptr<unsigned char []> p(new unsigned char[colorPaletteSize * 4]);
//        for(size_t i = 0; i < colorPaletteSize; ++i){
//            for(size_t j = 0; j < 4; ++j){
//                p[i*4 + j] = (unsigned char)(colorPallete[0][i][j]);
//            }
//        }
//        fOCL.setColorPallete(std::move(p), colorPaletteSize);
        fOCL.setColorPallete(std::move(palleteCombiner((unsigned char *)colorPallete, colorPaletteSize, fNitr1, fNitr2, itr)), colorPaletteSize);
        fOCL.iterateFractal();
        fOCL.renderFractal(d_pbo);
        itr++;
    }else{
        fNitr1++;
        fNitr2++;
        itr = 0;
        if(fNitr2 == fnList ){
            fNitr2 = 0;
        }
        if(fNitr1 == fnList ){
            fNitr1 = 0;
        }
    }
    glUnmapBuffer( GL_PIXEL_UNPACK_BUFFER );
}

void drawTexture(){
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(0, 0);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(0, H);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(W, H);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(W, 0);
    glEnd();
    glDisable(GL_TEXTURE_2D);
}


void display(){
    render();
    drawTexture();
    glutSwapBuffers();
    glutPostRedisplay();
}

void cleanup(void)
{
    //ToDo:Add Clean up
}

void initPixelBuffer(){
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * W * H * sizeof(cl_uchar), 0, GL_STREAM_DRAW);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
}


int main(int argc, char *argv[]){
    itr = 0;
    fNitr1 = 0;
    fNitr2 = 1;

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(W, H);
    glutCreateWindow("Fractal Flames");
    glewInit();
    gluOrtho2D(0, W, H, 0);

    fFunctionSet.createNewFunctionSet();
    fFunctionSet.addVariation(BasicCoeff::create(0.5,0,.5,0.3,0.2,0.4),0.4,1.2,{{swirl,{1.}}});
    fFunctionSet.addVariation(BasicCoeff::create(0.4,0.2,0,0.5,0.4,0),0.6,.2,{{linear,{.3f}}, {linear,{.7f}}});
    fFunctionSet.addSymmetryAndNormalize(symmetry_90);
    fnList++;

    fFunctionSet.createNewFunctionSet();
    fFunctionSet.addVariation(BasicCoeff::create(1.5,0.4,.5,0.3,0.5,0.1),.5,2.2,{{linear,{.5f}}, {linear,{.8f}}});
    fFunctionSet.addVariation(BasicCoeff::create(0.4,-0.2,0,-0.5,0.9,0.2),0.5,3.2,{{linear,{.1f}}, {linear,{.3f}}});
    fFunctionSet.addSymmetryAndNormalize(symmetry_90);
    fnList++;

    fFunctionSet.createNewFunctionSet();
    fFunctionSet.addVariation(BasicCoeff::create(1.2,1.4,-.5,-0.3,1.5,0.1),.5,2.2,{{linear,{.5f}}, {linear,{.8f}}});
    fFunctionSet.addVariation(BasicCoeff::create(0.4,-0.2,0,-0.5,0.2,0.2),.3,3.2,{{linear,{.8f}}});
    fFunctionSet.addVariation(BasicCoeff::create(0.5,0.2,0.5,-0.5,0.1,0.2),.2,0.2,{{swirl,{.5f}}, {linear,{.3f}}});
    fFunctionSet.addSymmetryAndNormalize(symmetry_120);
    fnList++;

    fOCL.initRandom(W,H,8192);
    fOCL.initFractal();
    glutDisplayFunc(display);
    initPixelBuffer();
    glutMainLoop();
    atexit(cleanup);

    return 0;
}
