
#ifndef PROJECT_H_
#define PROJECT_H_

#include <iostream>
#include <string>
#include <fstream>
#include <ostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <queue>


//Constants
#define MAXRHO 92
#define RADIUS 4
#define TILEWIDTH (2*RADIUS+1)
#define BLOCKWIDTH 17



//Point in 3-dimensional space
struct point{
    double x;
    double y;
    double z;
};

//Struct for comparing elements of winner array
struct winnerElement{
    unsigned long long votes;
    int index;
    bool operator<(const winnerElement& other) const
    {
        if(other.votes>votes){
            return true;
        }
        else if(other.votes==votes&&index<other.index){
            return true;
        }
        else{
            return false;
        }
    }
};

//Plane in spherical coordinate system
struct plane{
    int rho;
    int theta;
    int phi;
};

//Color in RGB
struct color{
    int r;
    int g;
    int b;
};

//Could have used same struct for point,plane, and color but did this to avoid confusion.

//Function prototypes
void compareAccumulators(unsigned long long* accumCPU,unsigned long long*accumGPU,int size);
__constant__ point c_points[2047];
unsigned long long* CPUHough(point points[],long pointCount,int rad);
unsigned long long* naive_compute_on_device(int winnerCount,point* h_points, long pointCount,int rad);
__global__ void naiveHough(point* points,unsigned long long* accumulator, int pointCount);
__global__ void naiveSupression(int radius, unsigned long long *accumulator, int width, int height, int depth, unsigned long long* d_output);
__global__ void naiveDetermineWinners(int winnerCount, unsigned long long* accumulator,int width, int height, int depth, unsigned long long* winners);
__global__ void optimizedHough2(unsigned long long* accumulator, int pointCount);
__global__ void optimizedSupression(unsigned long long *accumulator, int width, int height, int depth, unsigned long long* d_output);
unsigned long long* optimized_compute_on_device1(int winnerCount,point* h_points, long pointCount,int rad);
unsigned long long* optimized_compute_on_device2(int winnerCount,point* h_points, long pointCount,int rad); 
unsigned long long* optimized_compute_on_device3(int winnerCount,point* h_points, long pointCount); 
unsigned long long* optimized_compute_on_device4(int winnerCount,point* h_points, long pointCount);


#endif
