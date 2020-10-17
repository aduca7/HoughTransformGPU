#include "project.h"
using namespace std;


/*
CPU implementation of non-maximum-supression
unsigned long long* accumulator = accumulator containing votes from Hough Transform.
int radius = radius for range of supression around a given box
int width, height, depth = dimensions of accumulator.
*/
void nonMaximumSupression(unsigned long long* accumulator,int radius, int width, int height, int depth){
    //Initiailize values
    int rho,theta,phi=0;
    int rstart,rend,tstart,tend,pstart,pend=0;
    int currentVal=0;
    bool* isMax=new bool[height*width*depth]();
    int index;
    //Go through accumulator and set all values that are not the local maximum to 0
    for(int i=0;i<depth*height*width;i++){
        rho=i%width;
        theta=(i/width)%height;
        phi=(i/(width*height))%depth;
        //Calculate endpoints of range
        rstart=max(rho-radius,0);
        rend=min(rho+radius+1,width);
        tstart=max(theta-radius,0);
        tend=min(theta+radius+1,height);
        pstart=max(phi-radius,0);
        pend=min(phi+radius+1,depth);
        currentVal=accumulator[i];
        isMax[i]=true;
        //Check if local maximum
        for(int j=pstart;j<pend;j++){
            for(int k=tstart;k<tend;k++){
                for(int l=rstart;l<rend;l++){
                    index=j*(width)*(height)+k*width+l;
                    if(accumulator[index]>=currentVal&&index!=i){
                        isMax[i]=false;
                    }
                }
            }
        }
    }
    //Update accumulator
    for(int i=0;i<depth*height*width;i++){
        if(isMax[i]==false){
            accumulator[i]=0;
        }
    }
    //Free memory
    delete[] isMax;
}


/*
CPU function to pick winning planes
unsigned long long* accumulator = accumulator containing votes from Hough Transform.
int winnercount = number of planes to pick
int size = number of planes in accumulator
*/
unsigned long long* determineWinners(unsigned long long* accumulator, int winnerCount,int size){
    unsigned long long* result=new unsigned long long[winnerCount]();
    int index;
    for(int i=0;i<winnerCount;i++){
        index=(int)(max_element(accumulator,accumulator+size)-accumulator);
        accumulator[index]=0;
        result[i]=index;
    }
    return result;

}

/*
CPU version of hough
point points[] = array containing points
long pointcount = number of points
int rad= radius for non-maximum supression
*/
unsigned long long* CPUHough(point points[],long pointCount,int rad){
    //Start timing.
    cudaEvent_t start, stop;
    float time; 
    cudaEventCreate(&start); 
    cudaEventCreate(&stop); 
    cudaEventRecord( start, 0 ); 
    
    //Create lookup table of sin/cos values
    double rho;
    double sinLookup[180];
    double cosLookup[180];
    double conversion=M_PI/180;
    for(int i=0;i<180;i++){
        sinLookup[i]=sin(i*conversion);
        cosLookup[i]=cos(i*conversion);
    }
    //For each point, calculate every possible plane it could be a part of and increment each bin by 1 vote
    unsigned long long* accumulator=new unsigned long long[2*92*180*180]();
    for(int i=0;i<pointCount;i++){
        for(int theta=0;theta<180;theta++){
            for(int phi=0;phi<180;phi++){
                rho=points[i].x*sinLookup[phi]*cosLookup[theta]
                +points[i].y*sinLookup[phi]*sinLookup[theta]
                +points[i].z*cosLookup[phi];
                accumulator[phi*(180)*(2*92)+theta*2*92+((int)floor(rho)+92)]++;
                }
            }
        }
    //Run rest of algorithm
    nonMaximumSupression(accumulator,rad, 2*92, 180, 180);
    unsigned long long* winners=determineWinners(accumulator,10,2*92*180*180);

    //Stop Timing
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop ); 
    cudaEventDestroy( start ); 
    cudaEventDestroy( stop ); 

    printf("CPU Version Time elapsed: %f ms\n",time);

    delete[] accumulator;
    return winners;
}