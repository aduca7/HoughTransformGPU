#include "project.h"
using namespace std;


/*
naive GPU version of hough
point points[] = array containing points
long pointcount = number of points
Each thread has one point and calculates all votes for that point
*/
__global__ void naiveHough(point* points,unsigned long long* accumulator, int pointCount){
    int pointID=threadIdx.x+(blockDim.x*blockIdx.x);
    point p=points[pointID];
    double conversion=M_PI/180;
    double rho;
    if(pointID<pointCount){
        for(int theta=0;theta<180;theta++){
            for(int phi=0;phi<180;phi++){
                rho=p.x*sin(phi*conversion)*cos(theta*conversion)
                +p.y*sin(phi*conversion)*sin(theta*conversion)
                +p.z*cos(phi*conversion);
                //Use atomic to avoid collisions
                atomicAdd(&accumulator[phi*(180)*(2*92)+(theta*2*92)+((int)floor(rho)+92)],1);
            }
        }
    }
}

/*
Naive GPU implementation of non-maximum-supression
unsigned long long* accumulator = accumulator containing votes from Hough Transform.
int radius = radius for range of supression around a given box
int width, height, depth = dimensions of accumulator.
unsigned long long* d_output= output array
Same as CPU version but each thread does one iteration of loop concurrently.
*/
__global__ void naiveSupression(int radius, unsigned long long *accumulator, int width, int height, int depth, unsigned long long* d_output){
    int i=threadIdx.x+(blockDim.x*blockIdx.x);
    if(i<height*width*depth){
        int index;
        int rho=i%width;
        int theta=(i/width)%height;
        int phi=(i/(width*height))%depth;
        int rstart=max(rho-radius,0);
        int rend=min(rho+radius+1,width);
        int tstart=max(theta-radius,0);
        int tend=min(theta+radius+1,height);
        int pstart=max(phi-radius,0);
        int pend=min(phi+radius+1,depth);
        int currentVal=accumulator[i];
        for(int j=pstart;j<pend;j++){
            for(int k=tstart;k<tend;k++){
                for(int l=rstart;l<rend;l++){
                    index=j*(width)*(height)+k*width+l;
                        if(accumulator[index]>=currentVal&&index!=i){
                            currentVal=0;
                    }
                }
            }
        }
        d_output[i]=currentVal;
    }

}
/*
naive GPU function to pick winning planes
unsigned long long* accumulator = accumulator containing votes from Hough Transform.
int winnercount = number of planes to pick
int size = number of planes in accumulator
Loop through entire array to and pick highest value
VERY SLOW
*/
__global__ void naiveDetermineWinners(int winnerCount, unsigned long long* accumulator,int width, int height, int depth, unsigned long long* winners){
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    int end=width*height*depth;
    int count=0;
    if(accumulator[i]!=0){
        for(int j=0;j<end;j++){
            if(accumulator[i]<accumulator[j]||(i>j&&accumulator[i]==accumulator[j])){
                count++;
                if(count>=winnerCount){
                    break;
                }
            }
        }
        if(count<winnerCount){
            winners[count]=i;
        }
    }
}

//"Main" of naive GPU version
unsigned long long* naive_compute_on_device(int winnerCount,point* h_points, long pointCount,int rad){
    long bytes=pointCount*sizeof(point);
    long accumBytes=2*92*180*180*sizeof(unsigned long long);
    long winnerBytes=winnerCount*sizeof(unsigned long long);

    //Initialize device input vectors
    point* d_points;


	//Initialize output matrices
    unsigned long long* d_accumulator,*h_accumulator,*d_output, *d_winners, *h_winners;

    //Allocate memory
    h_accumulator=new unsigned long long[92*180*360]();
    h_winners=new unsigned long long[winnerCount]();
    cudaMalloc(&d_points, bytes);
    cudaMalloc(&d_accumulator,accumBytes);
    cudaMalloc(&d_output,accumBytes);
    cudaMalloc(&d_winners,winnerBytes);

    //Set accumulator to start at 0;
    cudaMemset(&d_accumulator,0,accumBytes);
    
	

	//Copy input to device
    cudaMemcpy(d_points,h_points,bytes,cudaMemcpyHostToDevice);
    


    cudaEvent_t start, stop;
    float time; 
    cudaEventCreate(&start); 
    cudaEventCreate(&stop); 
    cudaEventRecord( start, 0 ); 

    //Run kernel
    int blockSize=1024;
    int gridSize=ceil((double)pointCount/(double)1024);
    naiveHough<<<gridSize,blockSize>>>(d_points,d_accumulator,pointCount);
    gridSize=ceil((double)(2*92*180*180)/(double)1024);
    naiveSupression<<<gridSize,blockSize>>>(rad,d_accumulator,2*92,180,180,d_output);
    naiveDetermineWinners<<<gridSize,blockSize>>>(winnerCount,d_output,2*92,180,180,d_winners);

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop ); 
    cudaEventDestroy( start ); 
    cudaEventDestroy( stop ); 

    printf("GPU Version Time elapsed: %f ms\n",time);




	//Copy output to host and deallocate device memory.
    cudaMemcpy(h_winners,d_winners,winnerBytes,cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_accumulator,d_output,accumBytes,cudaMemcpyDeviceToHost);
	cudaFree(d_points);
    cudaFree(d_accumulator);
    cudaFree(d_output);
    cudaFree(d_winners);
    delete[] h_accumulator;
	return h_winners;
}
