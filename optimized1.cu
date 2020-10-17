#include "project.h"
using namespace std;


/*
First optimized GPU version of hough
point points[] = array containing points
long pointcount = number of points
Each thread has one row of accumulator to write to and adds votes to it directly.
Switches from scatter to gather, makes no further optimizations
*/
__global__ void optimizedHough1(point* points,unsigned long long* accumulator, int pointCount){
    int index=threadIdx.x+(blockDim.x*blockIdx.x);
    if(index<32400){
        int phi,theta;
        phi=index%180;
        theta=index/180;
        point p;
        double conversion=M_PI/180;
        double rho;
        for(int i=0;i<pointCount;i++){
                p=points[i];
                rho=p.x*sin(phi*conversion)*cos(theta*conversion)
                +p.y*sin(phi*conversion)*sin(theta*conversion)
                +p.z*cos(phi*conversion);
                accumulator[phi*(180)*(2*92)+(theta*2*92)+((int)floor(rho)+92)]++;
        }

    }
}

//"Main" of first optimized version
unsigned long long* optimized_compute_on_device1(int winnerCount,point* h_points, long pointCount,int rad){
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
    int gridSize=ceil((double)(180*180)/(double)1024);
    optimizedHough1<<<gridSize,blockSize>>>(d_points,d_accumulator,pointCount);
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
