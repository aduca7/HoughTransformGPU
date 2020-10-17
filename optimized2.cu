#include "project.h"
using namespace std;


/*
First optimized GPU version of hough
point points[] = array containing points
long pointcount = number of points
Each thread has one row of accumulator to write to and adds votes to it directly.
Gives local row to avoid global memory accesses, saves A LOT of time.
*/
__global__ void optimizedHough2(unsigned long long* accumulator, int pointCount){

    int index=threadIdx.x+(blockDim.x*blockIdx.x);
    if(index<32400){
        int phi,theta;
        double sinp,cosp,sint,cost;
        double conversion=M_PI/180;
        phi=index%180;
        theta=index/180;
        sinp=sin(phi*conversion);
        cosp=cos(phi*conversion);
        sint=sin(theta*conversion);
        cost=cos(theta*conversion);
        point p;
        double rho;
        ushort localAccumulator[184];
        memset(localAccumulator,0,184*sizeof(ushort));
        for(int i=0;i<pointCount;i++){
                p=c_points[i];
                rho=p.x*sinp*cost
                +p.y*sinp*sint
                +p.z*cosp;
                localAccumulator[((int)floor(rho)+92)]++;
        }
        //Write entire local row to global accumulator at end of thread to decrease global memory accesses.
        for(int r=0;r<184;r++){
            accumulator[phi*(180)*(2*92)+theta*2*92+r]+=localAccumulator[r];
        }

    }
}

//"Main" of second optimized version
unsigned long long* optimized_compute_on_device2(int winnerCount,point* h_points, long pointCount,int rad){
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
    







    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start, 0 );

    //Run kernel
    int blockSize=1024;
    int gridSize=ceil((double)(180*180)/(double)1024);
    long pointsLeft=pointCount;
    long startIndex=0;
    //int chunkSize=2047;
    // int i;
    // int chunkCount=pointCount/chunkSize;
    // for(i=0;i<chunkCount;i++){
    //     cudaMemcpyToSymbol(c_points,(&h_points[i*chunkSize]),sizeof(point)*chunkSize,cudaMemcpyHostToDevice);
    //     optimizedHough2<<<gridSize,blockSize>>>(d_accumulator,chunkSize);
    // }
    // if(pointCount%chunkSize!=0){
    //     cudaMemcpyToSymbol(c_points,(&h_points[chunkCount*chunkSize]),sizeof(point)*chunkSize,cudaMemcpyHostToDevice);
    //     optimizedHough2<<<gridSize,blockSize>>>(d_accumulator,pointCount-(chunkSize*chunkCount));

    // }
    while(pointsLeft>0){
        if(pointsLeft>=2047){
            pointCount=2047;
            pointsLeft-=2047;
        }
        else{
            pointCount=pointsLeft;
            pointsLeft=0;
        }
        cudaMemcpyToSymbol(c_points,&h_points[startIndex],sizeof(point)*pointCount);
        optimizedHough2<<<gridSize,blockSize>>>(d_accumulator,pointCount);
        startIndex+=pointCount;
    }
    
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
