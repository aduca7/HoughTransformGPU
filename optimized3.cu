#include "project.h"
using namespace std;




__global__ void optimizedHough3(unsigned long long* accumulator, int pointCount){
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
        for(int r=0;r<184;r++){
            accumulator[phi*(180)*(2*92)+theta*2*92+r]+=localAccumulator[r];
        }

    }
}
/*
Optimized GPU implementation of non-maximum-supression
unsigned long long* accumulator = accumulator containing votes from Hough Transform.
int radius = radius for range of supression around a given box
int width, height, depth = dimensions of accumulator.
unsigned long long* d_output= output array
Each thread is given a square of indices to perform non-maximum supression on
Due to limits of GPU not very efficient, with more shared memory and higher max thread count this would be much faster.

*/
__global__ void optimizedSupression(unsigned long long *accumulator, int width, int height, int depth, unsigned long long* d_output){
    //INitialize shared memory and variables
    __shared__ unsigned int accumS[BLOCKWIDTH][BLOCKWIDTH][BLOCKWIDTH];
    int tx= threadIdx.x;
    int ty = threadIdx.y;
    int r_o=blockIdx.z*TILEWIDTH;
    int r_i=r_o-RADIUS;
    int row_o= blockIdx.y* TILEWIDTH + ty;
	int col_o= blockIdx.x* TILEWIDTH + tx;
    int row_i= row_o-RADIUS;
    int col_i= col_o-RADIUS;
        
    //Copy accumulator segment to shared memory
    for(int i=0;i<BLOCKWIDTH;i++){
        if((row_i>=0)&&(row_i<180)&&(col_i>=0)&&(col_i<180)&&(r_i>=0)&&(r_i<width)){
            accumS[ty][tx][i]=accumulator[row_i*((width)*(180))+(col_i*width)+r_i+i];
            
        }
        else{
            accumS[ty][tx][i]=0;
        }
    }
    __syncthreads();
    //Perform supression
    unsigned long long current;
    int lend;
    int jend=ty+TILEWIDTH;
    int kend=tx+TILEWIDTH;
    if(ty < TILEWIDTH && tx< TILEWIDTH){
        for(int i=0;i<TILEWIDTH;i++){
            current=accumS[ty+RADIUS][tx+RADIUS][i+RADIUS];
            for(int j=ty;j<jend;j++){
                for(int k=tx;k<kend;k++){
                    lend=i+TILEWIDTH;
                    for(int l=i;l<lend;l++){
                        if((accumS[j][k][l]>=current)&&((j!=ty+RADIUS)||(k!=tx+RADIUS)||(l!=i+RADIUS))){
                            current=0;
                            }
                        }
                    }
                }
                if(row_o<180&&col_o<180&&r_o+i<width){
                    d_output[row_o*((width)*(180))+(col_o*width)+r_o+i]=current;
        }
        }
    }
    
}




//"Main" of third optimized version
unsigned long long* optimized_compute_on_device3(int winnerCount,point* h_points, long pointCount){
    long bytes=pointCount*sizeof(point);
    long accumBytes=2*92*180*180*sizeof(unsigned long long);
    long winnerBytes=winnerCount*sizeof(unsigned long long);

    //Initialize device input vectors
    point* d_points;


	//Initialize output matrices
    unsigned long long* d_accumulator,*h_accumulator,*d_output, *d_winners, *h_winners, *h_output;

    //Allocate memory
    h_accumulator=new unsigned long long[92*180*360]();
    h_output=new unsigned long long[92*180*360]();
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
        optimizedHough3<<<gridSize,blockSize>>>(d_accumulator,pointCount);
        startIndex+=pointCount;
    }
    
    //BLOCKWIDTH=16 so 16*16 threads in a block
	dim3 dimBlock(BLOCKWIDTH,BLOCKWIDTH);
	//Tilesize=14 so width/14 * height/14 blocks in a grid.
    dim3 dimGrid(ceil((double)180/TILEWIDTH),ceil((double)180/TILEWIDTH),ceil((double)184/TILEWIDTH));
    optimizedSupression<<<dimGrid,dimBlock>>>(d_accumulator,2*92,180,180,d_output);

    gridSize=ceil((double)(2*92*180*180)/(double)1024);
    naiveDetermineWinners<<<gridSize,blockSize>>>(winnerCount,d_output,2*92,180,180,d_winners);

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    printf("GPU Version Time elapsed: %f ms\n",time);




	//Copy output to host and deallocate device memory.
    cudaMemcpy(h_winners,d_winners,winnerBytes,cudaMemcpyDeviceToHost);
	cudaFree(d_points);
    cudaFree(d_accumulator);
    cudaFree(d_output);
    cudaFree(d_winners);
    delete[] h_accumulator;
    delete[] h_output;
	return h_winners;
}
