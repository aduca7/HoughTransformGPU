#include "project.h"
using namespace std;




__global__ void optimizedHough4(unsigned long long* accumulator, int pointCount){
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
optimized GPU function to pick winning planes
unsigned long long* accumulator = accumulator containing votes from Hough Transform.
int winnercount = number of planes to pick
int size = number of planes in accumulator
Does reduction with < operation and outputs in different format
*/
__global__ void optimizedDetermineWinners(int winnerCount, unsigned long long* accumulator,int width, int height, int depth, winnerElement* winners){
    int tx=threadIdx.x;
    int index=(blockDim.x*blockIdx.x)+tx;
    __shared__ unsigned long long accumS[512];
    if(index<height*width*depth){
        accumS[tx]=accumulator[index];
    }
    else{
        accumS[tx]=0;
    }
    __syncthreads();
    int counter=0;
    for(int i=0;i<512;i++){
        if(accumS[i]>accumS[tx]||(tx>i&&accumS[tx]==accumS[i])){
            counter++;
        }
    }
    if(counter<winnerCount){
        winners[(blockIdx.x*winnerCount)+counter]={accumS[tx],index};
    }


}
//Same as above function but has same input and output
__global__ void odw2(int winnerCount, unsigned long long* accumulator,int width, int height, int depth, winnerElement* winners,int Ccount){
    int tx=threadIdx.x;
    int index=(blockDim.x*blockIdx.x)+tx;
    __shared__ unsigned long long accumS[512];
    int aindex;
    if(index<Ccount){
        accumS[tx]=winners[index].votes;
        aindex=winners[index].index;
        

    }
    else{
        aindex=-1;
        accumS[tx]=0;
    }

    __syncthreads();
    int counter=0;
    for(int i=0;i<512;i++){
        if(accumS[i]>accumS[tx]||(tx>i&&accumS[tx]==accumS[i])){
            counter++;
        }
    }
    __syncthreads();

    if(counter<winnerCount&&aindex!=-1){
        winners[(blockIdx.x*winnerCount)+counter]={accumS[tx],aindex};
        
    }
}


// __global__ void printer(winnerElement* winners,int Ccount){
//     if(threadIdx.x==0){
//         for(int i=0;i<Ccount;i++){
//             printf("%d %ld\n",winners[i].index,(long)winners[i].votes);
//         }
//     }
// }


unsigned long long* optimized_compute_on_device4(int winnerCount,point* h_points, long pointCount){
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
        optimizedHough4<<<gridSize,blockSize>>>(d_accumulator,pointCount);
        startIndex+=pointCount;
    }
    
    
	dim3 dimBlock(BLOCKWIDTH,BLOCKWIDTH);
	
    dim3 dimGrid(ceil((double)180/TILEWIDTH),ceil((double)180/TILEWIDTH),ceil((double)184/TILEWIDTH));
    optimizedSupression<<<dimGrid,dimBlock>>>(d_accumulator,2*92,180,180,d_output);

   


    
    cudaMalloc(&d_winners,winnerBytes);
    winnerElement* d_winnerElements;
    
    
    
    blockSize=512;
    gridSize=ceil((double)(2*92*180*180)/(double)512);
    cudaMalloc(&d_winnerElements,sizeof(winnerElement)*winnerCount*gridSize);
    optimizedDetermineWinners<<<gridSize,blockSize>>>(winnerCount,d_output,2*92,180,180,d_winnerElements);
    int Ccount=gridSize*winnerCount;
    gridSize=winnerCount*ceil((double)gridSize/(double)512);
    odw2<<<gridSize,blockSize>>>(winnerCount,d_output,184,180,180,d_winnerElements,Ccount);
    Ccount=gridSize*winnerCount;
    winnerElement *h_winnerElements=new winnerElement[Ccount]();

    cudaMemcpy(h_winnerElements,d_winnerElements,Ccount*sizeof(winnerElement),cudaMemcpyDeviceToHost);
    priority_queue<winnerElement> q;

    //printer<<<1,32>>>(d_winnerElements,Ccount);
    for(int i=0;i<Ccount;i++){
        q.push(h_winnerElements[i]);
    }
    for(int i=0;i<winnerCount;i++){
        h_winners[i]=q.top().index;
        q.pop();
    }



    

    

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    printf("GPU Version Time elapsed: %f ms\n",time);




	//Copy output to host and deallocate device memory.
    //cudaMemcpy(h_winners,d_winners,winnerBytes,cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_output,d_output,accumBytes,cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_accumulator,d_accumulator,accumBytes,cudaMemcpyDeviceToHost);

	cudaFree(d_points);
    cudaFree(d_accumulator);
    cudaFree(d_output);
    cudaFree(d_winners);
    delete[] h_accumulator;
    delete[] h_output;
    delete[] h_winnerElements;
	return h_winners;
}
