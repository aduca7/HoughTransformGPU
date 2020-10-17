#include "project.h"
using namespace std;


//Prints plane in formatted fashion
void printPlane(int i){
    int width=2*92;
    int height=180;
    int depth=180;
    int rho=(i%width)-92;
    int theta=(i/width)%height;
    int phi=(i/(width*height))%depth;
    cout<<"("<<phi<<", "<<theta<<", "<<rho<<")"<<endl;
}
//Calculate rho,phi,theta for a given index
plane makePlane(int i){
    int width=2*92;
    int height=180;
    int depth=180;
    int rho=(i%width)-92;
    int theta=(i/width)%height;
    int phi=(i/(width*height))%depth;
    return {rho,theta,phi};

}
//Make ply file based on points
void makePly(point* points,int pointCount){
    int index;
    ofstream ply("output.ply");
    ply<<"ply"<<endl<<"format ascii 1.0"<<endl;
    ply<<"element vertex "<<pointCount/100<<endl;
    ply<<"property float x"<<endl<<"property float y"<<endl<<"property float z"<<endl;
    ply<<"property uchar red"<<endl<<"property uchar green"<<endl<<"property uchar blue"<<endl;
    ply<<"end_header";
    //Take every hundreth point so file is not too enormous
    for(int i=0;i<pointCount;i++){
        index=i*100;
        ply<<endl<<points[index].x<<" "<<points[index].y<<" "<<points[index].z<<" 255"<<" 255"<<" 255";
    }
    ply.close();
    

}

//Make coloredPLY file to show planes, 
void makeColoredPly(point* points,int pointCount, vector<plane> planes,vector<color> colors,int winnerCount){
    double conversion=M_PI/180;
    bool colored;
    for(int i=0;i<winnerCount;i++){
        cout<<planes[i].phi<<" "<<planes[i].theta<<" "<<planes[i].rho<<endl;
        cout<<16*colors[i].r<<" "<<16*colors[i].g<<" "<<16*colors[i].b<<endl;
    }
    int rho,phi,theta,j;
    point p;
    int index;
    ofstream ply("output.ply");
    ply<<"ply"<<endl<<"format ascii 1.0"<<endl;
    ply<<"element vertex "<<pointCount/100<<endl;
    ply<<"property float x"<<endl<<"property float y"<<endl<<"property float z"<<endl;
    ply<<"property uchar red"<<endl<<"property uchar green"<<endl<<"property uchar blue"<<endl;
    ply<<"end_header";
    //Calculate all points that are a part of each plane and color them accordingly
    for(int i=0;i<pointCount;i++){
        colored=false;
        index=i*100;
        point p=points[index];
        for(j=0;j<winnerCount;j++){
            theta=planes[j].theta;
            phi=planes[j].phi;
            rho=(int)floor(p.x*sin(phi*conversion)*cos(theta*conversion)
                    +p.y*sin(phi*conversion)*sin(theta*conversion)
                    +p.z*cos(phi*conversion));
            if(abs(planes[j].rho-rho)<=1){
                colored=true;
                ply<<endl<<p.x<<" "<<p.y<<" "<<p.z<<" "<<colors[j].r*16<<" "<<colors[j].g*16<<" "<<colors[j].b*16;
                break;
            }
            }
        if(colored==false){
            ply<<endl<<p.x<<" "<<p.y<<" "<<p.z<<" 0"<<" 0"<<" 0";
        }
        

        
    }

    ply.close();
}

//Check if two accumulators are equal
//Used to test accuracy of GPU version.
void compareAccumulators(unsigned long long* accumCPU,unsigned long long*accumGPU,int size){
    bool same=true;
    int count=0;
    for(int i=0;i<size;i++){
        if(accumCPU[i]!=accumGPU[i]){
            count++;
            same=false;
            printPlane(i);
        }
    }
    if(same){
        cout<<"CPU and GPU are the same."<<endl;
        cout<<count<<endl;
    }
    else{
        cout<<"CPU and GPU are not the same."<<endl;
        cout<<count<<endl;
    }
}

//Main
int main(int argc, char* argv[]){
    //Check correct number of args
    if(argc!=3){
        cout<<"Usage: ./project <input_file> <number of points>"<<endl;
        return 1;
    }
        long pointCount;
    try{
        pointCount=stol(argv[2]);
    }
    catch(exception e){
        cout<<"Invalid number of points: "<<argv[2]<<"."<<endl;
        return 1;
    }

    //Initialize variables
    ifstream input_file;
    input_file.open(argv[1]);
    string line;

    point* points;
    points=new point[pointCount];
    
    
    long index=0;
    float x,y,z;
    istringstream iss;

    //Read in points from file
    while(index<pointCount&&getline(input_file, line)){
        iss.str(line);
        iss>>x;
        iss>>y;
        iss>>z;
        points[index]={x,y,z};
        index++;
        iss.clear();
    }
    
    input_file.close();
    int rad=4;
    int winnerCount=10  ;
    //Run all versions of algorithm and compare times
    unsigned long long* winnerCPU=CPUHough(points,pointCount,rad);
    unsigned long long* winnerGPU=naive_compute_on_device(winnerCount,points,pointCount,rad);
    unsigned long long* optimizedGPU1=optimized_compute_on_device1(winnerCount,points,pointCount,rad);
    unsigned long long* optimizedGPU2=optimized_compute_on_device2(winnerCount,points,pointCount,rad);
    unsigned long long* optimizedGPU3=optimized_compute_on_device3(winnerCount,points,pointCount);
    unsigned long long* optimizedGPU4=optimized_compute_on_device4(winnerCount,points,pointCount);
    vector<plane> planes;
    vector<color> colors;
    srand(123789);
    compareAccumulators(optimizedGPU2,optimizedGPU3,180*180*184);

    //Make ply files
    for(int i=0;i<winnerCount;i++){
        // printPlane(winnerCPU[i]);
        // printPlane(winnerGPU[i]);
        // printPlane(optimizedGPU1[i]);
        // printPlane(optimizedGPU2[i]);
        // printPlane(optimizedGPU3[i]);
        // printPlane(optimizedGPU4[i]);
        planes.push_back(makePlane(optimizedGPU4[i]));
        colors.push_back({rand()%16,rand()%16,rand()%16});
        cout<<endl;
    }
    makeColoredPly(points,pointCount,planes,colors,winnerCount);
    compareAccumulators(winnerCPU,winnerGPU,180*180*2*92);
    //Delete accumulators
    delete[] winnerCPU;
    delete[] winnerGPU;
    delete[] optimizedGPU1;
    delete[] optimizedGPU2;
    delete[] optimizedGPU3;
    delete[] optimizedGPU4;
    delete[] points;
    
    return 0;
}