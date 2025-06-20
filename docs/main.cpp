// Evolutionary Robotics 
// Kubota Lab.


#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/time.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#include <ode/ode.h>
#include <drawstuff/drawstuff.h>


#ifdef _MSC_VER
#pragma warning(disable:4244 4305)  // for VC++, no precision loss complaints
#endif

// select correct drawing functions

#ifdef dDOUBLE
//#define dsDrawLine dsDrawLineD
#define dsDrawBox dsDrawBoxD
//#define dsDrawSphere dsDrawSphereD
#define dsDrawCylinder dsDrawCylinderD
#define dsDrawCapsule dsDrawCapsuleD
#endif


#define PI 3.14159

double rnd() // uniform random number ( 0 <= rnd() < 1.0) 
{
	return((double)(rand()%RAND_MAX)/(double)RAND_MAX);
}


double rndn()            //   normal random number 
{
	return (rnd()+rnd()+rnd()+rnd()+rnd()+rnd()+
			rnd()+rnd()+rnd()+rnd()+rnd()+rnd()-6.0);
}


double radf(double q)   // degree to radian
{
    return(q*PI/180.0);
}


double degf(double q)   // radian to degree
{
    return(q/PI*180.0);
}


#define DOF   3     // degree of freedom
#define LEG   6     // number of legs
#define TLEG 18     // Total legs
#define DLEG  6     // Dummy legs
#define ROB  25     // Number of Robots
//
static dWorldID world;
static dSpaceID space;
static dJointGroupID contactgroup;
static dJointID joint[ROB+1][TLEG+1];
static dJointID joint2[ROB+1][DLEG+1];
static dGeomID ground;
static dBodyID b_body[ROB+1][5];
static dBodyID body[ROB+1][TLEG+1];	//  body Existence
static dBodyID body2[ROB+1][DLEG+1];	//  body Existence
static dGeomID box[ROB+1][TLEG+1];     //  body Geometrical information
static dGeomID bar[ROB+1][TLEG+1];     //  Leg Information
static dGeomID bar2[ROB+1][DLEG+1];     //  Leg Information

const dReal* pos, * DR;
dReal posDBf[ROB+1][3], RDBf[ROB+1][12];        //  Body Display
dReal posDTf[ROB+1][TLEG+1][3], RDTf[ROB+1][TLEG+1][12];        //  all legs Display
dReal posDT2f[ROB+1][DLEG+1][3], RDT2f[ROB+1][DLEG+1][12];      //  root Legs Display


// Robot Body
dReal box_pos[3]	={0.0,0.0,0.5};
dReal box_length	=1.0;
double box_width	=0.4;
double box_height	=0.2;
double box_mass		=1;

// Robot Leg
double bar_length	=0.1;
double bar_width	=0.2;
double bar_height	=0.1;
double bar_mass		=0.05;
double bar_rest		=0.04;

dReal bar_pos[TLEG][3]		={
    {(box_length-bar_length)*0.5,  -box_width*0.5-bar_width*0.5-(bar_rest+bar_width),  box_pos[2]},
    {(box_length-bar_length)*0.5,  -box_width*0.5-bar_width*0.5-(bar_rest+bar_width)*2,  box_pos[2]},
    {(box_length-bar_length)*0.5,  -box_width*0.5-bar_width*0.5-(bar_rest+bar_width)*3,  box_pos[2]},
    {0.0,                       -box_width*0.5-bar_width*0.5-(bar_rest+bar_width),  box_pos[2]},
    {0.0,                       -box_width*0.5-bar_width*0.5-(bar_rest+bar_width)*2,  box_pos[2]},
    {0.0,                       -box_width*0.5-bar_width*0.5-(bar_rest+bar_width)*3,  box_pos[2]},
    {-(box_length-bar_length)*0.5, -box_width*0.5-bar_width*0.5-(bar_rest+bar_width),  box_pos[2]},
    {-(box_length-bar_length)*0.5, -box_width*0.5-bar_width*0.5-(bar_rest+bar_width)*2,  box_pos[2]},
    {-(box_length-bar_length)*0.5, -box_width*0.5-bar_width*0.5-(bar_rest+bar_width)*3,  box_pos[2]},
    {(box_length-bar_length)*0.5,   box_width*0.5+bar_width*0.5+(bar_rest+bar_width),  box_pos[2]},
    {(box_length-bar_length)*0.5,   box_width*0.5+bar_width*0.5+(bar_rest+bar_width)*2,  box_pos[2]},
    {(box_length-bar_length)*0.5,   box_width*0.5+bar_width*0.5+(bar_rest+bar_width)*3,  box_pos[2]},
    {0.0,                       box_width*0.5+bar_width*0.5+(bar_rest+bar_width),  box_pos[2]},
    {0.0,                       box_width*0.5+bar_width*0.5+(bar_rest+bar_width)*2,  box_pos[2]},
    {0.0,                       box_width*0.5+bar_width*0.5+(bar_rest+bar_width)*3,  box_pos[2]},
    {-(box_length-bar_length)*0.5,  box_width*0.5+bar_width*0.5+(bar_rest+bar_width),  box_pos[2]},
    {-(box_length-bar_length)*0.5,  box_width*0.5+bar_width*0.5+(bar_rest+bar_width)*2,  box_pos[2]},
    {-(box_length-bar_length)*0.5,  box_width*0.5+bar_width*0.5+(bar_rest+bar_width)*3,  box_pos[2]}
};


double bar_pos2[DLEG+1][3]		={
    {(box_length-bar_length)*0.5,  -box_width*0.5-bar_width*0.5,    box_pos[2]},
    {0.0,                       -box_width*0.5-bar_width*0.5,    box_pos[2]},
    {-(box_length-bar_length)*0.5, -box_width*0.5-bar_width*0.5,    box_pos[2]},
    {(box_length-bar_length)*0.5,   box_width*0.5+bar_width*0.5,    box_pos[2]},
    {0.0,                       box_width*0.5+bar_width*0.5,    box_pos[2]},
    {-(box_length-bar_length)*0.5,  box_width*0.5+bar_width*0.5,    box_pos[2]}
};


// start simulation - set viewpoint

float   xyz[3]  = {0, 0, 0},
        xyz2[3] = {0.8317,     -2.9817, 2.0};
float   hpr[3]  = {101.0000f,-27.5000f, 0.0000f};

double  gain=5.0;      //  Gain
int     robotD=0,       //  robot ID for display
        robID[ROB+1],  //  GA Host ID for Locomotion
        vel_counter,
        cIte[ROB+1],    //  continuous iterations
        Dmode=1,        //  display mode, 0:no nodes, 1:all, 2:ND, 3:Rank
        Robo_view=0,    //  robot view
        posz[ROB+1];         // normal:1;  overturn:-1

double  rp[ROB+1][3]= {{0,0,0}},
        rpp[ROB+1][3]={{0,0,0}},   // rp: robot position, rpp: previous robot position
        rv[ROB+1][3]= {{0.001,0,0}},
        rvp[ROB+1][3]={{0.001,0,0}},   // rv: robot velocity, rvp: previous robot velocity
        ra[ROB+1]={0},
        rap[ROB+1]={0},        //  ra: robot moving angle, rap: previous robot moving angle,
        rd[ROB+1]={0},
        rdp[ROB+1]={0},          //  rd: moving direction angle
        rl[ROB+1]={0.001},
        rlp[ROB+1]={0.001},      //  rl: moving length
        rr[ROB+1][3]= {{1,0,0}},
        rrp[ROB+1][3]={{0,0,0}};   // rr: robot posture,  rpp: previous robot posture




#define GAN    100  // host population size
#define GAV     10  // virus population size
#define GAL     10  // Chromosome length, Motion sequences
#define TRL    500  //  number of trials (iterations)

int     ERmode=0,       //  ER mode, 0: all, 1:Forward, 2:Left, 2:Right
        iteration=0,      // iterations
        samstep=20,      // sampling steps for feedback control
        timesmax=20,    // max time sequences
        times;        // time sequence (motion sequence)

dReal   qmin[DOF+1]=    {  -45,   0,   0},  // Min Angle for target motion   
        qrange[DOF+1]=  {   90,  60,  60},  // Range  for target motion      
        qang[ROB+1][LEG+1][DOF+1],                //  current joint angle
        tang[ROB+1][LEG+1][DOF+1],                //  target joint angle
        qinit[DOF+1]=   {  0, 45, 45};  // Init Angle  for target motion

dReal   tvirus[GAV+2][DOF+1],           // a set of joint angles
        thost[GAN+2][GAL+1][2][DOF+1],  // a sequence of joint angles
        bfith[TRL+1][10],   // best fitness, [trial] 0:Sum, 1:Left, 2:Forward, 3:Right
        cfith[TRL+1][10],   // current fitness, [trial] 0:Sum, 1:Left, 2:Forward, 3:Right
        fith[GAN+1][10],    // fitness of a host [host ID] 0:Sum, 1:Left, 2:Forward, 3:Right, 4:dx, 5:dy
        fitv[GAV+1][10];    // fitness of a virus  [virus ID] 0:Forward, 1:Left, 2:Right

int     gaID[GAN+1],        //  Robot ID to
        thostl[GAN+1],      // host length, the numbe of postures in 1 trial
        bhostl[TRL+1][10],  // best host length, the numbe of postures in 1 trial
        chostl[TRL+1],      // current host length, the numbe of postures in 1 trial
        gar[GAN+1],         // ID to ranking
        raID[GAN+1],        // ranking to ID
        gand[GAN+1][10],    // ID non-dominated rank
        gac[GAN+1],         //  direction class, 1:L, 2:F, 3:R
        gak,                //  initial k-th individual
        gai,                // host ID for simulation
        gaj;                // posture sequence ID




#pragma mark -
#pragma mark GNG

// GNG

#define MN      5    //  Number of GNG
#define NN   1000    //  Maximal nunber of Neurons
#define IN     10    //  Input Dim
#define TD  10000    //  Teaching Data
#define HT  10000    //  history

typedef struct {
    int ngn;                    //  number of neurons of GNG
    int maxNeuron;              //  Max Number of Neurons
    int inp;                   //  Input Dim.
    int necSum[NN+1];           //  the number of connected neurons
    int nec[NN+1][NN+1];        // age and connectibity 0: not-connected, 1 or larger: age
    double neu[NN+1][IN+1];       // reference vector
    double nv[NN+1];            // active neuron ( -1: inactive, 1: activeated (>0)
                                // lager > nv_min: weighted sum of error over time)
    double dis[NN+1];         //   distance between i-th neuron and input

    //  Data-set
    int dataMax;                 // Max teaching data Number
    int dataNo;                 //  teaching data Number
    double tdata[TD+1][IN+1];   //  teaching data: (x,y,z), device ID (0-300),
                                //  year (20xx), month (1-12), day(1-31), time(0-86399),
    double aveEr[HT+1];            // average error history
    double maxEr[HT+1];            // max error history
    
    double Dsize;               //  Drawing Size (raduis of circle for display)
    
} GNG_st;        //  GNG information

int maxNeuronNo=320,       //  Max Neurons
    dataMaxNo=6400;        //  Data Max

GNG_st  GNGinfo[MN+1];      //  GNG info

float   gngnodeXY[3]={0, 0, 1.0};     //  node position for display
double  nni[IN+1],          //  Inputs to GNG
        rlearn1=0.05,       //  learning rate for the nearest node
        rlearn2=0.01,       //  learning rate for nodes connecting with the nearest node
        rdis=0.995;        //  temporal discount rate for error of nodes
        
int     gng=1,              //  the number of GNG
        gngl,               //  iteration times ??
        gngID=0,            //  the current GNG ID
        cutage=50,          //  cut age for GNG
        nnbest1,            //  nearest neuron
        nnbest2,            //  nearest neuron
        nnbest3;            //  second nearest neuron
  
//int gcollision[1000],   //  global collision counter
//    ltimes=0,   //  loop timer for GNG learning
//    htimes=0,   //  history loop timer for GNG learning
//    gngl=0,     //  for node addition frequency
//    gldata=0,   //  people movement counter for data
//    gltimes=0,   //  global loop timer
//    ghtimes=0;   //  global data history


void GNG_GID(int m) //  Selection of the Globally nearest neuron ID
{
    int g,h,k,i,j;
    double d;
    
    for (i=0; i<GNGinfo[m].ngn; i++){     //  calculate distance between i-th neuron and input
        d=0;
        if (GNGinfo[m].nv[i]>0){
            for (j=0; j<GNGinfo[m].inp; j++)
                d+=(GNGinfo[m].neu[i][j]-nni[j])*(GNGinfo[m].neu[i][j]-nni[j]);
            GNGinfo[m].dis[i]=sqrt(d);
        }
        else
            GNGinfo[m].dis[i]=1000000;
    }
    
   if (GNGinfo[m].dis[0]<GNGinfo[m].dis[1]){
        k=0;        //  nearest
        h=1;        //  2nd nearest
    }
    else{
        k=1;        //  nearest
        h=0;        //  2nd nearest
    }
    
    if (GNGinfo[m].dis[2]<GNGinfo[m].dis[k]){
        g=h;        //  3rd nearest
        h=k;        //  2nd nearest
        k=2;        //  nearest
     }
    else if (GNGinfo[m].dis[2]<GNGinfo[m].dis[h]){
        g=h;        //  3rd nearest
        h=2;        //  2nd nearest
     }
    else
        g=2;        //  3rd nearest

    
    for (i=3; i<GNGinfo[m].ngn; i++){
        if (GNGinfo[m].dis[i]<GNGinfo[m].dis[k]){
            g=h;        //  3rd nearest
            h=k;        //  2nd nearest
            k=i;        //  nearest
        }
        else if (GNGinfo[m].dis[i]<GNGinfo[m].dis[h]){
            g=h;        //  3rd nearest
            h=i;        //  2nd nearest
         }
        else if (GNGinfo[m].dis[i]<GNGinfo[m].dis[g])
            g=i;        //  3rd nearest
    }
    nnbest1=k;   // the nearest
    nnbest2=h;   // the 2nd nearest
    nnbest3=g;   // the 3rd nearest
}


void GNG_add(int m){ // add a Neuron
    int i,j,k=0,
        h,g=0;
    
    k=0;    //  select the worst neuron
    for (i=1; i<GNGinfo[m].ngn; i++)
        if (GNGinfo[m].nv[i]>GNGinfo[m].nv[k])
            k=i;
    if (GNGinfo[m].nv[k]>1.0){        //  to be updated
        h=0;    //  select the worst neuron connecting with k-th neuron
        while (GNGinfo[m].nec[k][h]==0)
            h++;

        for (j=h+1; j<GNGinfo[m].ngn; j++){
            if ((GNGinfo[m].nec[k][j]>=1) && (GNGinfo[m].nv[j]>GNGinfo[m].nv[h]))
                h=j;
        }
        
        g=0;
        while (GNGinfo[m].nv[g]>0)
            g++;
        if (g==GNGinfo[m].ngn){     //  if current no of nodes
            if (g<GNGinfo[m].maxNeuron)
                GNGinfo[m].ngn++;
            else
                return;
        }
        GNGinfo[m].nv[k]*=0.5;     //  reduce the error of parent neurons
        GNGinfo[m].nv[h]*=0.5;

//        printf("[%d] Add Neuron[%d/%d], k:%f h:%f from [%d],[%d]\n",
//               m,g,GNGinfo[m].ngn, GNGinfo[m].nv[k], GNGinfo[m].nv[h], k,h);
        for (j=0; j<GNGinfo[m].inp; j++)
            GNGinfo[m].neu[g][j]=(GNGinfo[m].neu[h][j]+GNGinfo[m].neu[k][j])*0.5;  // generate new node
        GNGinfo[m].nv[g]=(GNGinfo[m].nv[k]+GNGinfo[m].nv[h])*0.1;
        GNGinfo[m].nec[g][k]=1;        // connect with the new node
        GNGinfo[m].nec[k][g]=1;
        GNGinfo[m].nec[g][h]=1;
        GNGinfo[m].nec[h][g]=1;
        GNGinfo[m].nec[k][h]=0;        //  remove the original connection between k and h
        GNGinfo[m].nec[h][k]=0;

//        if (GNGinfo[m].type==1){        //  GCS
//            for (i=0; i<GNGinfo[m].ngn; i++){
//                if ((i!=h)&&(i!=k)&&(GNGinfo[m].nec[h][i]>=1)&&(GNGinfo[m].nec[k][i]>=1)){
//                    GNGinfo[m].nec[g][i]=1;        // connect with the nearest node
//                    GNGinfo[m].nec[i][g]=1;
//                }
//            }
//        }
    }
    printf("[%d] Add Neuron[%d/%d]\n",
           m,g,GNGinfo[m].ngn);

}


void GNG_check(int m) // remove the node if it is not connected with others
{
    int i,j,k;
    
    k=0;
    while (k==0){
        for (i=0; i<GNGinfo[m].ngn;i++){
            if (GNGinfo[m].nv[i]>0){
                k=0;
                for (j=0; j<GNGinfo[m].ngn;j++)
                    if (GNGinfo[m].nec[i][j]>0)
                    k++;
                if (k==0){
                    GNGinfo[m].nv[i]=-1;
                    for (j=0; j<GNGinfo[m].inp; j++)
                        GNGinfo[m].neu[i][j]=0;
                    for (j=0; j<GNGinfo[m].ngn; j++)
                        GNGinfo[m].nec[j][i]=0;
                    i=GNGinfo[m].ngn;
                }
            }
        }
    }
}
   
void GNG_initAll()  // initialization for GNG
{
    int m;
    
    gngID=0;            //  the current GNG ID
    
    for (m=0; m<gng; m++){
        GNGinfo[m].ngn=0;
        GNGinfo[m].maxNeuron=maxNeuronNo;   //  Max Number of Neurons
        GNGinfo[m].inp=3;           //  Input Dim.
        GNGinfo[m].dataNo=0;        //  Data
        GNGinfo[m].dataMax=dataMaxNo;    //  Max Data
    }
}


void GNG_init(int m)  // initialization for GNG after teaching data generation
{
    int h,i,j,k;
    

    for (i=0; i<GNGinfo[m].maxNeuron; i++){
        GNGinfo[m].nv[i]=-1;
        for (j=0; j<GNGinfo[m].inp; j++)
            GNGinfo[m].neu[i][j]=0;
        for (j=0; j<GNGinfo[m].maxNeuron; j++)
            GNGinfo[m].nec[i][j]=0;
    }
    
    i=0;
    j=0;
    k=0;
    while ((i==j) && (i==k)){
        i=(int)(GNGinfo[m].dataNo*rnd());
        j=(int)(GNGinfo[m].dataNo*rnd());
        k=(int)(GNGinfo[m].dataNo*rnd());
    }
    
    for (h=0; h<GNGinfo[m].inp; h++)
        GNGinfo[m].neu[0][h]=GNGinfo[m].tdata[i][h];
    printf("Init: GNG-Node[%d][0] (%4.2f, %4.2f, %4.2f)   \n",
            m,GNGinfo[m].neu[0][0],GNGinfo[m].neu[0][1],GNGinfo[m].neu[0][2]);
    for (h=0; h<GNGinfo[m].inp; h++)
        GNGinfo[m].neu[1][h]=GNGinfo[m].tdata[j][h];
    printf("Init: GNG-Node[%d][1] (%4.2f, %4.2f, %4.2f)   \n",
            m,GNGinfo[m].neu[1][0],GNGinfo[m].neu[1][1],GNGinfo[m].neu[1][2]);
    for (h=0; h<GNGinfo[m].inp; h++)
        GNGinfo[m].neu[2][h]=GNGinfo[m].tdata[k][h];
    printf("Init: GNG-Node[%d][2] (%4.2f, %4.2f, %4.2f)   \n",
            m,GNGinfo[m].neu[2][0],GNGinfo[m].neu[2][1],GNGinfo[m].neu[2][2]);

    GNGinfo[m].ngn=3;
    
    GNGinfo[m].nv[0]=1;
    GNGinfo[m].nv[1]=1;
    GNGinfo[m].nv[2]=1;

    GNGinfo[m].nec[0][1]=1;    //  connectivity
    GNGinfo[m].nec[1][0]=1;
    GNGinfo[m].nec[0][2]=1;    //  connectivity
    GNGinfo[m].nec[2][0]=1;
    GNGinfo[m].nec[2][1]=1;    //  connectivity
    GNGinfo[m].nec[1][2]=1;
    
    printf("\nGNG[%d] Initilized (%d,%d,%d)\n\n",m,i,j,k);
    
}

void drawGNG(int m)
{
    int i,j;
    float   R[12]={1.0,0,0,0, 0,1.0,0,0, 0,0,1.0,0},
            r=0.03;
    float   SposXYZ[3], Spos2XYZ[3];  //  Shpere position
    
    dsSetColorAlpha(0.0, 0.0, 1.0, 0.6);
    
    for (i=0; i<GNGinfo[m].ngn-1; i++)
        if (GNGinfo[m].nv[i]>0){
            SposXYZ[0]=GNGinfo[m].neu[i][0];
            SposXYZ[1]=GNGinfo[m].neu[i][1];
            SposXYZ[2]=GNGinfo[m].neu[i][2]+0.1;
            for (j=i+1; j<GNGinfo[m].ngn; j++){
                if (GNGinfo[m].nec[i][j]>=1){
                    Spos2XYZ[0]=GNGinfo[m].neu[j][0];
                    Spos2XYZ[1]=GNGinfo[m].neu[j][1];
                    Spos2XYZ[2]=GNGinfo[m].neu[j][2]+0.1;
                    dsDrawLine(SposXYZ,Spos2XYZ);
                }
            }
        }
    
    for (i=0; i<GNGinfo[m].ngn; i++)
        if (GNGinfo[m].nv[i]>0){
            SposXYZ[0]=GNGinfo[m].neu[i][0];
            SposXYZ[1]=GNGinfo[m].neu[i][1];
            SposXYZ[2]=GNGinfo[m].neu[i][2]+0.1;
            dsDrawSphere(SposXYZ, R, r);
        }
}



void GNG_learning(int m) //  mini-batch short-term Learning
{
    int it=1000,     //  local learninng iterations to add a node to the network
        s,
        h,k,i,j,t;
    
    rlearn1=0.05;
    rlearn2=0.01;
    
    for (t=1; t<=it; t++){
        gngl++;
        s=(int)(GNGinfo[m].dataNo*rnd());  //  random sampling for learning
        for (j=0; j<GNGinfo[m].inp; j++)
            nni[j]=GNGinfo[m].tdata[s][j];
        
        GNG_GID(m);  //  Global Nearest Node Selection
        k=nnbest1;  //  Nearest
        h=nnbest2;  //  Second Nearest
        
        //        printf("best:%d, 2nd-best:%d\n",nnbest1,nnbest2);
        GNGinfo[m].nv[k]+=GNGinfo[m].dis[k];     // vital value of a node
        for (j=0;j<GNGinfo[m].inp; j++)      //   update of reference vectors
            GNGinfo[m].neu[k][j]+=(nni[j]-GNGinfo[m].neu[k][j])*rlearn1;
        
        GNGinfo[m].nec[k][h]=1;  //  reset to 1 (age)
        GNGinfo[m].nec[h][k]=1;
        
        for (i=0; i<GNGinfo[m].ngn; i++){
            GNGinfo[m].nv[i]*=rdis;        //  temporal discount to evaluate the state
            if (GNGinfo[m].nec[k][i]>0){   //  if the nearest neuron is connected with k-th neuron
                for (j=0; j<GNGinfo[m].inp; j++){
                    GNGinfo[m].neu[i][j]+=(nni[j]-GNGinfo[m].neu[i][j])*rlearn2;
                    
                    //                                        *exp(-GNGinfo[m].dis[i]/1.0)*rlearn2;       // Distance-based Learning
                }
                GNGinfo[m].nv[i]+=GNGinfo[m].dis[i];     //  *0.5;             //  error update
                GNGinfo[m].nec[k][i]++;    //  ageing
                GNGinfo[m].nec[i][k]=GNGinfo[m].nec[k][i];
            }
        }
        if ((GNGinfo[m].ngn<GNGinfo[m].maxNeuron)&&(gngl%500==0)){   //  add neuron per every it iteration
            GNG_add(m);     // generate and add ngn-th neuron
        }
        
        for (i=0;i<GNGinfo[m].ngn-1;i++)
            for (j=i+1;j<GNGinfo[m].ngn;j++){
                if (GNGinfo[m].nec[i][j]>cutage){  // cut conneectiong if no-selection for long term
                    GNGinfo[m].nec[i][j]=0;
                    GNGinfo[m].nec[j][i]=0;
                    GNG_check(m);
                }
            }
    }
}

#pragma mark -
#pragma mark Files

void writedata()
{
    FILE *fp;
    int i,j,k;
    char tn[20]="data000.txt";
    char nmn[20]="0123456789";
    k=(int)(iteration/100);
    tn[4]=nmn[k];  
    
    if ((fp=fopen(tn,"w+"))==0){
        printf("can't create DATA file");
        exit(1);
    }
    
    for (i=0;i<=iteration;i++){
        for (j=0;j<4;j++)
            fprintf(fp,"%f\t%f\t%d\t%d\t",
                    bfith[i][j],        // best fitness
                    cfith[i][j],        // current fitness
                    bhostl[i][j],       // best host length
                    chostl[i]);         // current host length
        fprintf(fp,"\n");
    }
    
    fclose(fp);
    printf("DATA write end \n ");
}



// this is led by dSpaceCollide when two objects in space are
// potentially colliding.
static void nearCallback (void *data, dGeomID o1, dGeomID o2)
{
  int i,n;

  const int N = 10;
  dContact contact[N];
  n = dCollide (o1,o2,N,&contact[0].geom,sizeof(dContact));
  if (((ground == o1) || (ground == o2))&& (n > 0)) {
    for (i=0; i<n; i++) {
        
        contact[i].surface.mode =/*dContactSlip1 | dContactSlip2 | dContactApprox1 |*/
        dContactSoftERP | dContactSoftCFM; 
        contact[i].surface.mu = dInfinity;	//dInfinity or 0 or 1
        contact[i].surface.soft_erp = 0.9;      //  0.5
        contact[i].surface.soft_cfm = 1e-5;      //  0.3

        dJointID c = dJointCreateContact (world,contactgroup,&contact[i]);
        dJointAttach (c, dGeomGetBody(contact[i].geom.g1),
                         dGeomGetBody(contact[i].geom.g2));
    }
  }
}

static void start()
{
  dsSetViewpoint (xyz2,hpr);
  printf ("Start\n");
}


// called when a key pressed


//float   xyz2[3] = {0.8317,-2.9817,2.0};


static void command (int cmd)
{
    int i;
    switch (cmd) {
            
        case 'q':          //  write data & quit
            writedata();
            exit(1);
        
        case 'a':          //  write data & quit
            Dmode++;
            if (Dmode>=4) Dmode=0;
            if (Dmode==0)
                printf("\nDisplay No data node \n\n");
            else if (Dmode==1)
                printf("\nDisplay all data nodes \n\n");
            else if (Dmode==2)
                printf("\nDisplay rank data nodes \n\n");
            else if (Dmode==3)
                printf("\nDisplay Non-Dominated data nodes \n\n");
            break;
        
        case '1':          //  write data & quit
            printf("\nForward\n");
            ERmode=1;
            break;
        case '2':          //  write data & quit
            printf("\nLeft (Positive angle\n");
            ERmode=2;
            break;
        case '3':          //  write data & quit
            printf("\nRight (Negative angle\n");
            ERmode=3;
            break;
        case '0':          //  write data & quit
            printf("all direction");
            ERmode=0;
            break;
            
        case 32: //viewpoint
            Robo_view++;
            if (Robo_view>3) Robo_view = 0;
            
            if (Robo_view == 0) {
                xyz2[0]=xyz[0]+0.8317;
                xyz2[1]=xyz[1]-2.9817;
                xyz2[2] = 2.0;
                hpr[0] =101.0; hpr[1] = -27.5; hpr[2] =   0;
            }
            else if (Robo_view == 1) {
                for (i=0;i<2;i++)
                    xyz2[i]=xyz[i];
                xyz2[2] = 4.0;
                hpr[0] = 90; hpr[1] = -90; hpr[2] =   0;
            }
            else if (Robo_view == 2) {
                for (i=0;i<2;i++)
                    xyz2[i]=xyz[i];
                xyz2[2] = 6.0;
                hpr[0] = 90; hpr[1] = -90; hpr[2] =   0;
            }
            else {
                for (i=0;i<2;i++)
                    xyz2[i]=xyz[i];
                xyz2[2] = 15.0;
                hpr[0] = 90; hpr[1] = -90; hpr[2] =   0;
            }
            dsSetViewpoint(xyz2, hpr);
            break;
    }
}

#pragma mark -
#pragma mark VE-GA

void init_genes(int n)      //  generare new genes
{
    int i,j,m;
    thostl[n]=2+(int)(rnd()*3);
    for (m=0;m<thostl[n];m++){      // m: intermediate postures
//            printf("Host [%d][%d]\n",n,m);
        for (i=0;i<2;i++){          //   Leg: 0 right phase, 1: left phase (?)
            for (j=0;j<DOF;j++){
                thost[n][m][i][j]=qmin[j]+qrange[j]*rnd();      // Host
//                    printf("ta[%d][%d]:%f ",i,j,thost[n][m][i][j]);
            }
//                printf("\n");
        }
    }
}

void VEGA_ND()      //  non-dominated individuals
{
    int i,j;
//    int g1,g2,g3;
    
    for (i=0;i<GAN;i++)
        for (j=0;j<5;j++)
            gand[i][j]=0;
    for (i=0;i<GAN-1;i++){
        for (j=i+1;j<GAN;j++){
            if ((fith[i][4]>fith[j][4])&&(fith[i][5]>fith[j][5]))   //  left
                gand[j][1]++;   //  dominated
            if ((fith[i][4]>fith[j][4])&&(fith[i][5]<fith[j][5]))   //  right
                gand[j][3]++;   //  dominated
            if ((fith[j][4]>fith[i][4])&&(fith[j][5]>fith[i][5]))   //  left
                gand[i][1]++;   //  dominated
            if ((fith[j][4]>fith[i][4])&&(fith[j][5]<fith[i][5]))   //  right
                gand[i][3]++;   //  dominated
        }
    }
    printf(" - Npn Dominated - fin\n");
}



void VEGA_rank()
{
    int h,i,j,k;
//    int g1,g2,g3;
    
    for (i=0;i<GAN;i++){
        gac[i]=-1;      //  direction class, 1:Left, 2:Forward, 3:Right
        gar[i]=-1;      //  GA ID to total ranking
        raID[i]=-1;      //  total ranking to GA ID
    }
    for (j=0; j<GAN; j++){
        h=(j+1)%3+1;
        k=0;
        while (gac[k]!=-1)  // initial ID selection
            k++;
        for (i=k+1;i<GAN;i++)
            if (gac[i]==-1)
                if (fith[i][h]>fith[k][h])
                    k=i;
        gac[k]=h;
        gar[k]=j;
        raID[j]=k;
    }
    printf("\n");
    for (i=0;i<GAN;i++)
        printf("GA[%d/%d]C:%d,R:%d,%4.2f,  ",
               i, gaID[i], gac[i], gar[i], fith[i][gac[i]]);
    printf("\n\n");

    for (i=0;i<GAN;i++)
        printf("Rank[%d]:%d, ",
               i, raID[i]);
    printf("\n\n");
}

void VEGA_LR(int n)     //  exchange the phase // under development
{
    int j,m;
    double d;
    
    printf("Phase Chnge :%d\n", n);
    
    for (m=0; m<thostl[n]; m++){
        for (j=0;j<DOF;j++){
            if (j%3==0){
                d=thost[n][m][0][j];
                thost[n][m][0][j]=-thost[n][m][1][j];
                thost[n][m][1][j]=d;
            }
            else {
                d=thost[n][m][0][j];
                thost[n][m][0][j]=thost[n][m][1][j];
                thost[n][m][1][j]=d;
            }
        }
    }
}

void VEGA_reverse(int n)
{
    int i,j,m;
//    double d;
    
        printf("Reverse:%d\n", n);    //  moving direction
     
    for (m=0;m<thostl[n];m++){      // backup m: intermediate postures
        for (i=0; i<2; i++){          //   Leg: 0 right phase, 1: left phase (?)
            for (j=0; j<DOF; j+=3){
                thost[n][m][i][j]=-thost[n][m][i][j];
            }
        }
    }
}

void VEGA_Operators(int g1, int g2, int g3)  //  worst, better, good
{
    int i,j,k,m;
    double r,d;
    r=rnd()*0.5;
    
    thostl[g1]=thostl[g2];
    for (m=0; m<thostl[g1]; m++){  // Reproduction + elite crossover + simple mutation
        if ((rnd()<r)&&(m<thostl[g3])){
            for (i=0; i<2; i++)      //  Phase 1,2
                for (j=0; j<DOF; j++)
                    thost[g1][m][i][j]=thost[g3][m][i][j]+rndn()*qrange[j]*0.2;
        }
        else{
            for (i=0; i<2; i++)      //  Phase 1,2
                for (j=0; j<DOF; j++)
                    thost[g1][m][i][j]=thost[g2][m][i][j]+rndn()*qrange[j]*0.1;
        }
        for (i=0; i<2; i++)      //  Phase 1,2
            for (j=0; j<DOF; j++){
                if (thost[g1][m][i][j]<qmin[j])
                    thost[g1][m][i][j]=qmin[j]+rnd()*0.01;
                else if (thost[g1][m][i][j]>qmin[j]+qrange[j])
                    thost[g1][m][i][j]=qmin[j]+qrange[j]-rnd()*0.01;
            }
    }
    if ((thostl[g1]<GAL-1)&&(rnd()<0.15)){ // insertion mutation
        printf("-- random insertion mutation  --\n");
        k=(int)(thostl[g1]*rnd());
        if (k<thostl[g1]){
            for (m=thostl[g1]; m>k; m--)
                for (i=0; i<2; i++){
                    for (j=0; j<DOF; j++)
                        thost[g1][m][i][j]=thost[g1][m-1][i][j];
                }
            for (i=0; i<2; i++)         //   Leg: 0 right phase, 1: left phase (?)
                for (j=0; j<DOF; j++)
                    thost[g1][k][i][j]=qmin[j]+qrange[j]*rnd();      // Host
        }
        thostl[g1]++;
    }
    else if ((thostl[g1]>3)&&(rnd()<0.15)){ // deletion mutation
        thostl[g1]--;
        printf("-- deletion mutation  --\n");
        k=(int)(thostl[g1]*rnd());
        if (k<thostl[g1]-1){
            for (m=k; m<thostl[g1]; m++)
                for (i=0; i<2; i++)
                    for (j=0; j<DOF; j++)
                        thost[g1][m][i][j]=thost[g1][m+1][i][j];
        }
    }
    if (rnd()<0.1){
        printf("-- phase exchange mutation  --\n");
        m=(int)(thostl[g1]*rnd());
        for (j=0; j<DOF; j++){
            d=thost[g1][m][0][j];
            thost[g1][m][0][j]=thost[g1][m][1][j];
            thost[g1][m][1][j]=d;
        }
    }
    else if (rnd()<0.1){
        k=(int)(thostl[g1]*rnd());
        m=(int)(thostl[g1]*rnd());
        if (k!=m){
            printf("-- order exchange mutation  --\n");
            for (i=0; i<2; i++)
                for (j=0; j<DOF; j++){
                    d=thost[g1][m][i][j];
                    thost[g1][m][i][j]=thost[g1][k][i][j];
                    thost[g1][k][i][j]=d;
                }
        }
    }
}

void VEGA_main()
{
    int //  h,      //  ER mode, target moving direction
        i,n;
    int rk,         //  rank
        g1,g2,g3;
//    char dn[10][30]={"all", "Left Turn", "Forward", "Right Turn"};
    
    VEGA_rank();
    VEGA_ND();
//    if (ERmode==0)            //  evolutionary direction
//        h=iteration%3+1;
//    else
//        h=ERmode;
    
    rk=GAN-1;       //  worst ranking
    for (n=0; n<ROB; n++){
        if (robID[n]==-1){
            while (gaID[raID[rk]]!=-1){  // worst selection, future :(gac[g1]!=h)&&
                rk--;
//                printf("rk[%d],%d",rk,raID[rk]);
            }
            g1=raID[rk];
            g2=(int)(rnd()*ROB*0.5);
            g2=raID[g2];                //  better
            g3=(int)(rnd()*ROB);
            g3=raID[g3];                //  good
            printf("Rob[%d],g1:%d,g2:%d,g3:%d, ",
                   n,g1,g2,g3);
            VEGA_Operators(g1, g2, g3);
            
            robID[n]=g1;     //  robot ID to GA ID
            gaID[g1]=n;      //  GA ID to robot ID
        }
    }
    printf("\n\n");
    
    for (i=0;i<ROB;i++)
        printf("R[%d]:%d, ",
               i, robID[i]);        //  rob ID to GA ID
    printf("\n\n");
}

#pragma mark -
#pragma mark Locomotion Main

void loco_main()        //  Locomotion Update for multi-robot,
{
    int GAstarted=0,     // 0: not started (still init), 1: started
        h,i,j,k,n;
    double  pi=PI*0.25,  //  target angle for right and left turn
    a,      //  a: angle
    p=0,    //  outer product
    q=0,    //  inner product
    d=0,    //  distance in 2D
    e=0;    //  norm in 2D (3D posture)
    
    times++;
    gaj++;
    if (times>timesmax){
        for (n=0; n<ROB; n++){        //  evaluation, fitness calculation
            gai=robID[n];               //  GA ID in robots
            rap[n]=ra[n];                 //  posture Angle
            rdp[n]=rd[n];                 //  Direction angle
            rlp[n]=rl[n];                 //  moving Length
            for (i=0; i<3; i++){
                rpp[n][i]=rp[n][i];       //  position
                rvp[n][i]=rv[n][i];       //  velocity (delta)
                rrp[n][i]=rr[n][i];       //  posture
            }
            const dReal *pos0= dBodyGetPosition(b_body[n][0]);     // current position
            const dReal *rot0= dBodyGetRotation(b_body[n][0]);     // current posture, unit vector
            
            for (i=0; i<3; i++)
                posDBf[n][i]=pos0[i];
            for (i=0; i<12; i++)
                RDBf[n][i]=rot0[i];        //  Body Display
            for(i=0;i<DLEG;i++){
                const dReal *pos1=dBodyGetPosition(body2[n][i]);
                const dReal *rot1=dBodyGetRotation(body2[n][i]);
                for (j=0; j<3; j++)
                    posDT2f[n][i][j]=pos1[j];
                for (j=0; j<12; j++)
                    RDT2f[n][i][j]=rot1[j];
            }
            for(i=0;i<TLEG;i++){
                const dReal *pos1=dBodyGetPosition(body[n][i]);
                const dReal *rot1=dBodyGetRotation(body[n][i]);
                for (j=0; j<3; j++)
                    posDTf[n][i][j]=pos1[j];
                for (j=0; j<12; j++)
                    RDTf[n][i][j]=rot1[j];
            }
            
            if ((rot0[4]==0)&&(rot0[0]==0))
                ra[n]=0;
            else
                ra[n]=atan2(rot0[4],rot0[0]);
            a=ra[n]-rap[n];
            if (a>PI)
                a-=PI*2;
            else if (a<-PI)
                a+=PI*2;        //  posture angle change
            
            rr[n][0]=rot0[0];
            rr[n][1]=rot0[4];
            
            d=0;
            e=0;
            p=0;
            q=0;
            for (i=0; i<2; i++) {
                rp[n][i]=pos0[i];
                rv[n][i]=rp[n][i]-rpp[n][i]; //  moving length
                e+=rrp[n][i]*rrp[n][i];   //  or rr[]
                d+=rv[n][i]*rv[n][i];
                q+=rv[n][i]*rrp[n][i];    //  inner product of posture and moving direction vector
            }
            d=sqrt(d);  //  moving distance
            rl[n]=d;       //  moving length
            e=sqrt(e);  //  2D norm of 3D posture
            if ((rl[n]!=0)&&(e!=0))
                q=q/rl[n]/e;  //  cos(a)       //  positive: forward, negative: backward
            
            p=rv[n][0]*rrp[n][1]-rv[n][1]*rrp[n][0];    //  outer product
            if ((rl[n]!=0)&&(e!=0))
                p=p/rl[n]/e;  //  sin(a)
            rdp[n]=asin(p);                    //  moving direction of center
            
            gngnodeXY[0]=d*cos(a);          //  current node data
            gngnodeXY[1]=d*sin(a);
            
            fith[gai][1]=d*exp(-(a-pi)*(a-pi));     //  turn left
            fith[gai][2]=d*exp(-a*a);               //  go forwaqrd
            fith[gai][3]=d*exp(-(a+pi)*(a+pi));     //  turn right
            
            fith[gai][0]=0;
            for (i=1;i<4;i++){
                fith[gai][0]+=fith[gai][i];
                cfith[iteration][i]=fith[gai][i];   // current ftness
            }
            cfith[iteration][0]=fith[gai][0];
            chostl[iteration]=thostl[gai];          // current host length
            fith[gai][4]=d*cos(a);      //  dx
            fith[gai][5]=d*sin(a);      //  dy
            
            if (rot0[10]<-0.7)
                posz[n]=-1;
            else
                posz[n]=1;
            printf ("[%d] L: %5.3f, v(%5.3f,%5.3f), In-P:%5.3f(%5.3f), Out-P:%5.3f(%5.3f),A:%5.3f\n",
                    gai,rl[n], rv[n][0], rv[n][1], q, degf(acos(q)), p, degf(asin(p)), degf(a));
            printf ("Current f[1,L]:%5.3f, f[2,F]:%5.3f, f[3,R]:%5.3f, dxy(%5.3f,%5.3f)\n",
                    fith[gai][1],fith[gai][2],fith[gai][3],fith[gai][4],fith[gai][5]);
            
            if (iteration<GAN)
                h=iteration+1;
            else
                h=GAN;
            for (j=0;j<4;j++){
                k=0;
                for (i=0;i<h;i++){
                    if (fith[i][j]>fith[k][j])  // best selection
                        k=i;
                }
                bfith[iteration][j]=fith[k][j];
                bhostl[iteration][j]=thostl[k];
            }
            //        printf ("Best fit[0,F]:%5.3f, fit[1,L]:%5.3f, fit[2,R]:%5.3f\n",
            //                bfith[iteration][0],bfith[iteration][1],bfith[iteration][2]);
            if ((iteration>0)&&(iteration%100==0))
                writedata();
            
            if (n==robotD){
                if (Robo_view == 0) {
                    for (i=0;i<2;i++)
                        xyz[i]+=rp[n][i]-rpp[n][i];
                    xyz2[0]=xyz[0]+0.8317;
                    xyz2[1]=xyz[1]-2.9817;
                    xyz2[2] = 2.0;
                    hpr[0] =101.0; hpr[1] = -27.5; hpr[2] =   0;
                }
                else if (Robo_view == 1) {
                    for (i=0;i<2;i++){
                        xyz[i]+=rp[n][i]-rpp[n][i];
                        xyz2[i]=xyz[i];
                    }
                    xyz2[2] = 4.0;
                    hpr[0] = 90; hpr[1] = -90; hpr[2] =   0;
                }
                else if (Robo_view == 2) {
                    for (i=0;i<2;i++){
                        xyz[i]+=rp[n][i]-rpp[n][i];
                        xyz2[i]=xyz[i];
                    }
                    xyz2[2] = 6.0;
                    hpr[0] = 90; hpr[1] = -90; hpr[2] =   0;
                }
                else {
                    for (i=0;i<2;i++){
                        xyz[i]+=rp[n][i]-rpp[n][i];
                        xyz2[i]=xyz[i];
                    }
                    xyz2[2] = 15.0;
                    hpr[0] = 90; hpr[1] = -90; hpr[2] =   0;
                }
                dsSetViewpoint(xyz2, hpr);
            }
            if (q<0){       //
                cIte[n]++;
                printf("\n[%d] Reverse[%d]: In-P:%5.3f, angle:%5.3f\n\n",
                       gai, cIte[n], q, a);
                VEGA_reverse(gai);  //  reverse
            }
            //        else if (a<0){
            //            cIte[n]++;
            //            printf("\n[%d] ExchangeLR[%d]: Inner Product:%5.3f, angle:%5.3f\n\n",
            //                   gai, cIte, q, a);
            //            VEGA_LR(gai);  //  reverse
            //        }
            else {
                GNGinfo[gngID].tdata[iteration][0]=gngnodeXY[0];
                GNGinfo[gngID].tdata[iteration][1]=gngnodeXY[1];    //  teaching data
                iteration++;    // Next Generation
                if (gak<GAN){
                    gaID[gai]=-1;           //  released to the next search
                    robID[n]=gak;           //  GA ID  in Robots for initialization
                    gaID[gak]=n;      //  robot ID in GA
                    gak++;
                }
                else{
                    GAstarted=1;        //  VE-GA
                    robID[n]=-1;        //  released to the next search
                    gaID[gai]=-1;       //  released to the next search
                }
                cIte[n]=0;
            }
            if (cIte[n]>2){                //  new genes
                init_genes(gai);
                cIte[n]=0;
            }
            
            for (i=0;i<LEG;i++)
                for (j=0;j<DOF;j++)
                    if (i<3)
                        tang[n][i][j]=-radf(qinit[j]);
                    else
                        tang[n][i][j]= radf(qinit[j]);
            printf("Iterations:%d, host:%d \n", iteration, gai);
        }   //  for (n<ROB)
        if (GAstarted==1)
            VEGA_main();
        else{
            printf("\n");
            for (i=0;i<GAN;i++)
                printf("GA[%d]:%d, ",
                       i, gaID[i]);
            printf("\n\n");
        }
        gaj=-1;     //  to the next trial
        times=0;
    }
    else
        for (n=0; n<ROB; n++) {
            gai=robID[n];
            gaj=gaj%thostl[gai];
            //        printf("%d: Posture[%d][%d]\n ",times, gai, gaj);
            for (j=0;j<DOF;j++)
                for (i=0;i<LEG;i++){
                    if (j==0){
                        if (i%2==0)
                            tang[n][i][j]= radf(thost[gai][gaj][0][j]);
                        else
                            tang[n][i][j]= radf(thost[gai][gaj][1][j]);
                    }
                    else{
                        if (i%2==0){
                            if (i<3)
                                tang[n][i][j]=-radf(thost[gai][gaj][0][j]);
                            else
                                tang[n][i][j]= radf(thost[gai][gaj][0][j]);
                        }
                        else{
                            if (i<3)
                                tang[n][i][j]=-radf(thost[gai][gaj][1][j]);
                            else
                                tang[n][i][j]= radf(thost[gai][gaj][1][j]);
                        }
                    }
                }
        }
    
    for (i=0;i<LEG;i++)
        for (j=0;j<DOF;j++)
            tang[n][i][j]*=posz[n];
}


dReal   mat[10][4][4],      // homogeneous transform 
        qq[10]= {0, 0, 0, 0, 0, 0, 0},             // joint angle
        ll[10]= {0,bar_width, 0,bar_width,0,bar_width,0},             // link length
        mpos[10][4],        // position for matrix
        hpos[10][4];        // human position



void mvcal(int x)           // mat[x]*rpos[x] >> rpos[x-1]
{
    int i,j;
    dReal d;
    for (i=0;i<4;i++){
        d=0;
        for (j=0;j<4;j++)
            d+=mat[x][i][j]*mpos[x+1][j];
        mpos[x][i]=d;
    }
}



void matcalODE()       // original human arm calculation where human arm length is estimated
{
    int i,j,k;
    dReal l[10];
    
    for (i=0;i<6;i++)
        l[i]=ll[i];
        
    for (i=0;i<6;i++)
        qq[i]=radf(qq[i]);
        
    i=0;
    mat[i][0][0]=cos(qq[i]) ; mat[i][0][1]=0;  mat[i][0][2]=sin(qq[i]); mat[i][0][3]=0;
    mat[i][1][0]=0          ; mat[i][1][1]=1;  mat[i][1][2]=0         ; mat[i][1][3]=0;
    mat[i][2][0]=-sin(qq[i]); mat[i][2][1]=0;  mat[i][2][2]=cos(qq[i]); mat[i][2][3]=0;
    mat[i][3][0]=0;           mat[i][3][1]=0;  mat[i][3][2]=0;          mat[i][3][3]=1;

    i=1;
    mat[i][0][0]=1; mat[i][0][1]=0; mat[i][0][2]=0; mat[i][0][3]=0;
    mat[i][1][0]=0; mat[i][1][1]=1; mat[i][1][2]=0; mat[i][1][3]=l[i];
    mat[i][2][0]=0; mat[i][2][1]=0; mat[i][2][2]=1; mat[i][2][3]=0;
    mat[i][3][0]=0; mat[i][3][1]=0; mat[i][3][2]=0; mat[i][3][3]=1;
    
    i=2;
    mat[i][0][0]=1; mat[i][0][1]=0;          mat[i][0][2]=0;           mat[i][0][3]=0;
    mat[i][1][0]=0; mat[i][1][1]=cos(qq[i]); mat[i][1][2]=-sin(qq[i]); mat[i][1][3]=0;
    mat[i][2][0]=0; mat[i][2][1]=sin(qq[i]); mat[i][2][2]= cos(qq[i]); mat[i][2][3]=0;
    mat[i][3][0]=0; mat[i][3][1]=0;          mat[i][3][2]=0;           mat[i][3][3]=1;
    
    i=3;
    mat[i][0][0]=1; mat[i][0][1]=0; mat[i][0][2]=0; mat[i][0][3]=0;
    mat[i][1][0]=0; mat[i][1][1]=1; mat[i][1][2]=0; mat[i][1][3]=l[i];
    mat[i][2][0]=0; mat[i][2][1]=0; mat[i][2][2]=1; mat[i][2][3]=0;
    mat[i][3][0]=0; mat[i][3][1]=0; mat[i][3][2]=0; mat[i][3][3]=1;
        
    i=4;
    mat[i][0][0]=1; mat[i][0][1]=0;          mat[i][0][2]=0;           mat[i][0][3]=0;
    mat[i][1][0]=0; mat[i][1][1]=cos(qq[i]); mat[i][1][2]=-sin(qq[i]); mat[i][1][3]=0;
    mat[i][2][0]=0; mat[i][2][1]=sin(qq[i]); mat[i][2][2]= cos(qq[i]); mat[i][2][3]=0;
    mat[i][3][0]=0; mat[i][3][1]=0;          mat[i][3][2]=0;           mat[i][3][3]=1;
    
    
    // forward kinematics
    for (j=0;j<5;j++)
        for (k=0;k<6;k++)
            hpos[j][k]=0;
    
    for (j=0;j<5;j+=2){
        mpos[j+1][0]=0;   // ll[1]: Upper arm, [3]: lower arm, [5] :hand
        mpos[j+1][1]=l[j+1];
        mpos[j+1][2]=0;
        mpos[j+1][3]=1;
        for (i=j;i>=0;i--){
            mvcal(i);
        }
        hpos[j][0]=mpos[0][0];
        hpos[j][1]=mpos[0][1];
        hpos[j][2]=mpos[0][2];
//        printf(" [%d] (%f,%f,%f)\n", j, hpos[j][0],hpos[j][1],hpos[j][2]);
    }
    
//    scanf("%d",&i);
}



static void simLoop (int pause)
{
    int i,j,k,n;
    static dReal vel[TLEG+1];
    float   SposXYZ[3], an, //  robot angle
            R[12]={1.0,0,0,0, 0,1.0,0,0, 0,0,1.0,0},
            r=0.03;
    dReal box_sides[3] = {box_length,box_width,box_height};
    dReal bar_sides[3] ={ bar_length, bar_width, bar_height};
    dsSetTexture (DS_WOOD);

    //    n=gai;  // current individual ID
    
    if (!pause) {
        dSpaceCollide (space,0,&nearCallback);
        dWorldStep (world,0.01);    // standard simulation
        //        dWorldQuickStep(world,0.01);    //  quick simulation
        // remove all contact joints
        dJointGroupEmpty (contactgroup);
    }

    vel_counter++;
    if (vel_counter%samstep==0){        //  every 20 steps
        loco_main();               //  update the path
        vel_counter=0;
    }

    for (n=0; n<ROB; n++){      //  all robot control
        
        const dReal *pos0= dBodyGetPosition(b_body[n][0]);     // current position
        const dReal *rot0= dBodyGetRotation(b_body[n][0]);     // current direction
        if ((rot0[4]==0)&&(rot0[0]==0))
            an=0;
        else
            an=atan2(rot0[4],rot0[0]);
        
        SposXYZ[0]=pos0[0];
        SposXYZ[1]=pos0[1];
        SposXYZ[2]=gngnodeXY[2];
        dsSetColor (0.5,0.5,1);
        dsDrawBox (pos0,dBodyGetRotation(b_body[n][0]),box_sides);
        
        
        //        printf("V_time:%d\n",vel_counter);
        for(i=0;i<6;i++){
            for(j=0;j<3;j++){
                k=i*3+j;
                qang[n][i][j]=dJointGetHingeAngle(joint[n][k]);     //  current angle
                //                printf("ca[%d][%d]:%f ",i,j,degf(qang[i][j]));
                vel[k]=gain*(tang[n][i][j]-qang[n][i][j]);          //  error
            }
            //            printf("\n");
        }
        //        printf("\n");
        
        
        for(i=0;i<DLEG;i++) {
            if ((i==0)||(i==3))
                dsSetColor (0.8,1.0,0.8);
            else
                dsSetColor (0.5,0.5,1.0);
            dsDrawBox (dBodyGetPosition(body2[n][i]),dBodyGetRotation(body2[n][i]),bar_sides);
        }
        
        dsSetColor (0.6,0.6,1.0);
        for(i=0;i<TLEG;i++) {
            dJointSetHingeParam (joint[n][i],dParamVel,vel[i]); //  Veloocity Control
            dsDrawBox (dBodyGetPosition(body[n][i]),dBodyGetRotation(body[n][i]),bar_sides);
        }
        
        if (iteration>0){
            dsSetColorAlpha(0.5, 0.5, 1.0, 0.4);
            dsDrawBox (posDBf[n],RDBf[n],box_sides);
            for(i=0;i<DLEG;i++) {
                if ((i==0)||(i==3))
                    dsSetColorAlpha(0.8, 1.0, 0.8, 0.4);
                else
                    dsSetColorAlpha(0.5, 0.5, 1.0, 0.4);
                dsDrawBox (posDT2f[n][i],RDT2f[n][i],bar_sides);
            }
            dsSetColorAlpha(0.6, 0.6, 1.0, 0.4);
            for(i=0;i<TLEG;i++) {
                dsDrawBox (posDTf[n][i],RDTf[n][i],bar_sides);
            }
        }
        if (n==robotD){
            if (Dmode>0){
                SposXYZ[0]=pos0[0];
                SposXYZ[1]=pos0[1];
                SposXYZ[2]=gngnodeXY[2];
                dsSetColorAlpha(1.0, 1.0, 1.0, 1.0);
                dsDrawSphere(SposXYZ, R, r*2);            //  robot origin
                
                dsSetColorAlpha(1.0, 1.0, 1.0, 0.9);    //  all data
                SposXYZ[2]=gngnodeXY[2];
                for (i=0; i<iteration; i++){
                    SposXYZ[0]=pos0[0]  +GNGinfo[gngID].tdata[i][0]*cos(an)
                    -GNGinfo[gngID].tdata[i][1]*sin(an);
                    SposXYZ[1]=pos0[1]  +GNGinfo[gngID].tdata[i][0]*sin(an)
                    +GNGinfo[gngID].tdata[i][1]*cos(an);
                    dsDrawSphere(SposXYZ, R, r);
                }
                
                if (iteration>GAN){                     //  current GA data
                    if ((Dmode==1)||(Dmode==2)){
                        SposXYZ[2]=gngnodeXY[2]+0.2;
                        for (i=0; i<GAN; i++){
                            if (gac[i]==1)
                                dsSetColorAlpha(0.5, 1.0, 0, 0.8);
                            else if (gac[i]==2)
                                dsSetColorAlpha(0, 1.0, 1.0, 0.8);
                            else if (gac[i]==3)
                                dsSetColorAlpha(1.0, 0, 0.5, 0.8);
                            else
                                dsSetColorAlpha(1.0, 1.0, 1.0, 0.7);
                            SposXYZ[0]=pos0[0]  +fith[i][4]*cos(an)
                            -fith[i][5]*sin(an);
                            SposXYZ[1]=pos0[1]  +fith[i][4]*sin(an)
                            +fith[i][5]*cos(an);
                            dsDrawSphere(SposXYZ, R, r);
                        }
                    }
                    if ((Dmode==1)||(Dmode==3)){
                        SposXYZ[2]=gngnodeXY[2]+0.4;
                        for (i=0; i<GAN; i++){
                            if ((gand[i][1]==0)&&(fith[i][5]>=0))
                                dsSetColorAlpha(1.0, 0, 0, 0.8);    //  left
                            else if ((gand[i][3]==0)&&(fith[i][5]<0))
                                dsSetColorAlpha(0, 1.0, 0, 0.8);    //  right
                            else if ((gand[i][1]==1)&&(fith[i][5]>=0))
                                dsSetColorAlpha(0.5, 0, 0, 0.8);
                            else if ((gand[i][3]==1)&&(fith[i][5]<0))
                                dsSetColorAlpha(0, 0.5, 0, 0.8);
                            //                else if (gand[i][2]==2)
                            //                    dsSetColorAlpha(1.0, 0, 0.5, 0.8);
                            else
                                dsSetColorAlpha(1.0, 1.0, 1.0, 0.7);
                            SposXYZ[0]=pos0[0]  +fith[i][4]*cos(an)
                            -fith[i][5]*sin(an);
                            SposXYZ[1]=pos0[1]  +fith[i][4]*sin(an)
                            +fith[i][5]*cos(an);
                            dsDrawSphere(SposXYZ, R, r);
                        }
                    }
                }
            }
        }
    }
}


void robot_init(){      // VE-GA for motion generation
    int i,j,n;
    
    
    for (i=0; i<TRL; i++){
        for(j=0; j<3; j++){
            bfith[i][j]=0;                   // best ftness
            bhostl[i][j]=0;                  // best host length
            cfith[i][j]=0;                   // current ftness
        }
        chostl[i]=0;                  // current host length
    }

    
    for (n=0; n<GAN ;n++){        // n: Individual ID
        gaID[n]=-1;             //  robot ID
        gac[n]=-1;              //  direction class
        posz[n]=1;
        for(j=0; j<3; j++)
            fith[n][j]=0;
        init_genes(n);             //  generate new genes
    }

    // Initial Posture as a target
    iteration=0;
    for (n=0; n<ROB; n++){
        robID[n]=n;
        gaID[n]=n;
        cIte[n]=0;
        ra[n]=0;
        rap[n]=0;
        rd[n]=0;
        rdp[n]=0;
        for (i=0; i<3; i++){
            rp[n][i]=0;
            rpp[n][i]=0;
            rv[n][i]=0;
            rvp[n][i]=0;
            rr[n][i]=0;
            rrp[n][i]=0;
        }
        rl[n]=0.001;
        rlp[n]=0.001;
        rv[n][0]=0.001;
        rvp[n][0]=0.001;
        rr[n][0]=1.0;
        
        for (i=0; i<LEG; i++)
            for (j=0; j<DOF; j++)
                if (i<3)
                    tang[n][i][j]=-radf(qinit[j]);
                else
                    tang[n][i][j]= radf(qinit[j]);
    }
    gak=ROB;    //  next ga ID for initialization
    vel_counter=0;
        
    
 //   scanf("%d",&i);
    
/*    
    for (i=0;i<GAV;i++){
        fitv[i]=0;
        h=(int)(GAN*rnd());
        j=(int)(thostl[h]*rnd());
        m=(int)(2*rnd());
        for (k=0;k<DOF;k++)
            tvirus[i][k]=thost[h][j][m][k];
    }
*/
    
    times=0;       // times to take initial posture
    iteration=0;   // initialization
    gai=0;
    gaj=0;

}


int main (int argc, char **argv)
{
    int i,n;
    dMass m;
    
    double  bx=0, by=0;     //  base position
    
    srand((unsigned int)time(NULL));
	
    robot_init();
        
    
    // setup pointers to drawstuff callback functions
    dsFunctions fn;
    fn.version  = DS_VERSION;
    fn.start    = &start;
    fn.step     = &simLoop;
    fn.command  = &command;
    fn.stop     = 0;
	
    #if defined(__APPLE__) || defined(__MACOSX__)
        fn.path_to_textures = "textures";                     // Mac
    #else
        fn.path_to_textures = "../../drawstuff/textures";   // Windows
    #endif
    
    if(argc==2)
    {
        fn.path_to_textures = argv[1];
    }
    
    
    // create a world
    dInitODE();
    world = dWorldCreate();
    
    
    
    
    space = dHashSpaceCreate (0);
    contactgroup = dJointGroupCreate (0);
    dWorldSetGravity (world,0,0,-9.81);
    dWorldSetERP(world, 0.2);
    dWorldSetCFM(world, 0.00001);
    dWorldSetContactSurfaceLayer ( world, 0.05 );   // depth in the ground
    ground = dCreatePlane (space,0,0,1,0);
    
    
    // create a body
    
    for (n=0; n<ROB; n++){
        if (ROB==25){
            bx=(n%5)*8.0;
            by=(int)(n/5)*8.0;
        }
        else if (ROB==16){
            bx=(n%4)*8.0;
            by=(int)(n/4)*8.0;
        }
        b_body[n][0]=dBodyCreate(world);
        dBodySetPosition(b_body[n][0],bx+box_pos[0],by+box_pos[1],box_pos[2]);
        dMassSetBox(&m,1,box_length,box_width,box_height);
        dMassAdjust(&m,box_mass);
        dBodySetMass(b_body[n][0],&m);
        box[n][0]=dCreateBox(space,box_length,box_width,box_height);
        dGeomSetBody (box[n][0],b_body[n][0]);

        rp[n][0]=bx+box_pos[0];     //  robot position`
        rp[n][1]=by+box_pos[1];
        rp[n][2]=box_pos[2];

        for(i=0;i<TLEG;i++)
        {
            body[n][i]=dBodyCreate(world);
            dBodySetPosition(body[n][i],bx+bar_pos[i][0],by+bar_pos[i][1],bar_pos[i][2]);
            dMassSetBox(&m,1,bar_length,bar_width,bar_height);
            dMassAdjust(&m,bar_mass);
            dBodySetMass(body[n][i],&m);
            bar[n][i]=dCreateBox(space,bar_length,bar_width,bar_height);
            dGeomSetBody (bar[n][i],body[n][i]);
        }
        
        for(i=0;i<DLEG;i++)
        {
            body2[n][i]=dBodyCreate(world);
            dBodySetPosition(body2[n][i],bx+bar_pos2[i][0],by+bar_pos2[i][1],bar_pos2[i][2]);
            dMassSetBox(&m,1,bar_length,bar_width,bar_height);
            dMassAdjust(&m,bar_mass);
            dBodySetMass(body2[n][i],&m);
            bar2[n][i]=dCreateBox(space,bar_length,bar_width,bar_height);
            dGeomSetBody (bar2[n][i],body2[n][i]);
        }
        
        joint2[n][0]=dJointCreateHinge(world,0);
        dJointAttach(joint2[n][0],b_body[n][0],body2[n][0]);
        dJointSetHingeAnchor(joint2[n][0],bx+(box_length-bar_length)*0.5, by-box_width*0.5, box_pos[2]);
        dJointSetHingeAxis(joint2[n][0],0,1,0);
        
        joint2[n][1]=dJointCreateHinge(world,0);
        dJointAttach(joint2[n][1],b_body[n][0],body2[n][1]);
        dJointSetHingeAnchor(joint2[n][1],bx+0, by-box_width*0.5, box_pos[2]);
        dJointSetHingeAxis(joint2[n][1],0,1,0);
        
        joint2[n][2]=dJointCreateHinge(world,0);
        dJointAttach(joint2[n][2],b_body[n][0],body2[n][2]);
        dJointSetHingeAnchor(joint2[n][2],bx-(box_length-bar_length)*0.5, by-box_width*0.5, box_pos[2]);
        dJointSetHingeAxis(joint2[n][2],0,1,0);
        
        joint2[n][3]=dJointCreateHinge(world,0);
        dJointAttach(joint2[n][3],b_body[n][0],body2[n][3]);
        dJointSetHingeAnchor(joint2[n][3],bx+(box_length-bar_length)*0.5, by+box_width*0.5, box_pos[2]);
        dJointSetHingeAxis(joint2[n][3],0,1,0);
        
        joint2[n][4]=dJointCreateHinge(world,0);
        dJointAttach(joint2[n][4],b_body[n][0],body2[n][4]);
        dJointSetHingeAnchor(joint2[n][4], bx+0, by+box_width*0.5, box_pos[2]);
        dJointSetHingeAxis(joint2[n][4],0,1,0);
        
        joint2[n][5]=dJointCreateHinge(world,0);
        dJointAttach(joint2[n][5],b_body[n][0],body2[n][5]);
        dJointSetHingeAnchor(joint2[n][5],bx-(box_length-bar_length)*0.5, by+box_width*0.5, box_pos[2]);
        dJointSetHingeAxis(joint2[n][5],0,1,0);
        
        
        // Original links
        
        joint[n][0]=dJointCreateHinge(world,0);
        dJointAttach(joint[n][0],body2[n][0],body[n][0]);
        dJointSetHingeAnchor(joint[n][0],bx+(box_length-bar_length)*0.5, by-box_width*0.5-bar_width-bar_rest*0.5, box_pos[2]);
        dJointSetHingeAxis(joint[n][0],0,1,0);
        
        joint[n][3]=dJointCreateHinge(world,0);
        dJointAttach(joint[n][3],body2[n][1],body[n][3]);
        dJointSetHingeAnchor(joint[n][3],bx+0,                by-box_width*0.5-bar_width-bar_rest*0.5, box_pos[2]);
        dJointSetHingeAxis(joint[n][3],0,1,0);
        
        joint[n][6]=dJointCreateHinge(world,0);
        dJointAttach(joint[n][6],body2[n][2],body[n][6]);
        dJointSetHingeAnchor(joint[n][6],bx-(box_length-bar_length)*0.5, by-box_width*0.5-bar_width-bar_rest*0.5, box_pos[2]);
        dJointSetHingeAxis(joint[n][6],0,1,0);
        
        joint[n][9]=dJointCreateHinge(world,0);
        dJointAttach(joint[n][9],body2[n][3],body[n][9]);
        dJointSetHingeAnchor(joint[n][9],bx+(box_length-bar_length)*0.5, by+box_width*0.5+bar_width+bar_rest*0.5, box_pos[2]);
        dJointSetHingeAxis(joint[n][9],0,1,0);
        
        joint[n][12]=dJointCreateHinge(world,0);
        dJointAttach(joint[n][12],body2[n][4],body[n][12]);
        dJointSetHingeAnchor(joint[n][12],bx+0,              by+box_width*0.5+bar_width+bar_rest*0.5, box_pos[2]);
        dJointSetHingeAxis(joint[n][12],0,1,0);
        
        joint[n][15]=dJointCreateHinge(world,0);
        dJointAttach(joint[n][15],body2[n][5],body[n][15]);
        dJointSetHingeAnchor(joint[n][15],bx-(box_length-bar_length)*0.5, by+box_width*0.5+bar_width+bar_rest*0.5, box_pos[2]);
        dJointSetHingeAxis(joint[n][15],0,1,0);
        
        for(i=1;i<3;i++)
        {
            joint[n][i]=dJointCreateHinge(world,0);
            dJointAttach(joint[n][i],body[n][i-1],body[n][i]);
            dJointSetHingeAnchor(joint[n][i],bx+(box_length-bar_length)*0.5, by-box_width*0.5-bar_width*(i+1)-bar_rest*(i+0.5), box_pos[2]);
            dJointSetHingeAxis(joint[n][i],1,0,0);
        }
        
        
        for(i=4;i<6;i++)
        {
            joint[n][i]=dJointCreateHinge(world,0);
            dJointAttach(joint[n][i],body[n][i-1],body[n][i]);
            dJointSetHingeAnchor(joint[n][i],bx+0, by-box_width*0.5-bar_width*(i-2)-bar_rest*(i-2.5), box_pos[2]);
            dJointSetHingeAxis(joint[n][i],1,0,0);
        }
        
        for(i=7;i<9;i++)
        {
            joint[n][i]=dJointCreateHinge(world,0);
            dJointAttach(joint[n][i],body[n][i-1],body[n][i]);
            dJointSetHingeAnchor(joint[n][i],bx-(box_length-bar_length)*0.5, by-box_width*0.5-bar_width*(i-5)-bar_rest*(i-5.5), box_pos[2]);
            dJointSetHingeAxis(joint[n][i],1,0,0);
        }
        
        for(i=10;i<12;i++)
        {
            joint[n][i]=dJointCreateHinge(world,0);
            dJointAttach(joint[n][i],body[n][i-1],body[n][i]);
            dJointSetHingeAnchor(joint[n][i],bx+(box_length-bar_length)*0.5, by+box_width*0.5+bar_width*(i-8)+bar_rest*(i-8.5), box_pos[2]);
            dJointSetHingeAxis(joint[n][i],1,0,0);
        }
        for(i=13;i<15;i++)
        {
            joint[n][i]=dJointCreateHinge(world,0);
            dJointAttach(joint[n][i],body[n][i-1],body[n][i]);
            dJointSetHingeAnchor(joint[n][i],bx+0, by+box_width*0.5+bar_width*(i-11)+bar_rest*(i-11.5), box_pos[2]);
            dJointSetHingeAxis(joint[n][i],1,0,0);
        }
        
        for(i=16;i<18;i++)
        {
            joint[n][i]=dJointCreateHinge(world,0);
            dJointAttach(joint[n][i],body[n][i-1],body[n][i]);
            dJointSetHingeAnchor(joint[n][i], bx+(box_length-bar_length)*0.5, by+box_width*0.5+bar_width*(i-14)+bar_rest*(i-14.5), box_pos[2]);
            dJointSetHingeAxis(joint[n][i],1,0,0);
        }
        
        for(i=0;i<TLEG;i++){
            dJointSetHingeParam (joint[n][i],dParamLoStop,-PI*0.5);    //  range (Min)
            dJointSetHingeParam (joint[n][i],dParamHiStop, PI*0.5);    //  range (Max)
            dJointSetHingeParam (joint[n][i],dParamFMax,20.0);         // max force
            dJointSetHingeParam (joint[n][i],dParamFudgeFactor,0.1);
        }
        
        for(i=0;i<DLEG;i++){
            dJointSetHingeParam (joint2[n][i],dParamLoStop,0);    //  range (Min)
            dJointSetHingeParam (joint2[n][i],dParamHiStop, 0);    //  range (Max)
            dJointSetHingeParam (joint2[n][i],dParamFMax,10.0);         // max force
            dJointSetHingeParam (joint2[n][i],dParamFudgeFactor,0.1);
        }
    }
    
    if (ROB==25){
        xyz[0]=16.0;
        xyz[1]=16.0;
        robotD=12;
    }
    else if (ROB==16){
        xyz[0]=8.0;
        xyz[1]=8.0;
        robotD=5;
    }
    else if (ROB==9){
        xyz[0]=8.0;
        xyz[1]=8.0;
        robotD=4;
    }
    xyz2[0]+=xyz[0];
    xyz2[1]+=xyz[1];
    
    // run simulation
    dsSimulationLoop (argc,argv,640,480,&fn);
    
    dGeomDestroy (box[n][0]);
    for (i=0; i<TLEG; i++)
        dGeomDestroy (bar[n][i]);
    for (i=0; i<DLEG; i++)
        dGeomDestroy (bar2[n][i]);
    
    dJointGroupDestroy (contactgroup);
    dSpaceDestroy (space);
    dWorldDestroy (world);
    
    return 0;
}



