/************************************************************
 * Program: 10Visual.c                                      *  
 * Date: Oct 8, 2014                                        *
 * Latest update: Feb. 2016                                 *  
 * HL                       Status: debug                   *  
 * Version: x2.0;           Date Last Modified: Nov 5, 2014 * 
 * Purpose:                                                 *
 *    To visualize stereo vision data and lidar depth data  *  
 * g++ main.cpp -o main.o -lGL -lGLU -lglut -lm             *
 * Note: linking be sure to have included math lib          *
 *       e.g., -lm                                          *  
 * Update:                                                  * 
 * 1. OpenGL + OpenCV, image input and MAT class;           *
 * 2. Client/Server program for TCP/IP                      * 

g++ -ggdb `pkg-config --cflags opencv` -o main test.cpp `pkg-config --libs opencv` -lGL -lGLU -lglut -lm -lstdc++ 
 ************************************************************/ 
/* Note: 
 

turn camera spin = true
uncomment grid to vew the base plane



 Program: test.c 

 March 3, 2016: 
 TCP/IP data frame: 
 1. The payload of TCP/IP has the following framing format: 
    (1) DAdd    : 1 byte desitination address  
    (2) Opcode  : 1 byte 
    (3) DSAdd   : 1 byte desitination/sub address 
    (4) payload : upto 256 bytes or 1024 bytes 

 2  Opcode
    (1) PWM control related    : 0x00 - 0x0f;  
    (2) I2C interface related  : 0x10 - 0x1f; 
    (3) SPI interface related  : 0x20 - 0x2f; 
    (4) GPIO interface related : 0x30 - 0x3f;    

 Feb 2016 
 Purpose: Visualize Zeds depth image data 
 1  Use SetDot() module to display each pixels from 
    image array I(x,y); 

   void SetDot(int x, int y, int z, 
            unsigned char r_intensity, unsigned char g_intensity, 
            unsigned char b_intensity); 

   (x,y) are in the world coordinate system; 

 2 The graphics display (world coordinate system Xw-Yw-Zw) set up: 
   Z: blue, vehicle forward motion direction; 
   X: red, right-hand system, from x direction to driver side of the 
      vehicle; Z-X plane.  
   Y: green, right-hand system, the height; 
   ROI size: 256 by 256
             so 128 for positive and 128 for negative  
             128 maps to 5 meters for rplidar 
                         ??? meters for stereo vision.  

 3 I(x,y) depth image from stereo camera, 
   distance (depth) from the camera: image intensity I(x,y) equal to   
         the radius normalized to the (0,0,0) of the Xw-Yw-Zw;     
   location: (x,y) from I(x,y) mapped from the image plane (projection 
         plane) back to the world coordinate system;  
         e.g., via ITP (inverse transformation pipeline)   
          
 4 ITP
   4.1 Transformation pipeline 
       World-to-Viewer 

     |x'|   - sin(theta)              cos(theta)                               |x|   
     |y'|   - cos(theta)*cos(phi)    -sin(theta)*cos(phi)    sin(phi)          |y|  
     |z'|   - cos(theta)*sin(phi)    -sin(theta)*sin(phi)   -cos(phi)   rho    |z|  
     |1 |     0                       0                      0          1      |1|  

       Perspective projection 

       x'' = (D/z') * x' 
       y'' = (D/z') * y' 
 
   4.2 ITP (inverse transformation pipeline) 
       from (x'', y'') projection back to the world (x, y) 
       where (x,y) from I(x,y) is (x'',y'') , and world (x,y) is in Xw-Yw-Zw  

 5 Angle of FoV (Field of View), theta: [-60, 60], 120 degree. 

*/
#include <opencv2/opencv.hpp>   //for OpenCV 3.x  
#include <opencv/highgui.h>     //for OpenCV 3.x  
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <GL/gl.h>
#include <GL/glut.h>
//#include "readBMP.h"

#include <iostream>
 #include <unistd.h>

#include <cstdio>

using namespace cv;
using namespace std;

cv::VideoCapture *cap = NULL;
int width = 640;
int height = 480;
cv::Mat image;
cv::Mat temp_image;

void Display(void);
void CreateEnvironment(void);
void MakeGeometry(void);
void MakeLighting(void);
void MakeCamera(int,int,int);
void HandleKeyboard(unsigned char key,int x, int y);
void HandleSpecialKeyboard(int key,int x, int y);
void HandleMouse(int,int,int,int);
void HandleMainMenu(int);
void HandleSpeedMenu(int);
void HandleVisibility(int vis);
void HandleIdle(void);
void DrawTextXY(double,double,double,double,char *);
void GiveUsage(char *);
void SetDot(int x, int y, int z, 
            unsigned char r_intensity, unsigned char g_intensity, 
            unsigned char b_intensity); 
            //SetDot to plot 3D data 
void idle();
void reshape( int, int );

#define TRUE  1
#define FALSE 0
#define PI 3.141592653589793238462643

#define DRAFT  0
#define MEDIUM 1
#define BEST   2

int drawquality = DRAFT;
int spincamera = FALSE;
int cameradirection = 1;
double updownrotate = 60;
int ballbounce = TRUE;
double ballspeed = 2;

#define OVALID      1
#define SPHEREID    2
#define BOXID       3
#define PLANEID     4
#define TEXTID      5

FILE* rd;     //Light source array DDA 
FILE* rdp;    //Bresenham 

#define imageDimension  512  
#define circleDimension 512   

int depth_buffer[imageDimension][imageDimension];
int x_bresenham[circleDimension], y_bresenham[circleDimension];
int buffer_index, count;
int buffer_bresenham, count_bresenham, radius_file;

int window;
//Image *image;
int n,m; 
char *filename;

Mat src1, src2, hist;

/*------------------------------------------------*
    main program 
 *------------------------------------------------*/ 
int main(int argc,char **argv)
{

//  src1 = imread(argv[1] , CV_LOAD_IMAGE_COLOR);

  src1 = imread(argv[1] , IMREAD_GRAYSCALE);

if( src1.empty() )                      // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

src2 = Mat::zeros( src1.size(), src1.type() );
hist = Mat::zeros( src1.size(), src1.type() );

float intensity = 0;
int rows = src1.rows;
int cols = src1.cols;

for(int y = 0; y < rows; y++)
  { for(int x = 0; x < cols; x++)
    {
      intensity = src1.at<uchar>(y,x);
        src2.at<uchar>(y,x) = intensity;
    }
  }




   int i,j,depth,l=0, m=0;
   int mainmenu,speedmenu;

   printf("3D range data visualization\n"); 
   printf("1 depth.txt\n"); 


   //printf("read_bresenham.txt\n"); 

  printf("Camera input\n");
  int w,h;
  cap = new cv::VideoCapture(0);

  // check that video is opened
  if ( cap == NULL || !cap->isOpened() ) {
    fprintf( stderr, "could not start video capture\n" );
    return 1;
  }

  // get width and height
  w = (int) cap->get( CV_CAP_PROP_FRAME_WIDTH );
  h = (int) cap->get( CV_CAP_PROP_FRAME_HEIGHT );
  // On Linux, there is currently a bug in OpenCV that returns 
  // zero for both width and height here (at least for video from file)
  // hence the following override to global variable defaults: 
  width = w ? w : width;
  height = h ? h : height;


   for (i=1;i<argc;i++) {
      if (strstr(argv[i],"-h") != NULL) 
         GiveUsage(argv[0]);
      if (strstr(argv[i],"-q") != NULL) {
         if (i+1 >= argc)
            GiveUsage(argv[0]);
         drawquality = atoi(argv[i+1]);
         if (drawquality < DRAFT)
            drawquality = DRAFT;
         if (drawquality > BEST)
            drawquality = BEST;
         i++;
      }
   }

   /* Set things up and go */
   // initialize GLUT
   glutInit(&argc,argv);
   //glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
   glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE );
   glutInitWindowPosition( 20, 20 );
   glutInitWindowSize(600, 600);
   glutCreateWindow("OpenCV OpenGL Depth Version");

   // set up GUI callback functions
   glutDisplayFunc(Display);
   //glutDisplayFunc(display);
   glutReshapeFunc( reshape );
   //glutVisibilityFunc(HandleVisibility);
   glutKeyboardFunc(HandleKeyboard);
   glutSpecialFunc(HandleSpecialKeyboard);
   glutIdleFunc( idle );
   //glutMouseFunc(HandleMouse);
   
   CreateEnvironment();

   /* Set up menus */
   speedmenu = glutCreateMenu(HandleSpeedMenu);
   glutAddMenuEntry("Slow",1);
   glutAddMenuEntry("Medium",2);
   glutAddMenuEntry("fast",3);
   mainmenu = glutCreateMenu(HandleMainMenu);
   glutAddMenuEntry("Toggle camera spin",1);
   glutAddMenuEntry("Change Eye Location rho",2);    //HL: change the Eye Location
   glutAddSubMenu("Speed",speedmenu);
   glutAddMenuEntry("Quit",100);
   glutAttachMenu(GLUT_RIGHT_BUTTON);
   

   glutMainLoop();
   return(0);
}

/**************************************************** 
   This is where global settings are made, that is, 
   things that will not change in time 
*****************************************************/
void CreateEnvironment(void)
{
   glEnable(GL_DEPTH_TEST);

   if (drawquality == DRAFT) {
      glShadeModel(GL_FLAT);
   }

   if (drawquality == MEDIUM) {
      glShadeModel(GL_SMOOTH);
   }

   if (drawquality == BEST) {
      glEnable(GL_LINE_SMOOTH);
      glEnable(GL_POINT_SMOOTH);
      glEnable(GL_POLYGON_SMOOTH); 
      glShadeModel(GL_SMOOTH);    
      glDisable(GL_DITHER);         /* Assume RGBA capabilities */
   }

   glLineWidth(1.0);
   glPointSize(1.0);
   glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
   glFrontFace(GL_CW);
   glDisable(GL_CULL_FACE);
   glClearColor(0.0,0.0,0.0,0.0);         /* Background colour */
   glEnable(GL_COLOR_MATERIAL);
}

/**************************************************************
   The basic display callback routine
   creates the geometry, lighting, and viewing position
   In this case it rotates the camera around the scene
***************************************************************/
void Display(void)
{
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
/*
  //based on the way cv::Mat stores data, you need to flip it before displaying it
  cv::Mat tempimage;
  cv::flip(image, tempimage, 0);
  glDrawPixels( tempimage.size().width, tempimage.size().height, GL_BGR, GL_UNSIGNED_BYTE, tempimage.ptr() );
*/
  //set viewport
  //glViewport(0, 0, 250, 250);

  //set projection matrix using intrinsic camera params
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

   glPushMatrix();
   MakeCamera(0,0,0);
   //MakeLighting();
   MakeGeometry();
   glPopMatrix();

   /* glFlush(); not necessary for double buffers */
   glutSwapBuffers();

   // post the next redisplay
   glutPostRedisplay();
}

void reshape( int w, int h )
{
  // set OpenGL viewport (drawable area)
  glViewport( 0, 0, w, h );
}


void idle()
{
  // grab a frame from the camera
  (*cap) >> image;
   if (image.empty()){ 
	exit(1);
        }
   cv::resize(image, image, cv::Size(640, 480));
   //Mat imgTranslated(img.size(),img.type(),cv::Scalar::all(0));
//   image(cv::Rect(50,30,image.cols-150,image.rows-150)).copyTo(image(cv::Rect(0,0,image.cols-150,image.rows-150)));
}

/*************************************************************
   Create the geometry
**************************************************************/
void MakeGeometry(void)
{

   //based on the way cv::Mat stores data, you need to flip it before displaying it
  cv::Mat tempimage;
  cv::flip(image, tempimage, 0);
//  glBitmap(160, 120, 0, 0, 50, 50, tempimage);
//  glDrawPixels( tempimage.size().width, tempimage.size().height, GL_BGR, GL_UNSIGNED_BYTE, tempimage.ptr() );

   int i,j; 
   double radius = 0.5;
   static double theta = 0;
   GLfloat mshin1[] = {5.0};               /* For the sphere */
   GLfloat mspec1[] = {0.5,0.5,0.5,1.0};
   GLfloat mdiff1[] = {0.6,0.0,0.6,1.0};
   GLfloat mamb1[]  = {0.1,0.0,0.1,1.0};
   GLfloat mdiff2[] = {0.0,1.0,0.0,1.0};   /* color for green plane */
   GLfloat mamb2[]  = {0.0,0.2,0.0,1.0};
   GLfloat mdiff3[] = {0.5,0.5,0.5,1.0};   /* color for grey boxes */
   GLfloat mamb3[]  = {0.2,0.2,0.2,1.0};

   //define world coordinate system 
   float ORG[3] = {0,0,0};
   float XP[3] = {500,0,0}, XN[3] = {-1,0,0}; //length for the X-axis  
   float YP[3] = {0,500,0}, YN[3] = {0,-1,0};
   float ZP[3] = {0,0,500}, ZN[3] = {0,0,-1};

/*------------------------------------------------------*/
/*            Create a world cordinate system           */
/*            e.g., RGB xyz axis                        */
/*------------------------------------------------------*/
// glClear(GL_CLEAR_COLOR_BUFFER_BIT | GL _DEPTH_BUFFER_BIT);
glLineWidth (2.0);

glBegin (GL_LINES);
glColor3f (1,0,0); // X axis is red.
glVertex3fv (ORG);
glVertex3fv (XP ); 
glColor3f (0,1,0); // Y axis is green.
glVertex3fv (ORG);
glVertex3fv (YP );
glColor3f (0,0,1); // z axis is blue.
glVertex3fv (ORG);
glVertex3fv (ZP ); 
glEnd();

//*********************************************************************************************************************
cout << src1.rows << endl;
cout << src1.cols << endl;

//    SetDot(5,-5,11,r_int,g_int,0);
unsigned char r, g, b;
for (float i = 0; i < src1.rows; i++)
  {
    usleep(100);
    for (float j = 0; j < src1.cols - 100; j++)
    {
      b = src2.at<cv::Vec3b>(j, i)[0];
      g = src2.at<cv::Vec3b>(j, i)[1];
      r = src2.at<cv::Vec3b>(j, i)[2];
//      r = src2.at<uchar>(j,i)[0];
//      g = src2.at<uchar>(j,i)[1];
//      b = src2.at<uchar>(j,i)[2];

//      cout << "rows" << i/src1.rows << endl <<"cols"<< j/src1.cols; 
//      SetDot((i - src1.rows/2), (float)src2.at<uchar>(j,i)/2, (j - src1.cols/2), r, g, b);
      SetDot((i - src1.rows/2) , ((-(float)src2.at<uchar>(j,i)/2) + 128), ((j - src1.cols/2)), r, g, b);
//      SetDot((j - src1.cols/2), (i - src1.rows/2), (float)src2.at<uchar>(j,i)/2, 255, 255, 0);
    }
  }

//*********************************************************************************************************************

/*---------------------------------------------------*/
/*   Place a grey boxes around the place             */
/*---------------------------------------------------*/ 
   glLoadName(BOXID);
   glColor3f(0.75,0.75,0.75);
   if (drawquality > DRAFT) {
      glMaterialfv(GL_FRONT_AND_BACK,GL_DIFFUSE,mdiff3);
      glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT,mamb3);
   }
   glPushMatrix(); 
   glTranslatef(0,0,0);
   if (drawquality > DRAFT)
     glutSolidCube(200);  // solid cube 200x200x200  
   else
      glutWireCube(200); 
   glPopMatrix();

/*---------------------------------------------------*/
/*   draw 2D system set up                           */
/*---------------------------------------------------*/
/* note: the x-z plane is defined as usual x-y plane
         here, so if you want to have usual x-y plane
         effect, use x-z plane instead, e.g.
         x-displayed = x; 
         y-displayed = z;       ...(1) 
         (x,z) is defined in this program as (x,y) 
         where z ind, and x is function. 
*/ 
           
//----------------------------------------
//    void Draw_Grid() {
#define gridRegion 500 
#define DotSize 1.0   //this has to match up with the same dot size in SetDot() module

/*float ii; 
     for( ii = -gridRegion; ii <= gridRegion; ii += DotSize)
        {
         glBegin(GL_LINES);
            glColor3ub(50, 250, 150); //define r,g,b color 
            glVertex3f(-gridRegion, 0, ii);
            glVertex3f(gridRegion, 0, ii);
            glVertex3f(ii, 0,-gridRegion);
            glVertex3f(ii, 0, gridRegion);
         glEnd();
        } */
//} 
//Draw_Grid 

/*----------------------------------------------------*/
/*                   set ROI                          */
/* Note: Z-axis blue, X-axis red, Y-axis green        */
/*----------------------------------------------------*/
#define ROIindBound 128   // size of ROI: 256x256 
#define ROIfunBound 128   // using ind for independant variable x
                          // using fun for function, variable y 

      glColor3f(200.0f,200.0f,0.0f); //define color

      glLineWidth (4.0);  //thick line for the ROI boundary 
         glBegin(GL_LINES);
            //glColor3ub(200, 200, 0); //define r,g,b color 
            glVertex3f(-ROIfunBound,0,-ROIindBound); 
            glVertex3f(ROIfunBound,0,-ROIindBound);
            glVertex3f(ROIfunBound,0,-ROIindBound);  
            glVertex3f(ROIfunBound,0,ROIindBound);

            glVertex3f(ROIfunBound,0,ROIindBound);  
            glVertex3f(-ROIfunBound,0,ROIindBound);
            glVertex3f(-ROIfunBound,0,ROIindBound); 
            glVertex3f(-ROIfunBound,0,-ROIindBound);
         glEnd();
     glLineWidth (2.0);   //recover original line width 

/*-------------------------------------------------  
     Read stereo image data and plot 
  -------------------------------------------------*/ 

    //Set 4 landmarks  

unsigned char r_int, g_int, b_int;     
    
    r_int = 200; g_int = 200; b_int = 0; // set color  

/*----------------------------------------------
    Set 4 landmarks  
  ----------------------------------------------*/ 

    SetDot(10,10,11,r_int,g_int,0);       // 1st for red axis, 2nd for green axis  
    SetDot(-5,5,11,r_int,g_int,0);      
    SetDot(-5,-5,11,r_int,g_int,0); 
    SetDot(5,-5,11,r_int,g_int,0); 

/*---------------------------------------------- 
      set the protected region as a circle 
      which is 5 meter in radius (5 M due to rplidar 
      range 
  ---------------------------------------------*/  
/*    for(count_bresenham=0;count_bresenham<buffer_bresenham;count_bresenham++) 
    {  
      SetDot(x_bresenham[count_bresenham],y_bresenham[count_bresenham],0,
             r_int, 0, b_int); 
    } 
*/
    //printf("radius from file = %d\n", radius_file);

/*---------------------------------------------- 
      plot depth image map 
  ---------------------------------------------*/  
    //read image 
    //get image x,y dimension, e.g, 640x480 for example 
    //get image intensity I(x,y) which maps the depth 

    int imageR, imageC;  // image x,y dimension 
    int rho;             // distance 
    float alpha, temp;         // angle within FoV (fild of view)   
    float M_col;           // total number of column to map to FoV 
    //int   M_row;           // total number of rows  
    float FoV;             // FoV   
    int xWorld, yWorld;  // for visualization
 
    FoV = PI * (120.0/180.0);       // FoV   
    M_col = 32 + 1;   // total number of column to map to FoV 

/*----------------------------------------------
    Input depth image here 
    replace the following simulation code with
    the actual depth image data
  ----------------------------------------------*/ 
    //create simulation depth image  
    for (imageR=0; imageR<32; imageR++) 
    {
     for (imageC=0; imageC<32; imageC++) 
       {
       //depth_buffer[imageR][imageC] = imageC;
       depth_buffer[imageR][imageC] = 20;
       } 
}  

 /*   alpha = FoV / M_col; 
    temp = alpha; 
 
    for(imageR=0;imageR<32;imageR++) 
    {  
      for (imageC=0; imageC<32; imageC++) 
      {
       rho = depth_buffer[imageR][imageC];
       if(imageC < 32/2)
       {
       xWorld = rho * sin (temp);  
       yWorld = -1*rho * cos (temp);   
       } 
       else 
       {
       xWorld = rho * sin (temp);  
       yWorld = rho * cos (temp);   
       } 
       temp = temp + alpha; 

       SetDot(xWorld, yWorld, imageR,
             r_int, g_int, 0); 
       } 

    } 
    */
}

/*----------------------------------------------------*/
/*                       Set a dot to plot            */ 
/*----------------------------------------------------*/

void SetDot(int x, int y, int z, 
            unsigned char r_intensity, unsigned char g_intensity, 
            unsigned char b_intensity)
{

//begin of the dot plotting
//#define DotSize 5.0 

float         fx, fy, fz; 
float         fr_intensity, fg_intensity, fb_intensity; 

    fx = x; fy = y; fz = z; 
    
    glPushMatrix(); 

    glTranslatef(fx * DotSize, fz, fy * DotSize);

    //convert to float type 
    fr_intensity =  r_intensity;
    fg_intensity =  g_intensity;
    fb_intensity =  b_intensity;    
 
    glColor3f(fr_intensity, fg_intensity, fb_intensity); //define dot color
    
    glBegin(GL_QUADS);
      glVertex3f(0.0f,0.0f,0.0f);
      glVertex3f(DotSize,0.0f,0.0f);
      glVertex3f(DotSize,0.0f,DotSize); 
      glVertex3f(0.0f,0.0f,DotSize); 
    glEnd();//end drawing of points
  
    glPopMatrix(); 

//end of the dot plotting 
}

/************************************************    
   Set up the lighting environment
*************************************************/
void MakeLighting(void)
{
   GLfloat globalambient[] = {0.3,0.3,0.3,1.0};

   /* The specifications for 3 light sources */
   GLfloat pos0[] = {1.0,1.0,0.0,0.0};      /* w = 0 == infinite distance */
   GLfloat dif0[] = {0.8,0.8,0.8,1.0};

   GLfloat pos1[] = {5.0,-5.0,0.0,0.0};   /* Light from below */
   GLfloat dif1[] = {0.4,0.4,0.4,1.0};      /* Fainter */

   if (drawquality > DRAFT) {

      /* Set ambient globally, default ambient for light sources is 0 */
      glLightModelfv(GL_LIGHT_MODEL_AMBIENT,globalambient);

      glLightfv(GL_LIGHT0,GL_POSITION,pos0);
      glLightfv(GL_LIGHT0,GL_DIFFUSE,dif0);

      glLightfv(GL_LIGHT1,GL_POSITION,pos1);
      glLightfv(GL_LIGHT1,GL_DIFFUSE,dif1);

      glEnable(GL_LIGHT0);
      glEnable(GL_LIGHT1);
      glEnable(GL_LIGHTING);
   }
}

/*******************************************************
   Set up the camera
   Optionally creating a small viewport about 
   the mouse click point for object selection.
   Note: 
   1. change the camera E distance from the origin of 
      the world coordinate system, by incease rho, 
      the default set rho = 400, need to change to 
      interactive user defined way. 
********************************************************/
void MakeCamera(int pickmode,int x,int y)

{
   #define rho  400      //Eye distance from the origin of the world coordinate 

   static double theta = 0;
   GLint viewport[4];

   /* Camera setup */
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   if (pickmode == TRUE) {
      glGetIntegerv(GL_VIEWPORT,viewport); /* Get the viewport bounds */
      gluPickMatrix(x,viewport[3]-y,3.0,3.0,viewport);
   }
   gluPerspective(70.0,          /* Field of view */
                   1.0,          /* aspect ratio  */
                   0.1,1000.0);  /* near and far  */

   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
/*
   Note the rho is defined to give the eye location
*/
   gluLookAt(rho*cos(theta*PI/180)*sin(updownrotate*PI/180), //control the E distance
             rho*cos(updownrotate*PI/180),
             rho*sin(theta*PI/180)*sin(updownrotate*PI/180), 
             0.0,0.0,0.0,                                   /* Focus    */
             0.0,1.0,0.0);                                  /* Up       */
   if (spincamera)
      theta += (cameradirection * 0.2);
}

/*******************************************************
   Deal with plain key strokes
********************************************************/
void HandleKeyboard(unsigned char key,int x, int y)
{
   switch (key) {
   case 27: /* ESC */
   case 'Q':
   case 'q': exit(0); break;
   case 's':
   case 'S': spincamera = !spincamera; break;
   case 'b':
   case 'B': ballbounce = !ballbounce; break;
   }
}

/******************************************************
   Deal with special key strokes
*******************************************************/
void HandleSpecialKeyboard(int key,int x, int y)
{
   switch (key) {
   case GLUT_KEY_LEFT:  cameradirection = -1; break;
   case GLUT_KEY_RIGHT: cameradirection = 1;  break;
   case GLUT_KEY_UP:    updownrotate -= 2;  break;
   case GLUT_KEY_DOWN:  updownrotate += 2;  break;
   }
}

/******************************************************
   Handle mouse events
*******************************************************/
void HandleMouse(int button,int state,int x,int y)
{
   int i,maxselect = 100,nhits = 0;
   GLuint selectlist[100];

   if (state == GLUT_DOWN) {
      glSelectBuffer(maxselect,selectlist);
      glRenderMode(GL_SELECT);
      glInitNames();
      glPushName(-1);

      glPushMatrix();
      MakeCamera(TRUE,x,y);
      MakeGeometry();
      glPopMatrix();
      nhits = glRenderMode(GL_RENDER);

      if (button == GLUT_LEFT_BUTTON) {

      } else if (button == GLUT_MIDDLE_BUTTON) {

      } /* Right button events are passed to menu handlers */

      if (nhits == -1)
         fprintf(stderr,"Select buffer overflow\n");

      if (nhits > 0) {
         fprintf(stderr,"\tPicked %d objects: ",nhits);
         for (i=0;i<nhits;i++)
            fprintf(stderr,"%d ",selectlist[4*i+3]);
         fprintf(stderr,"\n");
      }

   }
}

/*************************************************
   Handle the main menu
**************************************************/
void HandleMainMenu(int whichone)
{
   switch (whichone) {
   case 1: spincamera = !spincamera; break;
   case 2: ballbounce = !ballbounce; break;
   case 100: exit(0); break;
   }
}

/*************************************************
   Handle the ball speed sub menu
**************************************************/
void HandleSpeedMenu(int whichone)
{
   switch (whichone) {
   case 1: ballspeed = 0.5; break;
   case 2: ballspeed = 2;   break;
   case 3: ballspeed = 10;  break;
   }
}

/************************************************
   Handle visibility
*************************************************/
void HandleVisibility(int visible)
{
   if (visible == GLUT_VISIBLE)
      glutIdleFunc(HandleIdle);
   else
      glutIdleFunc(NULL);
}

/************************************************
   On an idle event
*************************************************/
void HandleIdle(void)
{

   glutPostRedisplay();
}

/************************************************
   Draw text in the x-y plane
   The x,y,z coordinate is the bottom left corner 
   (looking down -ve z axis)
*************************************************/
void DrawTextXY(double x,double y,double z,double scale,char *s)
{
   int i;

   glPushMatrix();
   glTranslatef(x,y,z);
   glScalef(scale,scale,scale);
   for (i=0;i<strlen(s);i++)
      glutStrokeCharacter(GLUT_STROKE_ROMAN,s[i]);
   glPopMatrix();
}

/***********************************************
   Display the program usage information
************************************************/
void GiveUsage(char *cmd)
{
   fprintf(stderr,"Usage:    %s [-h] [-q n]\n",cmd);
   fprintf(stderr,"          -h   this text\n");
   fprintf(stderr,"          -q n quality, 0,1,2\n");
   fprintf(stderr,"Key Strokes and Menus:\n");
   fprintf(stderr,"           q - quit\n");
   fprintf(stderr,"           s - toggle camera spin\n");
   fprintf(stderr,"           b - toggle ball bounce\n");
   fprintf(stderr,"  left arrow - change rotation direction\n");
   fprintf(stderr," right arrow - change rotation direction\n");
   fprintf(stderr,"  down arrow - rotate camera down\n");
   fprintf(stderr,"    up arrow - rotate camera up\n");
   exit(-1);
}
