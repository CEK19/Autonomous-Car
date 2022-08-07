#include <iostream>
#include <opencv2/opencv.hpp>

#include "stdlib.h"
#include "math.h"
#include "time.h"
#include <iostream>

// #include <opencv2/imgproc.hpp>
// #include <opencv2/highgui.hpp>
using namespace cv;
using namespace std;

/*
This file is made by Duy Thinh
this is version 1.6.4.2
last update: 21/1/2022
*/

/*
                              _
                           _ooOoo_
                          o8888888o
                          88" . "88
                          (| -_- |)
                          O\  =  /O
                       ____/`---'\____
                     .'  \\|     |//  `.
                    /  \\|||  :  |||//  \
                   /  _||||| -:- |||||_  \
                   |   | \\\  -  /'| |   |
                   | \_|  `\`---'//  |_/ |
                   \  .-\__ `-. -'__/-.  /
                 ___`. .'  /--.--\  `. .'___
              ."" '<  `.___\_<|>_/___.' _> \"".
             | | :  `- \`. ;`. _/; .'/ /  .' ; |
             \  \ `-.   \_\_`. _.'_/_/  -' _.' /
   ===========`-.`___`-.__\ \___  /__.-'_.'_.-'================
                           `=--=-'         
   
   Lạy chúa trên cao, xin chúa phù hộ code con chạy không bug
   nam mô a di đà phật
*/

//Define below use for debug prupose, only yes or no.
#define debug_change_lane 0
#define debug_process_image 1

// ========================== Code part started ========================
#define pixel(f,i,c) (int)(*f.ptr(i,c))

#define BLACK_BINARY_VAL    0
#define WHITE_BINARY_VAL    1
#define BLACK_DEC_VAL       0
#define WHITE_DEC_VAL       255
#define GAP_DISTANCE        20

cv::Mat abs_sobel_thresh(const cv::Mat &img, char orient, int thresh_min, int thresh_max);
cv::Mat color_thresh_hold(const cv::Mat &rgb_img, const cv::Point2i sthresh, const cv::Point2i vthresh);
cv::Mat matPreprocessing(const cv::Mat& frame);

// ----------------------1. THRESH HOLD VARIABLE---------------------- //
const int GRAD_MIN_THRESH_HOLD  = 20; // ABLE TO CHANGE
const int GRAD_MAX_THRESH_HOLD  = 180; // ABLE TO CHANGE
const cv::Point2i S_THRESH_HOLD = cv::Point2i(80, 150); // ABLE TO CHANGE
const cv::Point2i V_THRESH_HOLD = cv::Point2i(80, 150); // ABLE TO CHANGE

cv::Mat abs_sobel_thresh(const cv::Mat &rgb_img, char orient = 'x', int thresh_min=25, int thresh_max=255){
    // STEP 1: RGB TO HLS
    const int ROW_IMG = rgb_img.rows;
    const int COL_IMG = rgb_img.cols;

    cv::Mat HLS_IMG;
    cv::cvtColor(rgb_img, HLS_IMG, cv::COLOR_RGB2HLS);

    // STEP 2: SAVE L-CHANNEL TO VARIABLE
    cv::Mat L_CHANNEL = cv::Mat(ROW_IMG, COL_IMG, CV_8U);
    for (int row = 0; row < ROW_IMG;  ++row){
        for (int col  = 0; col < COL_IMG; ++col){
            L_CHANNEL.at<uint8_t>(row, col) = HLS_IMG.at<cv::Vec3b>(row, col)[1];
        }
    }

    // STEP 3: CACULATE ABS SOBEL    
    cv::Mat grad_x_or_y;
    cv::Mat abs_grad_x_or_y;
    cv::Mat scale_sobel;
    double maxValue, minValue;
    
    if (orient == 'x'){
        cv::Sobel(L_CHANNEL, grad_x_or_y, CV_64F, 1, 0);
    }
    if (orient == 'y'){
        cv::Sobel(L_CHANNEL, grad_x_or_y, CV_64F, 0, 1);        
    }
    abs_grad_x_or_y = cv::abs(grad_x_or_y);
    cv::minMaxIdx(abs_grad_x_or_y, &minValue, &maxValue);   
    abs_grad_x_or_y.convertTo(scale_sobel, CV_64F, 255.0/maxValue);
    
    // STEP 4: RESCALE BACK TO 8 BIT INTEGER
    scale_sobel.convertTo(scale_sobel, CV_8U);

    // STEP 5: RETURN BINARY OUTPUT BETWEEN MIN & MAX THRESHHOLD
    cv::Mat BINARY_OUTPUT = ((scale_sobel >= thresh_min) & (scale_sobel <= thresh_max))/255;

    return BINARY_OUTPUT;
}

cv::Mat color_thresh_hold(const cv::Mat &rgb_img, const cv::Point2i sthresh, const cv::Point2i vthresh){

    // STEP 1: CONVERT RGB TO HLS
    const int ROW_IMG = rgb_img.rows;
    const int COL_IMG = rgb_img.cols;

    cv::Mat HLS_IMG;
    cv::cvtColor(rgb_img, HLS_IMG, cv::COLOR_RGB2HLS);

    // STEP 2: SAVE S-CHANNEL TO VARIABLE
    cv::Mat S_CHANNEL = cv::Mat(ROW_IMG, COL_IMG, CV_8U);
    for (int row = 0; row < ROW_IMG;  ++row){
        for (int col  = 0; col < COL_IMG; ++col){
            S_CHANNEL.at<uint8_t>(row, col) = HLS_IMG.at<cv::Vec3b>(row, col)[2];
        }
    }

    cv::Mat S_BINARY = ((S_CHANNEL > sthresh.x) & (S_CHANNEL <= sthresh.y))/255;

    // STEP 3: SAVE V-CHANNEL TO VARIABLE
    cv::Mat HSV_IMG;
    cv::cvtColor(rgb_img, HSV_IMG, cv::COLOR_RGB2HSV);

    cv::Mat V_CHANNEL = cv::Mat(ROW_IMG, COL_IMG, CV_8U);
    for (int row = 0; row < ROW_IMG; ++row){
        for (int col = 0; col < COL_IMG; ++col){
            V_CHANNEL.at<uint8_t>(row, col) = HSV_IMG.at<cv::Vec3b>(row, col)[2];
        }
    }

    cv::Mat V_BINARY = ((V_CHANNEL > vthresh.x) & (V_CHANNEL <= vthresh.y))/255;

    // STEP 4: COMBINE S_CHANNEL AND V_CHANNEL
    cv::Mat COMBINE_S_V_BINARY = ((S_BINARY == 1) & (V_BINARY == 1))/255;
    
    int count = 0; 
    for (int row = 0; row  < COMBINE_S_V_BINARY.rows; ++row){
        for (int col = 0; col < COMBINE_S_V_BINARY.cols; ++col){
            if(static_cast<int>(COMBINE_S_V_BINARY.at<uint8_t>(row, col)) == 1) ++count;
        }
    }
    return COMBINE_S_V_BINARY;
}

cv::Mat matPreprocessing(const cv::Mat& frame ) {
  // ----------------------PREPROCESSING IMAGE---------------------- //
  // STEP 1: GRAD X & GRAD Y (SOBEL) & COLOR_THRESH_HOLD
  cv::Mat gradX_binary        = abs_sobel_thresh(frame, 'x', GRAD_MIN_THRESH_HOLD, GRAD_MAX_THRESH_HOLD);
  cv::Mat gradY_binary        = abs_sobel_thresh(frame, 'y', GRAD_MIN_THRESH_HOLD, GRAD_MAX_THRESH_HOLD);
  cv::Mat c_binary            = color_thresh_hold(frame, S_THRESH_HOLD, V_THRESH_HOLD);

  // STEP 2: COMBINE GRAD X + GRAD Y + COLOR THRESH_HOLD TOGETHER
  cv::Mat preprocessImage     = ((gradX_binary == WHITE_BINARY_VAL) & (gradY_binary == WHITE_BINARY_VAL) | (c_binary == WHITE_BINARY_VAL));
  return preprocessImage;
}

class Ram{
  public:
  int     CSR=15;
  int     LRS=20; // number of point get  ( LRS < 50)
  int     RLS=20; // Range of Lane search
  int     sample_jump=20;
  float   Speed=0.1;

//==================== You shouldn't change any variables below ! ===========================

  int     frame_count=0;
  float   final_angular_velo=0;
  int     FSM_state=0;
  int     counter_state_1=0;
  float   output_speed = 0;
  int     lane_follow = 1;  // 1= center, 2=left, 3=right

  float   change_lane_V_angular;
  float   change_lane_clk1;
  float   change_lane_alpha;
  float   change_lane_b;
  float   change_lane_d1;
  int     change_lane_direction;
  float   change_lane_clk2;
  float   change_lane_remaning_S;

  float   Now_FPS;
  float   counter_FPS=0;
  double  now_time;
  double  previous_time;

  // ============== Parameter ========================

  float acceleration_max=0.05;
  float acceleration_ratio=0.3;
};

Ram ram;

class Lane{
public:
  int col[50]={0}; //col index
  int row[50]={0}; //row index
  bool trust[50]={0};

  void checkCSR(void){
    for(int i=1; i<ram.LRS-1;i++){
      float tmp=(col[i-1]+col[i+1])/2.0-col[i];
      //printf("check %d: %d and %d => %d \n",i,col[i],trust[i],((tmp*tmp < CSR) & trust[i]));
      trust[i]=((tmp*tmp < ram.CSR) & trust[i]);
    }
  }

};

struct LaneInfomation
{
  Lane leftLane;
  Lane rightLane;
  Lane middleLane;
};

Mat get_Trainform_matrix(){
  Point2f src_p[4];
  Point2f dst_p[4];

  src_p[0]=Point2f(470.0f, 0.0f);
  src_p[1]=Point2f(640.0f, 150.0f);
  src_p[2]=Point2f(0.0f, 150.0f);
  src_p[3]=Point2f(170.0f, 0.0f);

  dst_p[0]=Point2f(1280.0f, 0.0f);
  dst_p[1]=Point2f(512.0f, 768.0f);
  dst_p[2]=Point2f(256.0f, 768.0f);
  dst_p[3]=Point2f(-512.0f, 0.0f);

  Mat trans_matrix=getPerspectiveTransform(src_p, dst_p);
  return trans_matrix;
 }

 Mat get_Trainform_matrix(Mat inputImage){
  Point2f src_p[4];
  Point2f dst_p[4];

  // src_p[0]=Point2f(470.0f, 0.0f);
  // src_p[1]=Point2f(640.0f, 150.0f);
  // src_p[2]=Point2f(0.0f, 150.0f);
  // src_p[3]=Point2f(170.0f, 0.0f);

  // dst_p[0]=Point2f(1280.0f, 0.0f);
  // dst_p[1]=Point2f(512.0f, 768.0f);
  // dst_p[2]=Point2f(256.0f, 768.0f);
  // dst_p[3]=Point2f(-512.0f, 0.0f);

  double offsetValueWidth = 40;
  double offsetValueHeight = 80;

  src_p[0] = Point2f(350,380);
  src_p[1] = Point2f(50,768);
  src_p[2] = Point2f(inputImage.size().width - src_p[0].x, src_p[0].y);
  src_p[3] = Point2f(inputImage.size().width - src_p[1].x, src_p[1].y);

  dst_p[0] = Point2f(0, 0);
  dst_p[1] = Point2f(0, inputImage.size().height);
  dst_p[2] = Point2f(inputImage.size().width,0);
  dst_p[3] = Point2f(inputImage.size().width, inputImage.size().height);

  // cout <<" ==> "<<inputImage.size().width;


  Mat trans_matrix=getPerspectiveTransform(src_p, dst_p);
  return trans_matrix;
 }

LaneInfomation process(Mat warp){


  // Mat gray;

  // cvtColor(frame, gray, COLOR_RGB2GRAY);
  // Mat crop = gray(Range(240,480),Range(0,640));
  // Mat warp;

  // warp.create(crop.size(), crop.type());
  // warpPerspective(crop, warp, get_Trainform_matrix(),Size(768,768),INTER_LINEAR);

  // warp=warp(Range(256,768),Range(0,768));
  // GaussianBlur(warp, warp, Size(5,5), 0);

  // adaptiveThreshold(warp,warp,255,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY_INV,11,5);


  // line(warp, Point(6,408), Point(253,514),Scalar(0),7,8,0);
  // line(warp, Point(757,410), Point(518,512),Scalar(0),7,8,0);

  // Mat warp = frame;

  int find_started=0;
  Mat cut_for_sum;
  

  for (int i=448;i<512;i++){
      find_started += (int)(*warp.ptr(i,384));
    }
  if (find_started == 0 ) cut_for_sum=warp(Range(448,512),Range(128,640));
  else cut_for_sum=warp(Range(320,384),Range(128,640));
  
  Mat frame_for_draw;
  cvtColor(warp, frame_for_draw, COLOR_GRAY2RGB);

  Mat frame = warp;  //512*768
                // cut for sum :128*512

  int center=256;
  
  while (center < 512){
    int tmp=0;
    for (int i=0;i<64;i++){
      tmp=tmp+(int)(*cut_for_sum.ptr(i,center));
    }
    if (tmp > 2000) break;
    center++;
  }
  int right_start=center+128;

  center=255;
  while (center >0){
    int tmp=0;
    for (int i=0;i<64;i++){
      tmp=tmp+(int)(*cut_for_sum.ptr(i,center));
    }
    if (tmp > 2000) break;
    center--;
  }

  int left_start=center+128;
  if (debug_process_image){
    Point p1(left_start,0), p2(left_start,512);
    line(frame_for_draw, p1, p2, Scalar(255,0,0), 2, LINE_4);
    p1=Point(right_start,0);
    p2=Point(right_start,512);
    line(frame_for_draw, p1, p2, Scalar(255,0,0), 2, LINE_4);
  }

  //================================ detect started ===============================
  
  Lane left, right, mid;

  int count=0;
  int check_row=500;
  // Left check
  while (check_row > 500-ram.LRS*ram.sample_jump){
    for (int i=left_start+ram.RLS; i>left_start-ram.RLS; i--){
      // cout <<" look at "<<check_row<<" "<<i<<frame.size()<<endl;
      if (pixel(frame,check_row,i) != 0){
        left.col[count]=i;
        left.trust[count]=1;
        //rectangle(frame_for_draw, Point(i+1, check_row+1), Point(i-1,check_row-1),Scalar(0,0,255),2,8,0);
        if (i != left_start+ram.RLS) left_start=i;
        break;
      }
    }
    left.row[count]=check_row;
    count++;
    check_row-=ram.sample_jump;
  }

  count=0;
  check_row=500;

  while (check_row > 500-ram.LRS*ram.sample_jump){
    for (int i=right_start-ram.RLS; i<right_start+ram.RLS; i++){
      if (pixel(frame,check_row,i) != 0){
        right.col[count]=i;
        right.trust[count]=1;
        //rectangle(frame_for_draw, Point(i+1, check_row+1), Point(i-1,check_row-1),Scalar(0,100,255),2,8,0);
        if (i != right_start-ram.RLS)right_start=i;
        break;
      }
    }

    right.row[count]=check_row;
    count++;
    check_row-=ram.sample_jump;
  }
  
  //================================ CSR check ===============================
  left.checkCSR();
  right.checkCSR();

  for (int i=0; i<ram.LRS; i++){
    mid.row[i]=left.row[i];
    mid.col[i]=(left.col[i] + right.col[i])/2;
    mid.trust[i]=(left.trust[i] & right.trust[i]);
  }

  if (debug_process_image){
    for (int i=0; i<ram.LRS; i++){
      if (left.trust[i]){
        rectangle(frame_for_draw, Point(left.col[i]+1,left.row[i]+1), Point(left.col[i]-1,left.row[i]-1),Scalar(0,0,255),2,8,0);
      }
      else {
        rectangle(frame_for_draw, Point(left.col[i]+1,left.row[i]+1), Point(left.col[i]-1,left.row[i]-1),Scalar(0,255,0),2,8,0);
      }

      if (right.trust[i]){
        rectangle(frame_for_draw, Point(right.col[i]+1,right.row[i]+1), Point(right.col[i]-1,right.row[i]-1),Scalar(0,0,255),2,8,0);
      }
      else {
        rectangle(frame_for_draw, Point(right.col[i]+1,right.row[i]+1), Point(right.col[i]-1,right.row[i]-1),Scalar(0,255,0),2,8,0);
      }
      if (mid.trust[i]){
        rectangle(frame_for_draw, Point(mid.col[i]+1,mid.row[i]+1), Point(mid.col[i]-1,mid.row[i]-1),Scalar(200,0,255),2,8,0);
      }
      else {
        rectangle(frame_for_draw, Point(mid.col[i]+1,mid.row[i]+1), Point(mid.col[i]-1,mid.row[i]-1),Scalar(200,255,0),2,8,0);
      }
    }
  }
  int followed_index=0;
  switch(ram.lane_follow){  // 1= center, 2=left, 3=right
      case 1:
        followed_index = 384;
        break;
      case 2:
        mid = left;
        followed_index = 238;
        break;
      case 3:
        mid = right;
        followed_index = 529;
        break;
      default:
        printf("WARNING: ram.lane_follow is unknow value ! \n");
    }
  count=0;
  int final_index=0;
  for (int i=0; i<ram.LRS; i++){
    if (mid.trust[i])	{
      count++;
      final_index+=mid.col[i];
    }
  	if (count >= 5) break;
  }
  //printf(" => %5f \n",final_index/5.0);
  if (debug_process_image) imshow( "Warp", frame_for_draw);

  LaneInfomation returnLaneInfo;
  returnLaneInfo.leftLane = left;
  returnLaneInfo.middleLane = mid;
  returnLaneInfo.rightLane = right;
    return returnLaneInfo;
 }

int main(int argc, char **argv){
    clock_t start, end;
	for(int i=10; i<96;i++){
        stringstream ss;
        ss << i;
        Mat readImage = imread("D:/NDT/Cpp_ws/dataset/um_0000"+ss.str()+".png");
        // Mat readImage = imread("D:/NDT/Cpp_ws/dataset/things.jpg");
        if (readImage.size().width > 0){
            
            cout << "image readed image "<<ss.str()<<", size: "<< readImage.size() <<endl;  
            Mat ROI = readImage(Range(0,374),Range(0,1200));

            //roi size: 720 - 640
            
            resize(ROI, ROI, Size(768, 768), INTER_LINEAR);
            imshow("ROI ",ROI);
            // Mat grayImage;
            // cvtColor(ROI, grayImage, COLOR_RGB2GRAY);
            // Mat adaptiveFrame;
            // start = clock();
            // Mat nhanMethod = matPreprocessing(ROI);
            // end = clock();
            // cout <<"Nhan method time: "<<double(end - start)/ double(CLOCKS_PER_SEC)<<endl;
            // adaptiveThreshold(grayImage,adaptiveFrame,255,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY_INV,11,25);
            // start = clock();
            // cout <<"Thinh method time: "<<double(start - end)/ double(CLOCKS_PER_SEC)<<endl;
            // imshow("adaptive ",adaptiveFrame);
            // imshow("sobel ",nhanMethod);

            Mat gray = matPreprocessing(ROI);
            imshow("gray image before",gray);
            imwrite("D:/NDT/Cpp_ws/dataset/export"+ss.str()+".png",gray);
            adaptiveThreshold(gray,gray,255,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY_INV,11,25);

            // cvtColor(ROI, gray, COLOR_RGB2GRAY);
            
            imshow("gray image",gray);
            Mat warp;

            warp.create(gray.size(), gray.type());
            warpPerspective(gray, warp, get_Trainform_matrix(ROI),Size(768,768),INTER_LINEAR);

            warp=warp(Range(256,768),Range(0,768));
            // GaussianBlur(warp, warp, Size(5,5), 0);

            // adaptiveThreshold(warp,warp,255,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY_INV,11,5);

            // final image has size: 512 - 768

            // imshow("final image",warp);
            process(warp);

        } else {
            cout << "image at "<<"dataset/um_0000"+ss.str()+".png"<<" unreadable " <<endl;
        }
        
        // 
        if ( (char)27 == (char) waitKey(0) ) {
            cout << "ESC recive: program exit now!";
            break;
        } else {
            
        }
    }
	return 0;
}
