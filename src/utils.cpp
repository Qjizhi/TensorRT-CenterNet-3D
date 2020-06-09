//
// Created by cao on 19-10-26.
//
#include "utils.h"
#include "ctdetConfig.h"
#include <sstream>
#define PI acos(-1)

// calibration matrix
float P[3][4] = {707.0493, 0., 604.0814, 45.75831, 0., 707.0493, 180.5066, -0.3454157, 0., 0., 1., 0.004981016};

float get_alpha(Rotation rot)
{
  int idx = rot.ang2 > rot.ang6?1:0;
  float alpha1 = atan2(rot.ang3, rot.ang4) + (-0.5 * PI);
  float alpha2 = atan2(rot.ang7, rot.ang8) + (0.5 * PI);

  return alpha1 * idx + alpha2 * (1 - idx);
}


void unproject_2d_to_3d(float dem_l, float pt_2d[2],float depth, float P[][4], float pt_3d[3])
{
  float z = depth - P[2][3];
  float x = (pt_2d[0] * depth - P[0][3] - P[0][2] * z) / P[0][0];
  float y = (pt_2d[1] * depth - P[1][3] - P[1][2] * z) / P[1][1];
  pt_3d[0] = x;
  pt_3d[1] = y;
  pt_3d[2] = z + dem_l/2;
}


float alpha2rot_y(float alpha, float x, float cx, float fx)
{
    float rot_y = alpha + atan2(x - cx, fx);
    if(rot_y > PI)   rot_y -= 2 * PI;
    if(rot_y < -PI)  rot_y += 2 * PI;
    return rot_y;
}


void compute_box_3d(float dim[3], float location[3], float rotation_y, float coordinates[][3])
{
    // dim: 3
    // location: 3
    // rotation_y: 1
    // coordinates 8 x 3
    float c = cos(rotation_y); 
    float s = sin(rotation_y);
    float R[3][3] = {c, 0, s, 0, 1, 0, -s, 0, c};
    float l = dim[2];
    float w = dim[1];
    float h = dim[0];

    float coordinates_ori[8][3] = {l/2, h/2, w/2, l/2, h/2, -w/2, -l/2, h/2, -w/2, -l/2, h/2, w/2, l/2, -h/2, w/2, l/2, -h/2, -w/2, -l/2, -h/2, -w/2, -l/2, -h/2, w/2};
    for (int m = 0; m < 8; m++)
    
    {
        for (int n = 0; n < 3; n++)
        {
            coordinates[m][n] = 0;
            for (int k = 0; k < 3; k++)
            {
                coordinates[m][n] += coordinates_ori[m][k] * R[k][n];
            }
        }
    }
    for (int i = 0; i < 8; i++) 
    {	
        coordinates[i][0] +=  location[0];
        coordinates[i][1] +=  location[1];
        coordinates[i][2] +=  location[2];
	}
}


void project_to_image(float pts_3d[][8],float P[][4], float pts_2d[][2])
{
    // pts_3d: 4 x 8
    // P: 3 x 4
    // pts_2d: 8 x 2
    float pts_2d_temp[3][8];
    for (int m = 0; m < 3; m++)
    {
        for (int n = 0; n < 8; n++)
        {
            pts_2d_temp[m][n] = 0;
            for (int k = 0; k < 4; k++)
            {
                pts_2d_temp[m][n] +=  P[m][k] * pts_3d[k][n];
            }
        }
    }
    for (int i = 0; i < 8; i++) 
    {	
        pts_2d[i][0] =  pts_2d_temp[0][i]/pts_2d_temp[2][i];
        pts_2d[i][1] =  pts_2d_temp[1][i]/pts_2d_temp[2][i];
	}    
    for (int i = 0; i < 8; i++) 
    {
        std::cout<<"i: "<<i<<" x: "<<pts_2d[i][0]<<" y: "<< pts_2d[i][1]<< std::endl;
    }  
    std::cout<<std::endl;
}



// std::vector<float> prepareImage(cv::Mat& img, const bool& forwardFace)
std::vector<float> prepareImage(cv::Mat& img, const int& forwardFace)
{


    int channel = ctdet::channel ;
    int input_w = ctdet::input_w;
    int input_h = ctdet::input_h;
    float scale = cv::min(float(input_w)/img.cols,float(input_h)/img.rows);
    auto scaleSize = cv::Size(img.cols * scale,img.rows * scale);

    cv::Mat resized;
    cv::resize(img, resized,scaleSize,0,0);


    cv::Mat cropped = cv::Mat::zeros(input_h,input_w,CV_8UC3);
    // std::cout<<"input_h: "<<input_h<<" input_w: "<<input_w<<std::endl;
    cv::Rect rect((input_w- scaleSize.width)/2, (input_h-scaleSize.height)/2, scaleSize.width,scaleSize.height);

    resized.copyTo(cropped(rect));


    cv::Mat img_float;
    // if(forwardFace)
    //     cropped.convertTo(img_float, CV_32FC3, 1.);
    // else
    //     cropped.convertTo(img_float, CV_32FC3,1./255.);

    if(forwardFace==1)
        cropped.convertTo(img_float, CV_32FC3, 1.);
    else
        cropped.convertTo(img_float, CV_32FC3,1./255.);

    //HWC TO CHW
    std::vector<cv::Mat> input_channels(channel);
    cv::split(img_float, input_channels);

    // normalize
    std::vector<float> result(input_h*input_w*channel);
    auto data = result.data();
    int channelLength = input_h * input_w;
    for (int i = 0; i < channel; ++i) {
        cv::Mat normed_channel = (input_channels[i]-ctdet::mean[i])/ctdet::std[i];
        memcpy(data,normed_channel.data,channelLength*sizeof(float));
        data += channelLength;
    }
    return result;
}

// void postProcess(std::vector<Detection> & result,const cv::Mat& img, const bool& forwardFace)
void postProcess(std::vector<Detection> & result,const cv::Mat& img, const int& forwardFace)
{
    using namespace cv;
    int mark;
    int input_w = ctdet::input_w;
    int input_h = ctdet::input_h;
    float scale = min(float(input_w)/img.cols,float(input_h)/img.rows);
    float dx = (input_w - scale * img.cols) / 2;
    float dy = (input_h - scale * img.rows) / 2;
    for(auto&item:result)
    {
        float x1 = (item.bbox.x1 - dx) / scale ;
        float y1 = (item.bbox.y1 - dy) / scale ;
        float x2 = (item.bbox.x2 - dx) / scale ;
        float y2 = (item.bbox.y2 - dy) / scale ;
        x1 = (x1 > 0 ) ? x1 : 0 ;
        y1 = (y1 > 0 ) ? y1 : 0 ;
        x2 = (x2 < img.cols  ) ? x2 : img.cols - 1 ;
        y2 = (y2 < img.rows ) ? y2  : img.rows - 1 ;
        item.bbox.x1  = x1 ;
        item.bbox.y1  = y1 ;
        item.bbox.x2  = x2 ;
        item.bbox.y2  = y2 ;
        // if(forwardFace){
        if(forwardFace==1){
            float x,y;
            for(mark=0;mark<5; ++mark ){
                 x = (item.marks[mark].x - dx) / scale ;
                 y = (item.marks[mark].y - dy) / scale ;
                 x = (x > 0 ) ? x : 0 ;
                 y = (y > 0 ) ? y : 0 ;
                 x = (x < img.cols  ) ? x : img.cols - 1 ;
                 y = (y < img.rows ) ? y  : img.rows - 1 ;
                item.marks[mark].x = x ;
                item.marks[mark].y = y ;
            }
        }
        if(forwardFace == 2)
        {
            // calculate center of 3d bbox
            float center[2] = {(x1 + x2)/2, (y1 + y2)/2};
            float depth = item.dep;
            float dimension[3] = {item.size3d.h, item.size3d.w, item.size3d.l}; 
            float dimension_l = item.size3d.l;
            float *point_3d = new float[3];
            unproject_2d_to_3d(dimension_l, center, depth, P, point_3d);
            // std::cout<< "x: "<< *point_3d<<" y: "<<*(point_3d+1)<<" z: "<<*(point_3d+2)<<std::endl;
            item.center.x = *point_3d;
            item.center.y = *(point_3d+1);
            item.center.z = *(point_3d+2);


            // calculate rotation angle around y axis
            // std::cout<< "alpha: "<< get_alpha(item.rot)<<std::endl;
            float rota_y = alpha2rot_y(get_alpha(item.rot), center[0], P[0][2], P[0][0]);
            // std::cout<< "rota_y: "<< rota_y <<std::endl;
            item.rota_y = rota_y;

            // calculate coordinates of 8 corner points
            float point_3d_8[8][3];
            float location[3] = {*point_3d, *(point_3d+1), *(point_3d+2)};
            compute_box_3d(dimension, location, rota_y, point_3d_8);
            // for (int i = 0; i < 8; i++) 
            // {
            //     std::cout<<"i: "<<i<<" x: "<<point_3d_8[i][0]<<" y: "<< point_3d_8[i][1]<<" z: "<< point_3d_8[i][2]<< std::endl;
            // }

            // calculate coordinates of 8 corner points on images
            float point_3d_8_temp[4][8];
            for (int m = 0; m < 8; m++)
            {
                for (int n = 0; n < 3; n++)
                {
                    point_3d_8_temp[n][m] = point_3d_8[m][n];
                }
                point_3d_8_temp[3][m] = 0.0;
            }
            // std::cout<<"********************************"<<std::endl;
            // for (int m = 0; m < 8; m++)
            // {
            //     for (int n = 0; n < 4; n++)
            //     {
            //         std::cout<<point_3d_8_temp[n][m]<<std::endl;
            //     }
            // }
            float pts_2d[8][2];
            project_to_image(point_3d_8_temp, P, pts_2d);
            // for (int i = 0; i < 8; i++) 
            // {
            //     std::cout<<"i: "<<i<<" x: "<<pts_2d[i][0]<<" y: "<< pts_2d[i][1]<< std::endl;
            // }
            item.bbox_3d.x1 = pts_2d[0][0];
            item.bbox_3d.y1 = pts_2d[0][1];
            item.bbox_3d.x2 = pts_2d[1][0];
            item.bbox_3d.y2 = pts_2d[1][1];
            item.bbox_3d.x3 = pts_2d[2][0];
            item.bbox_3d.y3 = pts_2d[2][1];
            item.bbox_3d.x4 = pts_2d[3][0];
            item.bbox_3d.y4 = pts_2d[3][1];
            item.bbox_3d.x5 = pts_2d[4][0];
            item.bbox_3d.y5 = pts_2d[4][1];
            item.bbox_3d.x6 = pts_2d[5][0];
            item.bbox_3d.y6 = pts_2d[5][1];
            item.bbox_3d.x7 = pts_2d[6][0];
            item.bbox_3d.y7 = pts_2d[6][1];
            item.bbox_3d.x8 = pts_2d[7][0];
            item.bbox_3d.y8 = pts_2d[7][1]; 
            // std::cout<<"item.bbox_3d.x1: "<< item.bbox_3d.x1 << std::endl;
            delete point_3d;
        }
    }
}

// void postProcess(std::vector<Detection> & result,const int &img_w ,const int& img_h, const bool& forwardFace)
void postProcess(std::vector<Detection> & result,const int &img_w ,const int& img_h, const int& forwardFace)
{


    int mark;
    int input_w = ctdet::input_w;
    int input_h = ctdet::input_h;
    float scale = std::min(float(input_w)/img_w,float(input_h)/img_h);
    float dx = (input_w - scale * img_w) / 2;
    float dy = (input_h - scale * img_h) / 2;
    //printf("%f %f %f %d %d \n",scale,dx,dy,img_w,img_h);
    for(auto&item:result)
    {

        float x1 = (item.bbox.x1 - dx) / scale ;
        float y1 = (item.bbox.y1 - dy) / scale ;
        float x2 = (item.bbox.x2 - dx) / scale ;
        float y2 = (item.bbox.y2 - dy) / scale ;
        x1 = (x1 > 0 ) ? x1 : 0 ;
        y1 = (y1 > 0 ) ? y1 : 0 ;
        x2 = (x2 < img_w  ) ? x2 : img_w - 1 ;
        y2 = (y2 < img_h ) ? y2  : img_h - 1 ;
        item.bbox.x1  = x1 ;
        item.bbox.y1  = y1 ;
        item.bbox.x2  = x2 ;
        item.bbox.y2  = y2 ;
        // if(forwardFace){
        if(forwardFace==1){
            float x,y;
            for(mark=0;mark<5; ++mark ){
                x = (item.marks[mark].x - dx) / scale ;
                y = (item.marks[mark].y - dy) / scale ;
                x = (x > 0 ) ? x : 0 ;
                y = (y > 0 ) ? y : 0 ;
                x = (x < img_w  ) ? x : img_w - 1 ;
                y = (y < img_h ) ? y  : img_h - 1 ;
                item.marks[mark].x = x ;
                item.marks[mark].y = y ;
            }
        }
    }
}

cv::Scalar randomColor(cv::RNG& rng) {
    int icolor = (unsigned) rng;
    return cv::Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}

// void drawImg(const std::vector<Detection> & result,cv::Mat& img,const std::vector<cv::Scalar>& color, const bool& forwardFace)
void drawImg(const std::vector<Detection> & result,cv::Mat& img,const std::vector<cv::Scalar>& color, const int& forwardFace)
{
    int mark;
    int box_think = (img.rows+img.cols) * .001 ;
    float label_scale = img.rows * 0.0009;
    int base_line ;
    for (const auto &item : result) {
        std::string label;
        std::stringstream stream;
        stream << ctdet::className[item.classId] << " " << item.prob << std::endl;
        std::getline(stream,label);

        auto size = cv::getTextSize(label,cv::FONT_HERSHEY_COMPLEX,label_scale,1,&base_line);


        // if(forwardFace==0){
        //     cv::putText(img,label,
        //             cv::Point(item.bbox.x2,item.bbox.y2 - size.height),
        //             cv::FONT_HERSHEY_COMPLEX, label_scale , color[item.classId], box_think/2, 8, 0);
        // }
        // if(forwardFace==1)
        // {
        //     for(mark=0;mark<5; ++mark )
        //     cv::circle(img, cv::Point(item.marks[mark].x, item.marks[mark].y), 1, cv::Scalar(255, 255, 0), 1);
        // }
        if(forwardFace==1)
        {
            cv::rectangle(img, cv::Point(item.bbox.x1,item.bbox.y1),
                cv::Point(item.bbox.x2 ,item.bbox.y2),
                color[item.classId], box_think*2, 8, 0);  
            for(mark=0;mark<5; ++mark )
            cv::circle(img, cv::Point(item.marks[mark].x, item.marks[mark].y), 1, cv::Scalar(255, 255, 0), 1);
        }
        else if (forwardFace==0)
        {
            cv::rectangle(img, cv::Point(item.bbox.x1,item.bbox.y1),
                cv::Point(item.bbox.x2 ,item.bbox.y2),
                color[item.classId], box_think*2, 8, 0);
            cv::putText(img,label,
                        cv::Point(item.bbox.x2,item.bbox.y2 - size.height),
                        cv::FONT_HERSHEY_COMPLEX, label_scale , color[item.classId], box_think/2, 8, 0);
        }
        else if (forwardFace==2)
        {

            cv::rectangle(img, cv::Point(item.bbox_3d.x1,item.bbox_3d.y1),
                cv::Point(item.bbox_3d.x6 ,item.bbox_3d.y6),
                color[item.classId], box_think*1, 8, 0);
            cv::rectangle(img, cv::Point(item.bbox_3d.x3,item.bbox_3d.y3),
                cv::Point(item.bbox_3d.x8 ,item.bbox_3d.y8),
                color[item.classId], box_think*1, 8, 0);        

            cv::line(img, cv::Point(item.bbox_3d.x1,item.bbox_3d.y1),
                cv::Point(item.bbox_3d.x2 ,item.bbox_3d.y2),
                color[item.classId], box_think*2, 8, 0);  
            cv::line(img, cv::Point(item.bbox_3d.x2,item.bbox_3d.y2),
                cv::Point(item.bbox_3d.x3 ,item.bbox_3d.y3),
                color[item.classId], box_think*2, 8, 0);       
            cv::line(img, cv::Point(item.bbox_3d.x3,item.bbox_3d.y3),
                cv::Point(item.bbox_3d.x4 ,item.bbox_3d.y4),
                color[item.classId], box_think*2, 8, 0);  
            cv::line(img, cv::Point(item.bbox_3d.x4,item.bbox_3d.y4),
                cv::Point(item.bbox_3d.x1 ,item.bbox_3d.y1),
                color[item.classId], box_think*2, 8, 0);                 
            cv::line(img, cv::Point(item.bbox_3d.x5,item.bbox_3d.y5),
                cv::Point(item.bbox_3d.x6 ,item.bbox_3d.y6),
                color[item.classId], box_think*2, 8, 0);  
            cv::line(img, cv::Point(item.bbox_3d.x6,item.bbox_3d.y6),
                cv::Point(item.bbox_3d.x7 ,item.bbox_3d.y7),
                color[item.classId], box_think*2, 8, 0);       
            cv::line(img, cv::Point(item.bbox_3d.x7,item.bbox_3d.y7),
                cv::Point(item.bbox_3d.x8 ,item.bbox_3d.y8),
                color[item.classId], box_think*2, 8, 0);  
            cv::line(img, cv::Point(item.bbox_3d.x8,item.bbox_3d.y8),
                cv::Point(item.bbox_3d.x5 ,item.bbox_3d.y5),
                color[item.classId], box_think*2, 8, 0);    

            cv::putText(img,label,
                        cv::Point(item.bbox.x2,item.bbox.y2 - size.height),
                        cv::FONT_HERSHEY_COMPLEX, label_scale , color[item.classId], box_think/2, 8, 0);
        }

    }
}