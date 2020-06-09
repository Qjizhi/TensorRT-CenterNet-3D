//
// Created by cao on 19-10-26.
//


#include <argparse.h>
#include <string>
#include <iostream>
#include <memory>
#include "ctdetNet.h"
#include "ctdetConfig.h"
#include "utils.h"



int main(int argc, const char** argv){
    optparse::OptionParser parser;
    parser.add_option("-e", "--input-engine-file").dest("engineFile").set_default("test.engine")
            .help("the path of onnx file");
    parser.add_option("-i", "--input-img-file").dest("imgFile").set_default("test.jpg");
    parser.add_option("-c", "--input-video-file").dest("capFile").set_default("test.h264");
    optparse::Values options = parser.parse_args(argc, argv);
    if(options["engineFile"].size() == 0){
        std::cout << "no file input" << std::endl;
        exit(-1);
    }

    cv::RNG rng(244);
    std::vector<cv::Scalar> color = { cv::Scalar(255, 0,0),cv::Scalar(0, 255,0)};
    //for(int i=0; i<ctdet::classNum;++i)color.push_back(randomColor(rng));


    cv::namedWindow("result",cv::WINDOW_NORMAL);
    cv::resizeWindow("result",1024,768);

    ctdet::ctdetNet net(options["engineFile"]);
    std::unique_ptr<float[]> outputData(new float[net.outputBufferSize]);

    cv::Mat img ;
    if(options["imgFile"].size()>0)
    {
        img = cv::imread(options["imgFile"]);
        auto inputData = prepareImage(img,net.forwardFace);

        net.doInference(inputData.data(), outputData.get());
        net.printTime();

        int num_det = static_cast<int>(outputData[0]);
        std::vector<Detection> result;
        result.resize(num_det);
        memcpy(result.data(), &outputData[1], num_det * sizeof(Detection));
        
        postProcess(result,img,net.forwardFace);

        // modified by feng for debug
        std::cout << "num_det: " << outputData[0] <<std::endl;
        // for(int i = 0; i < num_det; i++)
        // {
            // std::cout << "number: " <<i+1<<std::endl;
            // std::cout << "classID: " << result[i].classId <<std::endl;
            // std::cout << "bbox: " << result[i].bbox.x1<<" "<<result[i].bbox.y1<<" "<<result[i].bbox.x2<<" "<<result[i].bbox.y2<<std::endl;
            // std::cout << "bbox_3d: " << result[i].bbox_3d.x1<<" "<<result[i].bbox_3d.y1<<" "<<result[i].bbox_3d.x2<<" "<<result[i].bbox_3d.y2<<" "<<result[i].bbox_3d.x3<<" "<<result[i].bbox_3d.y3<<" "<<result[i].bbox_3d.x4<<" "<<result[i].bbox_3d.y4<<" "<<result[i].bbox_3d.x5<<" "<<result[i].bbox_3d.y5<<" "<<result[i].bbox_3d.x6<<" "<<result[i].bbox_3d.y6<<" "<<result[i].bbox_3d.x7<<" "<<result[i].bbox_3d.y7<<" "<<result[i].bbox_3d.x8<<" "<<result[i].bbox_3d.y8<<std::endl;
            // std::cout << "marks: " << result[i].marks[0].x<<" "<<result[i].marks[0].y<<" "<<result[i].marks[1].x<<" "<<result[i].marks[1].y<<" "<<result[i].marks[2].x<<" "<<result[i].marks[2].y<<" "<<result[i].marks[3].x<<" "<<result[i].marks[3].y<<" "<<result[i].marks[4].x<<" "<<result[i].marks[4].y<<std::endl;
            // std::cout << "probability: " << result[i].prob <<std::endl;
            // std::cout << "size: " << result[i].size3d.h <<" "<<result[i].size3d.w<<" "<<result[i].size3d.l<<std::endl;
            // std::cout << "dep: " << result[i].dep<<std::endl;
            // std::cout << "rot: " << result[i].rot.ang1 <<" "<<result[i].rot.ang2<<" "<<result[i].rot.ang3<<" "<<result[i].rot.ang4<<" "<<result[i].rot.ang5<<" "<<result[i].rot.ang6<<" "<<result[i].rot.ang7<<" "<<result[i].rot.ang8<<std::endl;
            // std::cout << "rota_y: " << result[i].rota_y<<std::endl;
            // std::cout << std::endl;
        // }
        drawImg(result,img,color,net.forwardFace);

        cv::imshow("result",img);
        cv::waitKey(0);
    }

    if(options["capFile"].size()>0){
        cv::VideoCapture cap(options["capFile"]);
        while (cap.read(img))
        {
            auto inputData = prepareImage(img,net.forwardFace);

            net.doInference(inputData.data(), outputData.get());
            net.printTime();

            int num_det = static_cast<int>(outputData[0]);

            std::vector<Detection> result;

            result.resize(num_det);

            memcpy(result.data(), &outputData[1], num_det * sizeof(Detection));

            postProcess(result,img,net.forwardFace);

            drawImg(result,img,color,net.forwardFace);

            cv::imshow("result",img);
            if((cv::waitKey(1)& 0xff) == 27){
                cv::destroyAllWindows();
                return 0;
            };

        }

    }

    // check GPU device infos, outputs are: 
    // 使用GPU device 0: Quadro M4000
    // SM的数量：13
    // 每个线程块的共享内存大小：48 KB
    // 每个线程块的最大线程数：1024
    // 每个EM的最大线程数：2048
    // 每个EM的最大线程束数：64
    // int dev = 0;
    // cudaDeviceProp devProp;
    // CUDA_CHECK(cudaGetDeviceProperties(&devProp, dev));
    // std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
    // std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
    // std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    // std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
    // std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
    // std::cout << "每个EM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;


    // cudaGridSize(92160);
    // x: 180 y: 11

    return 0;

}
