
#include "stdafx.h"

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>



// 测试图片加载
#if 0

#include <iostream>

using namespace cv;
using namespace std;

int main() {

	Mat src = imread("../images/1.jpg");

	cout << src.size << endl;
	cout << src.size[0] << endl;
	cout << src.size[1] << endl;
	cout << src.size[2] << endl;
	cout << src.size[3] << endl;

	return 0;
}

#endif

// cap121，读取网络参数并显示
# if 0
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

int main() {

	string bin_model = "../models/googlenet/bvlc_googlenet.caffemodel";
	string protxt = "../models/googlenet/bvlc_googlenet.prototxt";

	// load CNN model
	Net net = dnn::readNetFromCaffe(protxt, bin_model);

	// 获取各层信息
	vector<String> layer_names = net.getLayerNames();
	for (int i = 0; i < layer_names.size(); i++) {
		int id = net.getLayerId(layer_names[i]);
		auto layer = net.getLayer(id);
		printf("layer id:%d, type: %s, name:%s \n", id, layer->type.c_str(), layer->name.c_str());
	}

	return 0;
}
#endif

// cap122，使用tensorflow_inception_graph.pb的网络参数识别物体
#if 0
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

String labels_txt_file	= "D:/CVProjects/OpenCVDNN/models/googlenet/classification_classes_ILSVRC2012.txt";
String caffe_bin_file	= "D:/CVProjects/OpenCVDNN/models/googlenet/bvlc_googlenet.caffemodel";
String protxt_file		= "D:/CVProjects/OpenCVDNN/models/googlenet/bvlc_googlenet.prototxt";
vector<String> readClassNames();

int main(int argc, char** argv) {
	Mat src = imread("../images/space_shuttle.jpg");
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", src);
	vector<String> labels = readClassNames();

	Mat rgb;
	cvtColor(src, rgb, COLOR_BGR2RGB);
	int w = 224;
	int h = 224;

	// 加载网络
	Net net = readNetFromCaffe(protxt_file, caffe_bin_file);
	if (net.empty()) {
		printf("read caffe model data failure...\n");
		return -1;
	}
	Mat inputBlob = blobFromImage(src, 1.0f, Size(224, 224), Scalar(), true, false);
	inputBlob -= 117.0; // 均值

	// 执行图像分类
	Mat prob;
	net.setInput(inputBlob);	// 不设置"input"参数则默认放在最初的输入层
	prob = net.forward();		// 不设置"softmax2"参数则默认为最终输出层

	// 得到最可能分类输出
	Mat probMat = prob.reshape(1, 1);
	
	// 调试
#if 0
	cout << prob.cols << endl;
	cout << sum(probMat)[0] << endl;
	for (int row = 0; row < probMat.rows; row++) {
		float* curr_row = probMat.ptr<float>(row);
		for (int col = 0; col < 50; col++) {
			cout << *curr_row++ <<endl;
		}
	}
#endif

	Point classNumber;		// 最大值坐标 (列index，行index)
	double classProb;
	minMaxLoc(probMat, NULL, &classProb, NULL, &classNumber);
	int classidx = classNumber.x;
	printf("\n current image classification : %s, possible : %.2f", labels.at(classidx).c_str(), classProb);

	// 显示文本
	putText(src, labels.at(classidx), Point(20, 20), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, 8);
	imshow("Image Classification", src);
	imwrite("D:/result.png", src);
	waitKey(0);
	return 0;
}

// 读取所有类别物体的名称
std::vector<String> readClassNames()
{
	std::vector<String> classNames;

	std::ifstream fp(labels_txt_file);
	if (!fp.is_open())
	{
		printf("could not open file...\n");
		exit(-1);
	}
	std::string name;
	while (!fp.eof())
	{
		std::getline(fp, name);
		if (name.length())
			classNames.push_back(name);
	}
	fp.close();
	return classNames;
}
#endif


// cap124，使用MobileNetSSD_deploy.caffemodel模型进行检测
#if 1

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

const size_t width = 300;
const size_t height = 300;

String labelFile = "D:/CVProjects/OpenCVDNN/models/ssd/labelmap_det.txt";
String modelFile = "D:/CVProjects/OpenCVDNN/models/ssd/MobileNetSSD_deploy.caffemodel";
String model_text_file = "D:/CVProjects/OpenCVDNN/models/ssd/MobileNetSSD_deploy.prototxt";

String objNames[] = { "background",
"aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair",
"cow", "diningtable", "dog", "horse",
"motorbike", "person", "pottedplant",
"sheep", "sofa", "train", "tvmonitor" };

int main(int argc, char** argv) {
	Mat frame = imread("../images/objects.jpg");
	if (frame.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow("input image", WINDOW_AUTOSIZE);
	imshow("input image", frame);

	Net net = readNetFromCaffe(model_text_file, modelFile);

	Mat blobImage = blobFromImage(frame, 0.007843,
		Size(300, 300),
		Scalar(127.5, 127.5, 127.5), true, false);
	printf("blobImage width : %d, height: %d\n", blobImage.cols, blobImage.rows);

	// Network produces output blob with a shape 1x1xNx7 where N is a number of
	// detections and an every detection is a vector of values
	// [batchId, classId, confidence, left, top, right, bottom]
	net.setInput(blobImage, "data");
	Mat detection = net.forward("detection_out");
	
	
	vector<double> layersTimings;
	double freq = getTickFrequency() / 1000;
	double time = net.getPerfProfile(layersTimings) / freq;
	printf("execute time : %.2f ms\n", time);

	// 设定置信度阈值
	float confidence_threshold = 0.5;
	Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
	for (int i = 0; i < detectionMat.rows; i++) {
		float confidence = detectionMat.at<float>(i, 2);
		if (confidence > confidence_threshold) {
			size_t objIndex = (size_t)(detectionMat.at<float>(i, 1));
			float tl_x = detectionMat.at<float>(i, 3) * frame.cols;
			float tl_y = detectionMat.at<float>(i, 4) * frame.rows;
			float br_x = detectionMat.at<float>(i, 5) * frame.cols;
			float br_y = detectionMat.at<float>(i, 6) * frame.rows;

			Rect object_box((int)tl_x, (int)tl_y, (int)(br_x - tl_x), (int)(br_y - tl_y));
			rectangle(frame, object_box, Scalar(0, 0, 255), 2, 8, 0);
			putText(frame, format(" confidence %.2f, %s", confidence, objNames[objIndex].c_str()), Point(tl_x - 10, tl_y - 5), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 0), 2, 8);
		}
	}
	imshow("ssd-demo", frame);

	waitKey(0);
	return 0;
}

#endif


// cap125，使用MobileNetSSD_deploy.caffemodel模型进行视频检测
#if 0
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

const size_t width = 300;
const size_t height = 300;

// TODO 下面路径需要修改
String labelFile = "D:/projects/opencv_tutorial/data/models/ssd/labelmap_det.txt";
String modelFile = "D:/projects/opencv_tutorial/data/models/ssd/MobileNetSSD_deploy.caffemodel";
String model_text_file = "D:/projects/opencv_tutorial/data/models/ssd/MobileNetSSD_deploy.prototxt";

String objNames[] = { "background",
"aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair",
"cow", "diningtable", "dog", "horse",
"motorbike", "person", "pottedplant",
"sheep", "sofa", "train", "tvmonitor" };

int main(int argc, char** argv) {
	// load model
	Net net = readNetFromCaffe(model_text_file, modelFile);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

	VideoCapture cap = VideoCapture(0);
	Mat frame;
	while (true) {
		bool ret = cap.read(frame);
		if (!ret) break;
		Mat blobImage = blobFromImage(frame, 0.007843,
			Size(300, 300),
			Scalar(127.5, 127.5, 127.5), true, false);
		printf("blobImage width : %d, height: %d\n", blobImage.size[2], blobImage.size[3]);

		net.setInput(blobImage, "data");
		Mat detection = net.forward("detection_out");
		vector<double> layersTimings;
		double freq = getTickFrequency();
		double time = 1000.0 * net.getPerfProfile(layersTimings) / freq;
		printf("execute time : %.2f ms\n", time);


		Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
		float confidence_threshold = 0.5;
		for (int i = 0; i < detectionMat.rows; i++) {
			float confidence = detectionMat.at<float>(i, 2);
			if (confidence > confidence_threshold) {
				size_t objIndex = (size_t)(detectionMat.at<float>(i, 1));
				float tl_x = detectionMat.at<float>(i, 3) * frame.cols;
				float tl_y = detectionMat.at<float>(i, 4) * frame.rows;
				float br_x = detectionMat.at<float>(i, 5) * frame.cols;
				float br_y = detectionMat.at<float>(i, 6) * frame.rows;

				Rect object_box((int)tl_x, (int)tl_y, (int)(br_x - tl_x), (int)(br_y - tl_y));
				rectangle(frame, object_box, Scalar(0, 0, 255), 2, 8, 0);
				putText(frame, format(" confidence %.2f, %s", confidence, objNames[objIndex].c_str()), Point(tl_x - 10, tl_y - 5), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 0), 2, 8);
			}
		}
		imshow("ssd-video-demo", frame);

		// 若用户按下ESC键则退出
		char c = waitKey(10);
		if (c == 27) {
			break;
		}
	}

	waitKey(0);
	return 0;
}
#endif


// cap130，使用yolo进行目标检测
#if 0
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>
using namespace std;
using namespace cv;
using namespace cv::dnn;

String labelFile = "D:/CVProjects/OpenCVDNN/models/ssd/labelmap_det.txt";

String yolo_cfg = "D:/projects/pose_body/hand/yolov3.cfg";
String yolo_model = "D:/projects/pose_body/hand/yolov3.weights";
int main(int argc, char** argv)
{
	Net net = readNetFromDarknet(yolo_cfg, yolo_model);
	net.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
	net.setPreferableTarget(DNN_TARGET_CPU);
	std::vector<String> outNames = net.getUnconnectedOutLayersNames();
	for (int i = 0; i < outNames.size(); i++) {
		printf("output layer name : %s\n", outNames[i].c_str());
	}

	// 加载COCO数据集标签
	vector<string> classNamesVec;
	ifstream classNamesFile("D:/projects/opencv_tutorial/data/models/object_detection_classes_yolov3.txt");
	if (classNamesFile.is_open())
	{
		string className = "";
		while (std::getline(classNamesFile, className))
			classNamesVec.push_back(className);
	}

	// 加载图像 
	Mat frame = imread("D:/images/pedestrian.png");
	Mat inputBlob = blobFromImage(frame, 1 / 255.F, Size(416, 416), Scalar(), true, false);
	net.setInput(inputBlob);

	// 检测
	std::vector<Mat> outs;
	net.forward(outs, outNames);
	vector<double> layersTimings;
	double freq = getTickFrequency();
	double time = 1000.0 * net.getPerfProfile(layersTimings) / freq;
	ostringstream ss;
	ss << "detection time: " << time << " ms";

	putText(frame, ss.str(), Point(20, 20), 0, 0.5, Scalar(0, 0, 255));
	vector<Rect> boxes;
	vector<int> classIds;
	vector<float> confidences;
	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Network produces output blob with a shape NxC where N is a number of
		// detected objects and C is a number of classes + 4 where the first 4
		// numbers are [center_x, center_y, width, height]
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > 0.5)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	// 非最大抑制操作
	vector<int> indices;
	NMSBoxes(boxes, confidences, 0.5, 0.2, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		String className = classNamesVec[classIds[idx]];
		putText(frame, className.c_str(), box.tl(), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 0, 0), 2, 8);
		rectangle(frame, box, Scalar(0, 0, 255), 2, 8, 0);
	}

	imshow("YOLOv3-Detections", frame);
	waitKey(0);
	return;
}
#endif


// cap131，使用yolo-tiny3进行检测
#if 0
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>
using namespace std;
using namespace cv;
using namespace cv::dnn;
void image_detection();

String yolo_tiny_model = "D:/CVProjects/OpenCVDNN/models/yolov3-tiny-coco/yolov3-tiny.weights";
String yolo_tiny_cfg = "D:/CVProjects/OpenCVDNN/models/yolov3-tiny-coco/yolov3-tiny.cfg";
int main(int argc, char** argv)
{
	image_detection();
}

void image_detection() {
	Net net = readNetFromDarknet(yolo_tiny_cfg, yolo_tiny_model);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);
	std::vector<String> outNames = net.getUnconnectedOutLayersNames();
	for (int i = 0; i < outNames.size(); i++) {
		printf("output layer name : %s\n", outNames[i].c_str());
	}

	vector<string> classNamesVec;
	ifstream classNamesFile("D:/CVProjects/OpenCVDNN/models/yolov3-tiny-coco/object_detection_classes_yolov3.txt");
	if (classNamesFile.is_open())
	{
		string className = "";
		while (std::getline(classNamesFile, className))
			classNamesVec.push_back(className);
	}

	// 加载图像 
	//Mat frame = imread("../images/pedestrian.png");
	Mat frame = imread("../images/objects.jpg");

	Mat inputBlob = blobFromImage(frame, 1 / 255.F, Size(416, 416), Scalar(), true, false);
	net.setInput(inputBlob);

	// 检测
	std::vector<Mat> outs;
	net.forward(outs, outNames);
	vector<double> layersTimings;
	double freq = getTickFrequency() / 1000;
	double time = net.getPerfProfile(layersTimings) / freq;
	ostringstream ss;
	ss << "detection time: " << time << " ms";
	putText(frame, ss.str(), Point(20, 20), 0, 0.5, Scalar(0, 0, 255));
	vector<Rect> boxes;
	vector<int> classIds;
	vector<float> confidences;
	for (size_t i = 0; i < outs.size(); ++i)
	{
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > 0.5)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	vector<int> indices;
	NMSBoxes(boxes, confidences, 0.5, 0.2, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		String className = classNamesVec[classIds[idx]];
		putText(frame, className.c_str(), box.tl(), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 0, 0), 2, 8);
		rectangle(frame, box, Scalar(0, 0, 255), 2, 8, 0);
	}

	imshow("YOLOv3-Detections", frame);
	waitKey(0);
	return;
}
#endif