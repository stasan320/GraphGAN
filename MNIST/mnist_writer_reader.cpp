#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <Windows.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <ctime>

//#include <boost/lexical_cast.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return(int((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4);
}




void read_Mnist(std::string filename, std::vector<cv::Mat>& vec) {
	std::ifstream file(filename, std::ios::binary);
	if (file.is_open())
	{
		int magic_number;
		int number_of_images;
		int n_rows;
		int n_cols;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		file.read((char*)&n_rows, sizeof(n_rows));
		file.read((char*)&n_cols, sizeof(n_cols));

		number_of_images = ReverseInt(number_of_images);
		magic_number = ReverseInt(magic_number);
		n_rows = ReverseInt(n_rows);
		n_cols = ReverseInt(n_cols);

		std::cout << magic_number << " ";
		std::cout << number_of_images << " ";
		std::cout << n_rows << " ";
		std::cout << n_cols << " ";

		for (int i = 0; i < number_of_images; ++i) {
			cv::Mat tp(n_rows, n_cols, CV_8UC1);
			for (int r = 0; r < n_rows; r++) {
				for (int c = 0; c < n_cols; c++){
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					//std::cout << (int)temp << std::endl;
					tp.at<uchar>(r, c) = (uchar)temp;
					//vec[i].push_back((unsigned char)temp);
				}
			}

			cv::imshow("Window", tp);
			cv::waitKey(100);
			std::cout << i << std::endl;
		}
	}
}

void read_Mnist_Label(std::string filename, std::string save)
{
	std::ofstream saveLabel;
	saveLabel.open(save);
	std::fstream file(filename);
	if (file.is_open())
	{
		int magic_number;
		int number_of_images;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		magic_number = ReverseInt(magic_number);
		number_of_images = ReverseInt(number_of_images);
		std::cout << "magic number = " << magic_number << std::endl;
		std::cout << "number of images = " << number_of_images << std::endl;
		for (int i = 0; i < number_of_images; i++)
		{
			unsigned char label = 0;
			file.read((char*)&label, sizeof(label));
			//saveLabel << (int)label << " ";

			//std::cout << (int)label  << " " << 1<< std::endl;
		}
	}
	else {
		std::cout << "open file failed." << std::endl;
	}
	saveLabel.close();
	file.close();
}
int main() {


	


	std::ofstream files("D:\\Foton\\data\\d\\train-labels-idx1-ubyte");
	if (files.is_open())
	{
		int magic_number = 2049;
		int number_of_images = 10000;
		int num = number_of_images;

		magic_number = ReverseInt(magic_number);
		number_of_images = ReverseInt(number_of_images);

		files.write((char*)&magic_number, sizeof(magic_number));
		files.write((char*)&number_of_images, sizeof(number_of_images));

		std::cout << "magic number = " << magic_number << std::endl;
		std::cout << "number of images = " << number_of_images << std::endl;
		for (int i = 0; i < num; i++)
		{
			unsigned int label = 0;
			files.write((char*)&label, sizeof(label));
			//files << label;
			//saveLabel << (int)label << " ";

			//std::cout << (int)label << std::endl;
		}
	}


	read_Mnist_Label("D:\\Foton\\data\\d\\train-labels-idx1-ubyte", "D:\\Foton\\xz\\txt.d");


	std::ofstream file("D:\\Foton\\data\\d\\train-images-idx3-ubyte", std::ios::binary);
	if (file.is_open()) {
		//cv::Mat dtp = cv::imread("D:\\Foton\\ngnl_data\\img_align_celeba\\000001.jpg");
		int magic_number = 2051;
		int number_of_images = 10000;
		//int n_rows = 218;
		//int n_cols = 178;
		int n_rows = 28;
		int n_cols = 28;

		int num = number_of_images;
		int rows = n_rows;
		int cols = n_cols;

		number_of_images = ReverseInt(number_of_images);
		magic_number = ReverseInt(magic_number);
		n_rows = ReverseInt(n_rows);
		n_cols = ReverseInt(n_cols);
		
		file.write((char*)&magic_number, sizeof(magic_number));
		file.write((char*)&number_of_images, sizeof(number_of_images));
		file.write((char*)&n_rows, sizeof(n_rows));
		file.write((char*)&n_cols, sizeof(n_cols));

		std::cout << magic_number << " ";
		std::cout << number_of_images << " ";
		std::cout << n_rows << " ";
		std::cout << n_cols << " ";

		for (int i = 0; i < num; ++i) {
			//cv::Mat tp(rows, cols, CV_8UC1);
			cv::Mat tp(rows, cols, CV_8UC1);
			tp = cv::imread("D:\\Foton\\ngnl_data\\training\\0\\1 (" + std::to_string(i%5000 + 1) + ").png");
			/*if (i +1  < 10) {
				tp = cv::imread("D:\\Foton\\ngnl_data\\img_align_celeba\\00000" + std::to_string(i + 1) + ".jpg");
			}
			if ((i + 1 >= 10) && (i + 1 < 100)) {
				tp = cv::imread("D:\\Foton\\ngnl_data\\img_align_celeba\\0000" + std::to_string(i + 1) + ".jpg");
			}
			if ((i + 1 >= 100) && (i + 1 < 1000)) {
				tp = cv::imread("D:\\Foton\\ngnl_data\\img_align_celeba\\000" + std::to_string(i + 1) + ".jpg");
			}
			if ((i + 1 >= 1000) && (i + 1 < 10000)) {
				tp = cv::imread("D:\\Foton\\ngnl_data\\img_align_celeba\\00" + std::to_string(i + 1) + ".jpg");
			}
			if ((i + 1 >= 10000) && (i + 1 < 100000)) {
				tp = cv::imread("D:\\Foton\\ngnl_data\\img_align_celeba\\0" + std::to_string(i + 1) + ".jpg");
			}*/



			for (int r = 0; r < tp.rows; r++) {
				for (int c = 0; c < tp.cols; c++) {
					unsigned int temp = 0;
					temp = (int)tp.at<unsigned int>(r, c);
					//std::cout << (int)temp << "   " << r << "   " << c << std::endl;
					//tdp.at<uchar>(r, c) = (uchar)temp;
					//std::cout << temp << std::endl;
					file.write((char*)&temp, 1);
					//file << (unsigned char)temp;
					//std::cout << (int)temp << std::endl;

					//vec[i].push_back((unsigned char)temp);
				}
			}
		}
	}

	return 0;


	//read MNIST iamge into OpenCV Mat vector
	std::vector<cv::Mat> vec;
	std::vector<cv::Mat> vec2;
	//write_Mnist("D:\\Foton\\data.idx3-ubyte", vec);


	//read_Mnist("D:\\Foton\\train-images.idx3-ubyte", vec2);

	read_Mnist("D:\\Foton\\data\\d\\train-images-idx3-ubyte", vec2);
	std::cout << vec2.size() << std::endl;

	return 0;
	for (int i = 0; i < vec2.size(); i++) {
		cv::imshow("Window", vec2[i]);
		cv::waitKey(100);
	}

	return 0;
}
