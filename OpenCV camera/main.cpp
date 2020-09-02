#include <iostream>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    for (int i = -1; i < 500; i++) {
        cv::VideoCapture cam{ i };

        if (cam.isOpened()) {
            cv::String window = "Mirrow";

            cv::namedWindow(window);

            auto run = true;

            while (run) {
                cv::Mat img;

                cam >> img; // получить кадр

                if (!img.empty()) {
                    imshow(window, img);
                }

                run = (cv::waitKey(33) != 27); // выход по ESC
            }

            cv::destroyAllWindows();
        }
        else {
            std::cerr << "Cant open default cam # " << i << std::endl;
        }
    }

    return 0;
}
