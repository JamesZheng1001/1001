# include "mainwindow.h"
# include <QApplication>

# include <opencv2/core/core.hpp>
# include <opencv2/highgui/highgui.hpp>
# include <opencv2/imgproc/imgproc.hpp>
from sys import argv

import argc as argc
from cv2.cv2 import imread, namedWindow, WINDOW_AUTOSIZE, imshow, waitKey
from cv2.mat_wrapper import Mat
from pip._vendor.html5lib.treeadapters.sax import namespace

using
namespace
cv;

int
main(int
argc, char * argv[])
{
    QApplication
a(argc, argv);

cv::Mat
image = imread("D:\python\yolo\test_images\IMG_20210506_103652 (2).jpg");
namedWindow("Display window", WINDOW_AUTOSIZE);
imshow("Display window", image);
waitKey(0);

MainWindow
w;
w.show();
return a.exec();
}
