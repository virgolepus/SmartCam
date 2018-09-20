# SmartCam
SmartCam project for ArcelorMittal Hackathon

Dependencies
Follow the installation instruction here:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

Object Detector Tool
It's a tool to load pictures and output the box coordinates of user selected features.
OpenCV C++ is reuqired. To build the .cpp file with command line:
g++ $(pkg-config --cflags --libs opencv) -std=c++11 objectdetector.cpp -o objectdetector

TFWriter Tool
write_objectbox.py is a tool to read the output from objectdetector tool, load raw bytes of images, and write to the Tensorflow format.
