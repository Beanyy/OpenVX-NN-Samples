# OpenVX-NN-Samples

## Overview
Neural network examples using OpenVX. Yolo and InceptionV3 running on OpenVX. Both of these samples were generated by conveting existing caffe nets to OpenVX using [AMD's Caffe to OpenVX tool](https://github.com/GPUOpen-ProfessionalCompute-Libraries/amdovx-modules/tree/develop/utils/inference_generator)

The converter only generates a program which takes a raw input and raw output file so I've included scripts to convert an image to the proper input and a parser for the outputs.

### Compatability
Since both of these were generated using AMD's tool, it will look for the OpenVX includes and libraries in /opt/rocm by default so you'll have to change this in the CMakeLists.txt if you're running on a non-rocm setup.

To do this just modify the lines:
```
include_directories (/opt/rocm/include)
link_directories    (/opt/rocm/lib)
```

## Yolo
Input into this net is a 1x3x416x416 tensor and the output is 1x425x12x12.
The net was taked from [here (tiny_yolo_deploy.prototxt)](https://github.com/tsingjinyun/caffe-yolov2)

Build the sample:
```
(Modify CMakeLists.txt if needed)
mkdir build
cd build
cmake ..
make
```

Preprocess the input:
```
python jpg_to_raw.py images/dog.jpg images/dog.raw
```

Run the net:
```
./build/anntest $PWD images/dog.raw prediction.raw
```

Parse output:
You need also pass in the original image passed into the parser so it can overlay the predicted boxes on top of it.
```
python yolo_parse.py prediction.raw images/dog.jpg
```

## Inception V3
Input into this net is a 1x3x299x299 tensor and the output is 1x1000x1x1.
Build the sample:
```
(Modify CMakeLists.txt if needed)
mkdir build
cd build
cmake ..
make
```

Preprocess the input:
```
python jpg_to_raw.py images/grace_hopper.jpg images/grace_hopper.raw
```

Run the net:
```
./build/anntest $PWD images/grace_hopper.raw prediction.raw
```

Parse output:
```
python raw_to_class.py prediction.raw
```
