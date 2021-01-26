# TRT_Ultra_Fast_Lane_Detect

TRT_Ultra_Fast_Lane_Detect is an implementation of converting Ultra fast lane detection into tensorRT model by Python API.  There are some other works in our project are listed below:

- The detection procedure is encapsulated.
- The pytorch model is transformed into onnx model and trt model.
- The trt models have different versions: FP32, FP16, INT8.
- The Tusimple data set can be compressed by /calibration_data/make_mini_tusimple.py. There are redundancies in the Tusimple data set, for only 20-th frames are used. The compressed tusimple data set takes about 1GB.

The original project, model, and paper is available from https://github.com/cfzd/Ultra-Fast-Lane-Detection



### Ultra-Fast-Lane-Detection

PyTorch implementation of the paper "[Ultra Fast Structure-aware Deep Lane Detection](https://arxiv.org/abs/2004.11757)".

Updates: Our paper has been accepted by ECCV2020.

[![alt text](https://github.com/cfzd/Ultra-Fast-Lane-Detection/raw/master/vis.jpg)](https://github.com/cfzd/Ultra-Fast-Lane-Detection/blob/master/vis.jpg)

The evaluation code is modified from [SCNN](https://github.com/XingangPan/SCNN) and [Tusimple Benchmark](https://github.com/TuSimple/tusimple-benchmark).

Caffe model and prototxt can be found [here](https://github.com/Jade999/caffe_lane_detection).



### Trained models

The trained models can be obtained by the following table:

| Dataset  | Metric paper | Metric This repo | Avg FPS on GTX 1080Ti | Model                                                        |
| -------- | ------------ | ---------------- | --------------------- | ------------------------------------------------------------ |
| Tusimple | 95.87        | 95.82            | 306                   | [GoogleDrive](https://drive.google.com/file/d/1WCYyur5ZaWczH15ecmeDowrW30xcLrCn/view?usp=sharing)/[BaiduDrive(code:bghd)](https://pan.baidu.com/s/1Fjm5yVq1JDpGjh4bdgdDLA) |
| CULane   | 68.4         | 69.7             | 324                   | [GoogleDrive](https://drive.google.com/file/d/1zXBRTw50WOzvUp6XKsi8Zrk3MUC3uFuq/view?usp=sharing)/[BaiduDrive(code:w9tw)](https://pan.baidu.com/s/19Ig0TrV8MfmFTyCvbSa4ag) |



### Installation on NVIDIA Jetson
Requirement
*   NVIDIA Jetson
*   JetPack 4.5 / 4.4.1
*   USB WebCam

##### JetPack
Install JetPack 4.5 / 4.4.1 on your Jetson.  
[https://developer.nvidia.com/EMBEDDED/Jetpack](https://developer.nvidia.com/EMBEDDED/Jetpack)

##### Python3 virtualenv
The advantage of using virtualenv is that it does not pollute the host environment. Also, you can unify the troublesome python3 and pip3 commands into python and pip.  
The disadvantage is that the package has not been fully tested. Installation often suffers.  

Install packages on host.
```
sudo apt-get update
sudo apt-get install -y python3-dev python3-pip
sudo -H pip3 install -U pip testresources setuptools==49.6.0
sudo -H pip3 install -U futures==3.1.1 protobuf==3.12.2 pybind11==2.5.0
sudo -H pip3 install -U cython==0.29.21
sudo -H pip3 install -U numpy==1.18.5
sudo -H pip3 install -U matplotlib
sudo -H pip3 install -U scipy
```
Install Pytorch on host.(JetPack 4.5 / 4.4.1)  
Pytorch binary is different for each JetPack version.
```
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install -y openmpi-doc openmpi-bin libopenmpi-dev libopenblas-dev
wget https://nvidia.box.com/shared/static/cs3xn3td6sfgtene6jdvsxlr366m2dhq.whl -O torch-1.7.0-cp36-cp36m-linux_aarch64.whl
sudo -H pip3 install ./torch-1.7.0-cp36-cp36m-linux_aarch64.whl
mkdir ~/github; cd ~/github
git clone https://github.com/pytorch/vision
cd vision
git checkout v0.8.0
sudo python3 setup.py install
```


create virtualenv
```
pip3 install virtualenv
python3 -m virtualenv -p python3 ~/.virtualenv/ml --system-site-packages
echo "source ~/.virtualenv/ml/bin/activate" >> ~/.bashrc
# "source ~/.virtualenv/ml/bin/activate" in the shell script
. ~/.virtualenv/ml/bin/activate
```
now, python and pip are python3 and pip3.
```
pip install PySpin
pip install addict
pip install tqdm
pip install tensorboard
pip install pycuda
pip install pathspec
pip install gdown
```

##### ONNX 1.8.0
Install
```
sudo apt-get install protobuf-compiler libprotoc-dev
export ONNX_ML=1
mkdir ~/github
cd ~/github
git clone --recursive https://github.com/onnx/onnx.git
cd onnx
python setup.py install
pip install pytest nbval
echo "export ONNX_ML=1" >> ~/.bashrc
```
check
```
cd
env ONNX_ML=1 python -c "import onnx"
```
```
cd ~/github/onnx
pytest
```

##### TRT_Ultra_Fast_Lane_Detect
```
cd ~/github
git clone https://github.com/naisy/TRT_Ultra_Fast_Lane_Detect
cd TRT_Ultra_Fast_Lane_Detect
```

### Run

##### Download trained models
Download pre-trained pytorch models.
*   Tusimple Dataset
```
gdown https://drive.google.com/uc?id=1WCYyur5ZaWczH15ecmeDowrW30xcLrCn
```
*   CULane Dataset
```
gdown https://drive.google.com/uc?id=1zXBRTw50WOzvUp6XKsi8Zrk3MUC3uFuq
```

##### Convert to TensorRT model
Above all, you have to train or download a 4 lane model trained by the Ultra Fast Lane Detection pytorch version. You have to change some codes, if you want to use different lane number. 

*   Make model.onnx file
```
python torch2onnx.py --test_model tusimple_18.pth configs/tusimple_4.py
# or
python torch2onnx.py --test_model culane_18.pth configs/culane.py
```

*   Make model_fp16.engine file  
You can choose fp16 or fp32.
```
python onnx_to_tensorrt.py -p fp16 --model model.onnx
```

##### Edit tensorrt_run.py
Check camera informations.
```
sudo apt-get install v4l-utils
v4l2-ctl -d /dev/video0 --list-formats-ext
```
Edit tensorrt_run.py to set the resolution and fps supported by the camera.
```
vi tensorrt_run.py
```
(MJPEG settings)
```
    camera_device = '/dev/video0'
    camera_width, camera_height, camera_fps = 1280, 720, 25
```

##### Run inference
Need usb webcam.
```
python tensorrt_run.py --model model_fp16.engine configs/tusimple_4.py
# or
python tensorrt_run.py --model model_fp16.engine configs/culane.py
```

### Evalutaion

|            | Pytorch | libtorch | tensorRT(FP32) | tensorRT(FP16) | tensorRT(int8) |
| :--------: | :-----: | :------: | :------------: | :------------: | :------------: |
|  GTX1060   |  55fps  |  55fps   |     55fps      |  Unsupported   |     99fps      |
| Xavier AGX |  27fps  |  27fps   |       --       |       --       |       --       |
| Jetson TX1 |  8fps   |   8fps   |      8fps      |     16fps      |  Unsupported   |
| Jetson nano A01(4GB) |  -- | -- |      --        |     8fps       |  Unsupported   |

Where "--" denotes the experiment hasn't been completed yet.
Anyone with untested equipment can send his results to the issues. The results will be adopted.

