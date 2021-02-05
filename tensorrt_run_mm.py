from __future__ import print_function

import os
import argparse
import cv2
import tensorrt as trt
import common
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import pycuda.gpuarray as gpuarray
import time
import scipy.special
import torchvision.transforms as transforms
from PIL import Image
from utils.common import merge_config
from configs.constant import culane_row_anchor, tusimple_row_anchor

from paho.mqtt import client as mqtt_client
import random
from collections import defaultdict
import json

frame_result = []

img_transforms = transforms.Compose([
    transforms.Resize((288, 800)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

col_sample = np.linspace(0, 800 - 1, 100)
col_sample_w = col_sample[1] - col_sample[0]

color = [(255,255,0), (255,0,0), (0,0,255), (0,255,0)]

EXPLICIT_BATCH = []
if trt.__version__[0] >= '7':
    EXPLICIT_BATCH.append(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

def current_millis():
    # return current timestamp.
    # This func uses between read frame and prediction.
    return round(time.time() * 1000)

########################################
# MQTT
########################################
broker = 'localhost'
port = 1883
topic = '/mm/lane'
client_id = f'LD-1_{random.randint(0, 1000)}'
def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    # client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def publish(json_str):
    result = client.publish(topic, json_str)
    # result: [0, 1]
    status = result[0]
    if status == 0:
        print(f"Send json to topic `{topic}`")
    else:
        print(f"Failed to send message to topic {topic}")


def load_engine(trt_file_path, verbose=False):
    """Build a TensorRT engine from a TRT file."""
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
    print('Loading TRT file from path {}...'.format(trt_file_path))
    with open(trt_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine
            
def main():
    args, cfg = merge_config()
    print(cfg)

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
        row_anchor = culane_row_anchor
        img_w, img_h = 640, 480
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
        row_anchor = tusimple_row_anchor
        img_w, img_h = 640, 480
    else:
        raise NotImplementedError


    trt_file_path = args.model
    if not os.path.isfile(trt_file_path):
        raise SystemExit('ERROR: file (%s) not found!' % trt_file_path)
    engine_file_path = args.model
    engine = load_engine(trt_file_path, args.verbose)

    h_inputs, h_outputs, bindings, stream = common.allocate_buffers(engine)

    camera_device = '/dev/video0'
    camera_width, camera_height, camera_fps = 1280, 720, 120
    output_width, output_height = camera_width, camera_height
    cap = cv2.VideoCapture(f'v4l2src device={camera_device} io-mode=2 ! image/jpeg, width={camera_width},height={camera_height},framerate={camera_fps}/1 ! jpegparse ! jpegdec ! videoconvert ! video/x-raw,width={output_width},height={output_height},format=BGR ! appsink max-buffers=1 drop=True')

    ########################################
    # MQTT
    ########################################
    global client
    client = connect_mqtt()
    client.loop_start()

    with engine.create_execution_context() as context:
        frame_number = 0
        while True:
            frame_timestamp = current_millis()
            frame_result = defaultdict(list) # init previous result.
            frame_result['timestamp'] = frame_timestamp
            frame_result['type'] = 'LaneDetection'
            frame_result['frame_number'] = frame_number
            frame_result['width'] = output_width
            frame_result['height'] = output_height
            lane_data = []

            _,frame = cap.read()
            t1 = time.time()
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img_transforms(img).numpy()

            h_inputs[0].host = img
            t3 = time.time()
            trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=h_inputs, outputs=h_outputs, stream=stream)
            t4 = time.time()
            
            out_j = trt_outputs[0].reshape(cfg.griding_num+1, cls_num_per_lane, cfg.num_lanes)
            
            prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)


            idx = np.arange(cfg.griding_num) + 1
            idx = idx.reshape(-1, 1, 1)

            loc = np.sum(prob * idx, axis=0)
            out_j = np.argmax(out_j, axis=0)
            loc[out_j == cfg.griding_num] = 0
            out_j = loc

            # import pdb; pdb.set_trace()
            vis = frame

            lane_data = defaultdict(list)
            for i in range(out_j.shape[1]):
                point_id = 0
                if np.sum(out_j[:, i] != 0) > 2:
                    for k in range(out_j.shape[0]):
                        if out_j[k, i] > 0:
                            ppp = (int(out_j[k, i] * col_sample_w * camera_width / 800) - 1, int(camera_height * (row_anchor[k]/288)) - 1 )
                            cv2.circle(vis, ppp, camera_width//300 ,color[i],-1)
                            lane_dict = {
                                'pid' : point_id,
                                'x' : ppp[0],
                                'y' : ppp[1]
                                }
                            lane_data[i].append(lane_dict)
                            point_id += 1

            frame_result['data'] = lane_data
            json_str = json.dumps(frame_result)
            print(json_str)
            publish(json_str)

            t2 = time.time()
            print('Inference time', (t4-t3)*1000)
            print('FPS', int(1/((t2-t1))))
            #cv2.imshow("OUTPUT", vis)
            #cv2.waitKey(1)
            frame_number += 1


if __name__ == '__main__':
    main()
