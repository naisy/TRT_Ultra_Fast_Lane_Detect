import torch, os, cv2
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
from configs.constant import culane_row_anchor, tusimple_row_anchor
import torch
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import time
from torch.autograd import Variable
import onnx


class laneDetection():
    def __init__(self):
        torch.backends.cudnn.benchmark = True
        args, cfg = merge_config()
        if cfg.dataset == 'CULane':
            cls_num_per_lane = 18
        elif cfg.dataset == 'Tusimple':
            cls_num_per_lane = 56
        else:
            raise NotImplementedError

        net = parsingNet(pretrained = False, backbone=cfg.backbone, cls_dim = (cfg.griding_num+1, cls_num_per_lane, cfg.num_lanes), use_aux=False).cuda()

        state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
        compatible_state_dict = {}
        for k, v in state_dict.items():
            if 'module.' in k:
                compatible_state_dict[k[7:]] = v
            else:
                compatible_state_dict[k] = v

        net.load_state_dict(compatible_state_dict, strict=False)
        
        #not recommend to uncommen this line
        net.eval()
 
        img_transforms = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        if cfg.dataset == 'CULane':
            img_w, img_h = 1280, 720
            #img_w, img_h = 1640, 590
            row_anchor = culane_row_anchor
        elif cfg.dataset == 'Tusimple':
            img_w, img_h = 1280, 720
            #img_w, img_h = 960, 480
            row_anchor = tusimple_row_anchor
        else:
            raise NotImplementedError

        scale_factor = 1
        color = [(255,0,0),(0,255,0),(0,0,255),(255,255,0)]  
        idx = np.arange(cfg.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)      

        col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]


        self.args = args
        self.cfg = cfg
        self.img_w = img_w
        self.img_h = img_h
        self.net = net
        self.row_anchor = row_anchor
        self.img_transforms = img_transforms
        self.scale_factor = scale_factor
        self.color = color
        self.idx = idx
        self.cpu_img = None
        self.gpu_img = None
        self.type = None
        self.gpu_output = None
        self.cpu_output = None
        self.col_sample_w = col_sample_w
        
        
    def setResolution(self, w, h):
        self.img_w = w
        self.img_h = h

    def getFrame(self, frame):
        self.cpu_img = frame

    def setScaleFactor(self, factor=1):
        self.scale_factor = factor

    def preprocess(self):
        tmp_img = cv2.cvtColor(self.cpu_img, cv2.COLOR_BGR2RGB)
        if self.scale_factor != 1:
            tmp_img = cv2.resize(tmp_img, (self.img_w//self.scale_factor, self.img_h//self.scale_factor))
        tmp_img = Image.fromarray(tmp_img)
        tmp_img = self.img_transforms(tmp_img)
        self.gpu_img = tmp_img.unsqueeze(0).cuda()

    def inference(self):
        self.gpu_output = self.net(self.gpu_img)

    def parseResults(self): 
        self.cpu_output = self.gpu_output[0].data.cpu().numpy()
        self.prob = scipy.special.softmax(self.cpu_output[:-1, :, :], axis=0)

        self.loc = np.sum(self.prob * self.idx, axis=0)
        self.cpu_output = np.argmax(self.cpu_output, axis=0)

        self.loc[self.cpu_output == self.cfg.griding_num] = 0
        #self.cpu_output = self.loc

        # import pdb; pdb.set_trace()
        vis = self.cpu_img
        for i in range(self.loc.shape[1]):
            if np.sum(self.loc[:, i] > 0) > 40:
                for k in range(self.loc.shape[0]):
                    if self.loc[k, i] > 0:
                        ppp = (int(self.loc[k, i] * self.col_sample_w * self.img_w / 800) - 1, int(self.img_h * (self.row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                        cv2.circle(vis,ppp,3, self.color[i], -1)
     
        cv2.imshow("output",vis)
        cv2.waitKey(1)
        return vis


