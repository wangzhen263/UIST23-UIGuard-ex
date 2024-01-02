# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
fastRCNN_root = "/Users/che444/Desktop/DPCODE-CLEAN/finalCode/elementDetection/FASTER_RCNN"
sys.path.insert(0, fastRCNN_root)

# import _init_paths
import os
import json
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
from imageio import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


# args = argparse.Namespace()
# args.dataset = "dataset"
# args.cfg = "cfgs/res101.yml"
# args.net = "res101"
# args.set = "net"
# args.image_dir = "images"
# args.load_dir = "images"
# args.output_dir = "images"
# args.cuda = "images"
# args.mGPUs = "images"
# args.class_agnostic = True
# args.parallel_type = 0
# args.pretrained_model_name = "faster_rcnn.pth"
# args.bs = 1
# args.vis = False
# args.webcam_num = -1


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/res101.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models',
                      default="/srv/share/jyang375/models")
  parser.add_argument('--image_dir', dest='image_dir',
                      help='directory to load images for demo',
                      default="images")
  parser.add_argument('--output_dir', dest='output_dir',
                      help='directory to save json results',
                      default="images")
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic', default=True,
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--pretrained_model_name', dest='pretrained_model_name',
                      help='pretrained_model_name',
                      default="faster_rcnn.pth", type=str)
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  parser.add_argument('--webcam_num', dest='webcam_num',
                      help='webcam ID number',
                      default=-1, type=int)

  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)


##### MAIN
args = parse_args()

args.dataset = "ricoOriText"
args.net = "res101"
args.load_dir = os.path.join(fastRCNN_root, "results/run")
args.pretrained_model_name = "faster_rcnn.pth"
args.cuda = torch.cuda.is_available()



print('Called with args:')
print(args)
if "ricoDefault" in args.dataset:
    args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
elif 'ricoCustomized' in args.dataset:
    args.set_cfgs = ['ANCHOR_SCALES', '[2, 4, 8, 16]', 'ANCHOR_RATIOS', '[1, 2, 4, 8]', 'MAX_NUM_GT_BOXES', '30']
elif "rico" in args.dataset:
    args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[1, 2, 4, 8]']
else:
  raise Exception("Value error", "Dataset not exists")

args.cfg_file = "{}/cfgs/{}.yml".format(fastRCNN_root, args.net)



if args.cfg_file is not None:
  cfg_from_file(args.cfg_file)
if args.set_cfgs is not None:
  cfg_from_list(args.set_cfgs)

cfg.USE_GPU_NMS = args.cuda

print('Using config:')
pprint.pprint(cfg)
np.random.seed(cfg.RNG_SEED)

# train set
# -- Note: Use validation set and disable the flipped to enable faster loading.
input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
if not os.path.exists(input_dir):
  raise Exception('There is no input directory for loading network from ' + input_dir)
load_name = os.path.join(input_dir,args.pretrained_model_name)

pascal_classes = np.asarray(['__background__',
                     'Button', 'CheckBox', 'Chronometer',
                     'EditText', 'ImageButton', 'ImageView',
                     'ProgressBar', 'RadioButton', 'RatingBar',
                     'SeekBar', 'Spinner', 'Switch', 'ToggleButton',
                     'VideoView'
              ])

if "Text" in args.dataset:
    pascal_classes = np.asarray(['__background__',
                     'Button', 'CheckBox', 'Chronometer',
                     'EditText', 'ImageButton', 'ImageView',
                     'ProgressBar', 'RadioButton', 'RatingBar',
                     'SeekBar', 'Spinner', 'Switch', 'ToggleButton',
                     'VideoView', 'TextView'
              ])
# initilize the network here.
fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)

fasterRCNN.create_architecture()

print("load checkpoint %s" % (load_name))
if args.cuda > 0:
  checkpoint = torch.load(load_name)
else:
  checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
fasterRCNN.load_state_dict(checkpoint['model'])
if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


print('load model successfully!')

# pdb.set_trace()

print("load checkpoint %s" % (load_name))

# initilize the tensor holder here.
im_data = torch.FloatTensor(1)
im_info = torch.FloatTensor(1)
num_boxes = torch.LongTensor(1)
gt_boxes = torch.FloatTensor(1)

# ship to cuda
if args.cuda > 0:
  im_data = im_data.cuda()
  im_info = im_info.cuda()
  num_boxes = num_boxes.cuda()
  gt_boxes = gt_boxes.cuda()

# make variable
with torch.no_grad():
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

if args.cuda > 0:
  cfg.CUDA = True

if args.cuda > 0:
  fasterRCNN.cuda()

fasterRCNN.eval()

import glob
def test_single(image_path, output_dict_path=".", vis=False):
  # if not os.path.exists(output_dir):
  #   os.makedirs(output_dir)
  start = time.time()
  max_per_image = 100
  thresh = 0.05
    
  # imglist = os.listdir(args.image_dir)
  #### mobbin
  # imglist = glob.glob(os.path.join(image_dir, "**/**"), recursive=True)
  # imglist = [png for png in imglist if png.split(".")[-1].lower() in ["jpg", "png", "jpeg"] and "det" not in png]
  imglist = [image_path]
  num_images = len(imglist)

  print('Loaded Photo: {} images.'.format(num_images))

  results_dict = {}
  while (num_images >= 0):
      total_tic = time.time()
      num_images -= 1

      im_file = imglist[num_images]
      im_in = np.array(imread(im_file, as_gray=False, pilmode="RGB"))
      if imglist[num_images] not in results_dict:
        results_dict[imglist[num_images]] = []
      if len(im_in.shape) == 2:
        im_in = im_in[:,:,np.newaxis]
        im_in = np.concatenate((im_in,im_in,im_in), axis=2)
      # rgb -> bgr
      im = im_in[:,:,::-1]

      blobs, im_scales = _get_image_blob(im)
      assert len(im_scales) == 1, "Only single-image batch implemented"
      im_blob = blobs
      im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

      im_data_pt = torch.from_numpy(im_blob)
      im_data_pt = im_data_pt.permute(0, 3, 1, 2)
      im_info_pt = torch.from_numpy(im_info_np)

      with torch.no_grad():
        im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.resize_(1, 1, 5).zero_()
        num_boxes.resize_(1).zero_()

      # pdb.set_trace()
      det_tic = time.time()

      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

      scores = cls_prob.data
      boxes = rois.data[:, :, 1:5]

      if cfg.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data
          if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                if args.cuda > 0:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                box_deltas = box_deltas.view(1, -1, 4)
            else:
                if args.cuda > 0:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))

      pred_boxes /= im_scales[0]

      scores = scores.squeeze()
      pred_boxes = pred_boxes.squeeze()
      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()

      if vis:
          im2show = np.copy(im)
      for j in xrange(1, len(pascal_classes)):
          inds = torch.nonzero(scores[:,j]>thresh).view(-1)
          # if there is det
          if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
              cls_boxes = pred_boxes[inds, :]
            else:
              cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            # keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            if vis:
              im2show = vis_detections(im2show, pascal_classes[j], cls_dets.cpu().numpy(), 0.5)
            
            tmp_class = pascal_classes[j]
            dets = cls_dets.cpu().numpy()

            for i in range(np.minimum(10, dets.shape[0])):
                bbox = tuple(int(np.round(x)) for x in dets[i, :4]) #[x1,y1,x2,y2]
                score = dets[i, -1]
                if score > 0.5:
                    ## mobbin
                    results_dict[imglist[num_images]].append({"category":tmp_class, "bbox":bbox, "score":round(score.item(), 3)})
                    # print("+", tmp_class, bbox, score, score.item())

      if vis:
          cv2.imshow('1-faster RCNN before NMS', im2show)
          cv2.waitKey(0)
          result_path = output_dict_path.replace(".json", ".jpg")
          cv2.imwrite(result_path, im2show)


      misc_toc = time.time()
      nms_time = misc_toc - misc_tic
  with open(output_dict_path, "w") as f:
    json.dump(results_dict, f)
  return results_dict


