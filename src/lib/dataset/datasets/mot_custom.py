from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
from collections import defaultdict
from ..generic_dataset import GenericDataset
from utils.image import get_affine_transform, affine_transform

class MOT_Custom(GenericDataset):
  num_categories = 1
  default_resolution = [544, 960]
  # default_resolution = [272, 480]
  class_name = ['']
  max_objs = 256
  cat_ids = {1: 1, -1: -1}
  def __init__(self, opt, split):
    self.dataset_version = opt.dataset_version
    self.year = int(self.dataset_version[:2])
    print('Using MOT {} {}'.format(self.year, self.dataset_version))
    data_dir = os.path.join(opt.data_dir, 'mot{}'.format(self.year))

    if opt.dataset_version in ['17trainval', '17test']:
      ann_file = '{}.json'.format('train' if split == 'train' else \
        'test')
    elif opt.dataset_version == '17halftrain':
      ann_file = '{}.json'.format('train_half')
    elif opt.dataset_version == '17halfval':
      ann_file = '{}.json'.format('val_half')
    img_dir = os.path.join(data_dir, '{}'.format(
      'test' if 'test' in self.dataset_version else 'train'))
    
    print('ann_file', ann_file)
    ann_path = os.path.join(data_dir, 'annotations', ann_file)

    self.images = None
    # load image list and coco
    super(MOT_Custom, self).__init__(opt, split, ann_path, img_dir)

    self.num_samples = len(self.images)
    print('Loaded MOT {} {} {} samples'.format(
      self.dataset_version, split, self.num_samples))

  def __getitem__(self, index):
    opt = self.opt
    img, anns, img_info, img_path = self._load_data(index)


    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    s = max(img.shape[0], img.shape[1]) * 1.0 if not self.opt.not_max_crop \
      else np.array([img.shape[1], img.shape[0]], np.float32)
    aug_s, rot, flipped = 1, 0, 0
    if self.split == 'train':
      c, aug_s, rot = self._get_aug_param(c, s, width, height)
      s = s * aug_s
      if np.random.random() < opt.flip:
        flipped = 1
        img = img[:, ::-1, :]
        anns = self._flip_anns(anns, width)

    trans_input = get_affine_transform(
      c, s, rot, [opt.input_w, opt.input_h])
    trans_output = get_affine_transform(
      c, s, rot, [opt.output_w, opt.output_h])
    inp = self._get_input(img, trans_input)
    ret = {'image': inp}
    gt_det = {'bboxes': [], 'scores': [], 'clses': [], 'cts': []}

    pre_cts, track_ids = None, None
    if opt.tracking:
      # pre_image, pre_anns, frame_dist = self._load_pre_data(
      #   img_info['video_id'], img_info['frame_id'],
      #   img_info['sensor_id'] if 'sensor_id' in img_info else 1)

      pre_image, pre_anns, frame_dist = self._load_multi_pre_data(
        img_info['video_id'], img_info['frame_id'],
        img_info['sensor_id'] if 'sensor_id' in img_info else 1)
      pre_image = np.array(pre_image)

      # if flipped:
      #   pre_image = pre_image[:, ::-1, :].copy()
      #   pre_anns = self._flip_anns(pre_anns, width)

      if flipped:
        pre_image = pre_image[:, :, ::-1, :].copy()
        for pre_n, pre_ann in enumerate(pre_anns):
          pre_anns[pre_n] = self._flip_anns(pre_ann, width)

      # if opt.same_aug_pre and frame_dist != 0:
      #   trans_input_pre = trans_input
      #   trans_output_pre = trans_output
      # else:
      #   c_pre, aug_s_pre, _ = self._get_aug_param(
      #     c, s, width, height, disturb=True)
      #   s_pre = s * aug_s_pre
      #   trans_input_pre = get_affine_transform(
      #     c_pre, s_pre, rot, [opt.input_w, opt.input_h])
      #   trans_output_pre = get_affine_transform(
      #     c_pre, s_pre, rot, [opt.output_w, opt.output_h])

      trans_input_pres = []
      trans_output_pres = []
      for pre_t_i in range(0, pre_image.shape[0]):
        if opt.same_aug_pre and frame_dist[pre_t_i] != 0:
          trans_input_pre = trans_input
          trans_input_pres = trans_input_pres + [trans_input_pre]
          trans_output_pre = trans_output
          trans_output_pres = trans_output_pres + [trans_output_pre]
        else:
          c_pre, aug_s_pre, _ = self._get_aug_param(
            c, s, width, height, disturb=True)
          s_pre = s * aug_s_pre
          trans_input_pre = get_affine_transform(
            c_pre, s_pre, rot, [opt.input_w, opt.input_h])
          trans_output_pre = get_affine_transform(
            c_pre, s_pre, rot, [opt.output_w, opt.output_h])


      #pre_imgs = self._get_input(pre_image, trans_input_pre)

      pre_imgs = []
      for pre_t_i in range(0, pre_image.shape[0]):
        pre_img = self._get_input(pre_image[pre_t_i, :, :, :], trans_input_pre)
        pre_imgs = pre_imgs + [pre_img]
      pre_imgs = np.array(pre_imgs)
      ret['pre_imgs'] = pre_imgs

      pre_hm, pre_cts, track_ids = self._get_pre_dets(
        pre_anns[-1], trans_input_pre, trans_output_pre)


      ret['pre_img'] = pre_imgs[-1]
      if opt.pre_hm:
        ret['pre_hm'] = pre_hm

    ### init samples
    self._init_ret(ret, gt_det)
    calib = self._get_calib(img_info, width, height)

    num_objs = min(len(anns), self.max_objs)
    for k in range(num_objs):
      ann = anns[k]
      cls_id = int(self.cat_ids[ann['category_id']])
      if cls_id > self.opt.num_classes or cls_id <= -999:
        continue
      bbox, bbox_amodal = self._get_bbox_output(
        ann['bbox'], trans_output, height, width)
      if cls_id <= 0 or ('iscrowd' in ann and ann['iscrowd'] > 0):
        self._mask_ignore_or_crowd(ret, cls_id, bbox)
        continue
      self._add_instance(
        ret, gt_det, k, cls_id, bbox, bbox_amodal, ann, trans_output, aug_s,
        calib, pre_cts, track_ids)

    if self.opt.debug > 0:
      gt_det = self._format_gt_det(gt_det)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_info['id'],
              'img_path': img_path, 'calib': calib,
              'flipped': flipped}
      ret['meta'] = meta
    return ret

  def _load_multi_pre_data(self, video_id, frame_id, sensor_id=1, pre_num=4):
    img_infos = self.video_to_images[video_id]
    # If training, random sample nearby frames as the "previoud" frame
    # If testing, get the exact prevous frame

    if 'train' in self.split:
      img_ids = [(img_info['id'], img_info['frame_id']) \
          for img_info in img_infos \
          if abs(img_info['frame_id'] - frame_id) < self.opt.max_frame_dist and \
          (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
    else:
      img_ids = [(img_info['id'], img_info['frame_id']) \
          for img_info in img_infos \
            if (img_info['frame_id'] - frame_id) == -1 and \
            (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
      if len(img_ids) == 0:
        img_ids = [(img_info['id'], img_info['frame_id']) \
            for img_info in img_infos \
            if (img_info['frame_id'] - frame_id) == 0 and \
            (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]

    rand_ids = np.random.choice(len(img_ids), size=pre_num)
    # rand_id = np.random.choice(len(img_ids))
    imgs = []
    anns_s = []
    frame_dists = []
    for rand_id in rand_ids:
      img_id, pre_frame_id = img_ids[rand_id]
      frame_dist = abs(frame_id - pre_frame_id)
      img, anns, _, _ = self._load_image_anns(img_id, self.coco, self.img_dir)
      imgs = imgs + [img]
      anns_s = anns_s + [anns]
      frame_dists = frame_dists + [frame_dist]

    return imgs, anns_s, frame_dists

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    results_dir = os.path.join(save_dir, 'results_mot{}'.format(self.dataset_version))
    if not os.path.exists(results_dir):
      os.mkdir(results_dir)
    for video in self.coco.dataset['videos']:
      video_id = video['id']
      file_name = video['file_name']
      out_path = os.path.join(results_dir, '{}.txt'.format(file_name))
      f = open(out_path, 'w')
      images = self.video_to_images[video_id]
      tracks = defaultdict(list)
      for image_info in images:
        if not (image_info['id'] in results):
          continue
        result = results[image_info['id']]
        frame_id = image_info['frame_id']
        for item in result:
          if not ('tracking_id' in item):
            item['tracking_id'] = np.random.randint(100000)
          if item['active'] == 0:
            continue
          tracking_id = item['tracking_id']
          bbox = item['bbox']
          bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
          tracks[tracking_id].append([frame_id] + bbox)
      rename_track_id = 0
      for track_id in sorted(tracks):
        rename_track_id += 1
        for t in tracks[track_id]:
          f.write('{},{},{:.2f},{:.2f},{:.2f},{:.2f},-1,-1,-1,-1\n'.format(
            t[0], rename_track_id, t[1], t[2], t[3]-t[1], t[4]-t[2]))
      f.close()
  
  def run_eval(self, results, save_dir):
    self.save_results(results, save_dir)
    gt_type_str = '{}'.format(
                '_train_half' if '17halftrain' in self.opt.dataset_version \
                else '_val_half' if '17halfval' in self.opt.dataset_version \
                else '')
    gt_type_str = '_val_half' if self.year in [16, 19] else gt_type_str
    gt_type_str = '--gt_type {}'.format(gt_type_str) if gt_type_str != '' else \
      ''
    os.system('python tools/eval_motchallenge.py ' + \
              '../data/mot{}/{}/ '.format(self.year, 'train') + \
              '{}/results_mot{}/ '.format(save_dir, self.dataset_version) + \
              gt_type_str + ' --eval_official')
