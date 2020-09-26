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
from utils.image import get_affine_transform, affine_transform, gaussian_radius, draw_umich_gaussian
import math

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
        img_info['sensor_id'] if 'sensor_id' in img_info else 1, pre_num=opt.History_T)
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
          trans_output_pre = trans_output
          trans_input_pres = trans_input_pres + [trans_input_pre]
          trans_output_pres = trans_output_pres + [trans_output_pre]
        else:
          c_pre, aug_s_pre, _ = self._get_aug_param(
            c, s, width, height, disturb=True)
          s_pre = s * aug_s_pre
          trans_input_pre = get_affine_transform(
            c_pre, s_pre, rot, [opt.input_w, opt.input_h])
          trans_output_pre = get_affine_transform(
            c_pre, s_pre, rot, [opt.output_w, opt.output_h])
          trans_input_pres = trans_input_pres + [trans_input_pre]
          trans_output_pres = trans_output_pres + [trans_output_pre]

      #pre_imgs = self._get_input(pre_image, trans_input_pre)
      assert pre_image.shape[0] == opt.History_T

      pre_imgs = []
      for pre_t_i in range(0, pre_image.shape[0]):
        pre_img = self._get_input(pre_image[pre_t_i, :, :, :], trans_input_pres[pre_t_i])
        pre_imgs = pre_imgs + [pre_img]
      pre_imgs = np.array(pre_imgs)
      ret['pre_imgs'] = pre_imgs

      # pre_hm, pre_cts, track_ids = self._get_pre_dets(
      #   pre_anns, trans_input_pre, trans_output_pre)

      pre_hms = []
      pre_cts_s = []
      track_ids_s = []
      for pre_t_i in range(0, pre_image.shape[0]):
        pre_hm, pre_cts, track_ids = self._get_pre_dets(
          pre_anns[pre_t_i], trans_input_pres[pre_t_i], trans_output_pres[pre_t_i])
        # print(track_ids)
        pre_hms = pre_hms + [pre_hm]
        pre_cts_s = pre_cts_s + [pre_cts]
        track_ids_s = track_ids_s + [track_ids]

      pre_hm = pre_hms[-1]
      pre_cts = pre_cts_s[-1]
      track_ids = track_ids_s[-1]

      # for cts in pre_cts:
      #   print(pre_hm[0, round(cts[1] * 4), round(cts[0] * 4)])
      #   print(pre_hm.shape)
      #   # print(cts)


      # print(len(pre_anns[-1]))
      pre_lens = []
      for pre_len_i in range(0, pre_image.shape[0]):
        pre_lens = pre_lens + [len(pre_cts_s[pre_len_i]) if len(pre_cts_s[pre_len_i])<=opt.K else opt.K]
      ret['pre_len'] = np.array(pre_lens, dtype=np.int)

      pre_cts_fix = np.zeros((opt.History_T, opt.K, 2), dtype=np.float)

      for pre_len_i in range(0, pre_image.shape[0]):
        centers = np.array(pre_cts_s[pre_len_i])
        num = ret['pre_len'][pre_len_i] if ret['pre_len'][pre_len_i] <= opt.K else opt.K
        pre_cts_fix[pre_len_i, 0:num, :] = centers[0:num, :]

      ret['pre_cts_fix'] = pre_cts_fix
      ret['pre_img'] = pre_imgs[-1]
      if opt.pre_hm:
        ret['pre_hm'] = pre_hm

    ### init samples
    self._init_ret_new_and_old(ret, gt_det)
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
      # print('before:')
      # print(ret['hm'].mean())
      self._add_instance_new_and_old(
        ret, gt_det, k, cls_id, bbox, bbox_amodal, ann, trans_output, aug_s,
        calib, pre_cts, track_ids)
      # print('after:')
      # print(ret['hm'].mean())
    # print('old:')
    # print(ret['hm_old'].mean())
    # print('new:')
    # print(ret['hm_new'].mean())
    # print('hm:')
    # print(ret['hm'].mean())

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

  def _add_instance_new_and_old(
          self, ret, gt_det, k, cls_id, bbox, bbox_amodal, ann, trans_output,
          aug_s, calib, pre_cts=None, track_ids=None):
    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
    if h <= 0 or w <= 0:
      return
    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
    radius = max(0, int(radius))
    ct = np.array(
      [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
    ct_int = ct.astype(np.int32)
    ret['cat'][k] = cls_id - 1
    ret['mask'][k] = 1
    if 'wh' in ret:
      ret['wh'][k] = 1. * w, 1. * h
      ret['wh_mask'][k] = 1
    ret['ind'][k] = ct_int[1] * self.opt.output_w + ct_int[0]
    ret['reg'][k] = ct - ct_int
    ret['reg_mask'][k] = 1
    draw_umich_gaussian(ret['hm'][cls_id - 1], ct_int, radius)

    gt_det['bboxes'].append(
      np.array([ct[0] - w / 2, ct[1] - h / 2,
                ct[0] + w / 2, ct[1] + h / 2], dtype=np.float32))
    gt_det['scores'].append(1)
    gt_det['clses'].append(cls_id - 1)
    gt_det['cts'].append(ct)


    if 'tracking' in self.opt.heads:
      # print('True')
      # print(ann['track_id'])
      # print(track_ids)
      if ann['track_id'] in track_ids:
        pre_ct = pre_cts[track_ids.index(ann['track_id'])]
        ret['tracking_mask'][k] = 1
        ret['tracking'][k] = pre_ct - ct_int
        gt_det['tracking'].append(ret['tracking'][k])
        # print('draw')
        draw_umich_gaussian(ret['hm_old'][cls_id - 1], ct_int, radius)

      else:
        gt_det['tracking'].append(np.zeros(2, np.float32))

        draw_umich_gaussian(ret['hm_new'][cls_id - 1], ct_int, radius)

    if 'ltrb' in self.opt.heads:
      ret['ltrb'][k] = bbox[0] - ct_int[0], bbox[1] - ct_int[1], \
                       bbox[2] - ct_int[0], bbox[3] - ct_int[1]
      ret['ltrb_mask'][k] = 1

    if 'ltrb_amodal' in self.opt.heads:
      ret['ltrb_amodal'][k] = \
        bbox_amodal[0] - ct_int[0], bbox_amodal[1] - ct_int[1], \
        bbox_amodal[2] - ct_int[0], bbox_amodal[3] - ct_int[1]
      ret['ltrb_amodal_mask'][k] = 1
      gt_det['ltrb_amodal'].append(bbox_amodal)

    if 'nuscenes_att' in self.opt.heads:
      if ('attributes' in ann) and ann['attributes'] > 0:
        att = int(ann['attributes'] - 1)
        ret['nuscenes_att'][k][att] = 1
        ret['nuscenes_att_mask'][k][self.nuscenes_att_range[att]] = 1
      gt_det['nuscenes_att'].append(ret['nuscenes_att'][k])

    if 'velocity' in self.opt.heads:
      if ('velocity' in ann) and min(ann['velocity']) > -1000:
        ret['velocity'][k] = np.array(ann['velocity'], np.float32)[:3]
        ret['velocity_mask'][k] = 1
      gt_det['velocity'].append(ret['velocity'][k])

    if 'hps' in self.opt.heads:
      self._add_hps(ret, k, ann, gt_det, trans_output, ct_int, bbox, h, w)

    if 'rot' in self.opt.heads:
      self._add_rot(ret, ann, k, gt_det)

    if 'dep' in self.opt.heads:
      if 'depth' in ann:
        ret['dep_mask'][k] = 1
        ret['dep'][k] = ann['depth'] * aug_s
        gt_det['dep'].append(ret['dep'][k])
      else:
        gt_det['dep'].append(2)

    if 'dim' in self.opt.heads:
      if 'dim' in ann:
        ret['dim_mask'][k] = 1
        ret['dim'][k] = ann['dim']
        gt_det['dim'].append(ret['dim'][k])
      else:
        gt_det['dim'].append([1, 1, 1])

    if 'amodel_offset' in self.opt.heads:
      if 'amodel_center' in ann:
        amodel_center = affine_transform(ann['amodel_center'], trans_output)
        ret['amodel_offset_mask'][k] = 1
        ret['amodel_offset'][k] = amodel_center - ct_int
        gt_det['amodel_offset'].append(ret['amodel_offset'][k])
      else:
        gt_det['amodel_offset'].append([0, 0])


  def _init_ret_new_and_old(self, ret, gt_det):
    max_objs = self.max_objs * self.opt.dense_reg
    ret['hm'] = np.zeros(
      (self.opt.num_classes, self.opt.output_h, self.opt.output_w),
      np.float32)

    ret['hm_new'] = np.zeros(
      (self.opt.num_classes, self.opt.output_h, self.opt.output_w),
      np.float32)
    ret['hm_old'] = np.zeros(
      (self.opt.num_classes, self.opt.output_h, self.opt.output_w),
      np.float32)

    ret['ind'] = np.zeros((max_objs), dtype=np.int64)
    ret['cat'] = np.zeros((max_objs), dtype=np.int64)
    ret['mask'] = np.zeros((max_objs), dtype=np.float32)

    regression_head_dims = {
      'reg': 2, 'wh': 2, 'tracking': 2, 'ltrb': 4, 'ltrb_amodal': 4,
      'nuscenes_att': 8, 'velocity': 3, 'hps': self.num_joints * 2,
      'dep': 1, 'dim': 3, 'amodel_offset': 2}

    for head in regression_head_dims:
      if head in self.opt.heads:
        ret[head] = np.zeros(
          (max_objs, regression_head_dims[head]), dtype=np.float32)
        ret[head + '_mask'] = np.zeros(
          (max_objs, regression_head_dims[head]), dtype=np.float32)
        gt_det[head] = []

    if 'hm_hp' in self.opt.heads:
      num_joints = self.num_joints
      ret['hm_hp'] = np.zeros(
        (num_joints, self.opt.output_h, self.opt.output_w), dtype=np.float32)
      ret['hm_hp_mask'] = np.zeros(
        (max_objs * num_joints), dtype=np.float32)
      ret['hp_offset'] = np.zeros(
        (max_objs * num_joints, 2), dtype=np.float32)
      ret['hp_ind'] = np.zeros((max_objs * num_joints), dtype=np.int64)
      ret['hp_offset_mask'] = np.zeros(
        (max_objs * num_joints, 2), dtype=np.float32)
      ret['joint'] = np.zeros((max_objs * num_joints), dtype=np.int64)

    if 'rot' in self.opt.heads:
      ret['rotbin'] = np.zeros((max_objs, 2), dtype=np.int64)
      ret['rotres'] = np.zeros((max_objs, 2), dtype=np.float32)
      ret['rot_mask'] = np.zeros((max_objs), dtype=np.float32)
      gt_det.update({'rot': []})

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
