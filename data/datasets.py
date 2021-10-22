import os
import os.path as osp
from typing import Dict, List, Tuple
import cv2
import numpy as np
import random

import torch
from torch.utils import data

from h5dataloader.pytorch import HDF5Dataset
from h5dataloader.pytorch.structure import DTYPE_TORCH, CONVERT_TORCH
from h5dataloader.common.structure import *
from pointsmap import invertTransform, combineTransforms


class KittiDataset(data.Dataset):
    def __init__(self, root='./datasets/kitti', data_file='train.list', phase='train', joint_transform=None):

        self.root = root
        self.data_file = data_file
        self.files = []
        self.joint_transform = joint_transform
        self.phase = phase
        self.no_gt = False

        with open(osp.join(self.root, self.data_file), 'r') as f:
            data_list = f.read().split('\n')
            for data in data_list:
                if len(data) == 0:
                    continue

                data_info = data.split(' ')

                if len(data_info) == 3:
                    self.files.append({
                        "rgb": data_info[0],
                        "sparse": data_info[1],
                        "gt": data_info[2]
                    })
                else:
                    self.files.append({
                        "rgb": data_info[0],
                        "sparse": data_info[1],
                    })
                    self.no_gt = True
        self.nSamples = len(self.files)

    def __len__(self):
        return self.nSamples

    def read_calib_file(self, path):
        # taken from https://github.com/hunse/kitti
        float_chars = set("0123456789.e+- ")
        data = {}
        with open(path, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                value = value.strip()
                data[key] = value
                if float_chars.issuperset(value):
                    # try to cast to float array
                    try:
                        data[key] = np.array(
                            list(map(float, value.split(' '))))
                    except ValueError:
                        # casting error: data[key] already eq. value, so pass
                        pass

        return data

    def read_data(self, index):

        sparse = cv2.imread(
            osp.join(self.root, self.files[index]['sparse']), cv2.IMREAD_UNCHANGED)
        if not self.no_gt:
            gt = cv2.imread(
                osp.join(self.root, self.files[index]['gt']), cv2.IMREAD_UNCHANGED)
        else:
            gt = sparse
        img = cv2.imread(
            osp.join(self.root, self.files[index]['rgb']), cv2.IMREAD_COLOR)

        h, w = img.shape[0], img.shape[1]

        assert h == gt.shape[0] and w == gt.shape[1]
        assert h == sparse.shape[0] and w == sparse.shape[1]
        # read intrinsics
        if self.phase == 'train':
            calib_dir = self.files[index]['rgb'][0:14]
            cam2cam = self.read_calib_file(
                osp.join(self.root, calib_dir, 'calib_cam_to_cam.txt'))
            P2_rect = cam2cam['P_rect_02'].reshape(3, 4)
            K = P2_rect[:, :3].astype(np.float32)

        elif self.phase in ['val', 'test']:
            calib_name = self.files[index]['sparse'].replace('_velodyne_raw_', '_image_').replace(
                'png', 'txt').replace('velodyne_raw', 'intrinsics')
            with open(osp.join(self.root, calib_name), 'r') as f:
                calib = f.readline()
                calib = calib.splitlines()[0].rstrip().split(' ')
            K = np.zeros((3, 3), dtype=np.float32)
            for i in range(3):
                for j in range(3):
                    K[i, j] = float(calib[i*3+j])
        else:
            K = np.zeros((3, 3), dtype=np.float32)

        if self.no_gt:
            assert w == 1216 and h == 352
        else:
            H = 352
            s = int(round(w - 1216) / 2)
            img = img[h-H:, s:s+1216]
            gt = gt[h-H:, s:s+1216]
            sparse = sparse[h-H:, s:s+1216]
            if self.phase == 'train':
                K[0, 2] = K[0, 2] - s
                K[1, 2] = K[1, 2] - (h-H)

        return img, gt, sparse, K

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img, gt, sparse, K = self.read_data(index)

        if self.joint_transform is not None:
            img, gt, sparse = self.joint_transform((img, gt, sparse, 'kitti'))
        data = {}
        data['img'] = img
        data['gt'] = gt
        data['sparse'] = sparse
        data['K'] = K

        return data


def get_dataset(root='./datasets', data_file='train.list', dataset='kitti',
                phase='train', joint_transform=None):

    return KittiDataset(osp.join(root, dataset), data_file, phase, joint_transform)


NORMALIZE_INF = 2.0


class H5_Train_Dataset(HDF5Dataset):
    def __init__(self, h5_paths: List[str], config: str, quiet: bool = True, block_size: int = 0, use_mods: Tuple[int, int] = None,
                 visibility_filter_radius: int = 0, visibility_filter_threshold: float = 3.0, depth_norm_aug_rate: float = None, tr_err_range: float = 2.0, rot_err_range: float = 10.0) -> None:
        super(H5_Train_Dataset, self).__init__(h5_paths, config, quiet, block_size,
                                               use_mods, visibility_filter_radius, visibility_filter_threshold)
        # Depth Normalization Augmentation
        self.depth_norm_aug_rate: float = depth_norm_aug_rate
        # Random Pose Error
        self.tr_err_range: float = np.array(tr_err_range, dtype=np.float32)
        self.rot_err_range: float = np.deg2rad(rot_err_range)
        self.minibatch['sparse'][CONFIG_TAG_TF].append(('', False))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # Depth Normalization Augmentation
        self.depth_norm_c: float = 1.0
        if self.depth_norm_aug_rate is not None:
            self.depth_norm_c += (np.random.rand() * 2.0 -
                                  1.0) * self.depth_norm_aug_rate
        # Random Pose Error
        rot_vec: np.ndarray = np.random.rand(3)
        rot_vec /= np.linalg.norm(rot_vec)
        rot_abs: float = np.random.rand() * self.rot_err_range
        self.q_err: np.ndarray = self.__vec2quat(rot_vec, rot_abs)
        self.tr_err: np.ndarray = np.random.rand(3) * self.tr_err_range
        # Get Items.
        items: Dict[str, torch.Tensor] = super().__getitem__(index)
        # Add Pose Error.
        tr_norm: float = 1.0
        if self.minibatch['sparse'][CONFIG_TAG_NORMALIZE] is True:
            tr_norm *= self.minibatch['sparse'][CONFIG_TAG_RANGE][1] * \
                self.depth_norm_c
        return items

    def __vec2quat(self, vec: np.ndarray, abs: float) -> np.ndarray:
        # Rotation vector to Quaternion.
        xyz: np.ndarray = vec * np.sin(abs * 0.5)
        return np.append(xyz, np.cos(abs * 0.5))

    def depth_common(self, src: np.ndarray, minibatch_config: Dict[str, Union[str, Dict[str, str], List[int], bool, List[float]]]) -> np.ndarray:
        # Depth Normalization Augmentation
        dst = src
        if minibatch_config[CONFIG_TAG_NORMALIZE] is True:
            range_min, range_max = minibatch_config[CONFIG_TAG_RANGE][:2]
            range_max *= self.depth_norm_c
            dst = np.where(range_max < dst, NORMALIZE_INF,
                           (dst - range_min) / (range_max - range_min))
        shape = minibatch_config[CONFIG_TAG_SHAPE]
        if shape != dst.shape[:2]:
            dst = cv2.resize(dst, dsize=(
                shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return dst

    def create_pose_from_pose(self, key: str, link_idx: int, minibatch_config: Dict[str, Union[str, Dict[str, str], List[int], bool, List[float]]]) -> np.ndarray:
        """create_pose_from_pose
        "pose"を生成する
        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定
        Returns:
            np.ndarray: [tx, ty, tz, qx, qy, qz, qw]
        """
        translations = [np.array([0.0, 0.0, 0.0], dtype=np.float32)]
        quaternions = [np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)]
        for child_frame_id, invert in minibatch_config[CONFIG_TAG_TF]:
            tf_data: Dict[str, str] = self.tf[CONFIG_TAG_DATA].get(child_frame_id)

            if tf_data is None:
                # Random Pose Error
                trns: np.ndarray = self.tr_err
                qtrn: np.ndarray = self.q_err
            else:
                h5_key = tf_data[CONFIG_TAG_KEY]
                if h5_key[0] == '/':
                    h5_key = str(link_idx) + h5_key
                else:
                    h5_key = os.path.join(key, h5_key)
                trns: np.ndarray = self.h5links[h5_key][SUBTYPE_TRANSLATION][()]
                qtrn: np.ndarray = self.h5links[h5_key][SUBTYPE_ROTATION][()]

            if invert is True:
                trns, qtrn = invertTransform(translation=trns, quaternion=qtrn)

            translations.append(trns)
            quaternions.append(qtrn)

        translation, quaternion = combineTransforms(translations=translations, quaternions=quaternions)

        return np.concatenate([translation, quaternion])


class H5_Test_Dataset(HDF5Dataset):
    def __init__(self, h5_paths: List[str], config: str, quiet: bool = True, block_size: int = 0, use_mods: Tuple[int, int] = None,
                 visibility_filter_radius: int = 0, visibility_filter_threshold: float = 3.0, tr_err_range: float = 2.0, rot_err_range: float = 10.0) -> None:
        super(H5_Test_Dataset, self).__init__(h5_paths, config, quiet, block_size,
                                              use_mods, visibility_filter_radius, visibility_filter_threshold)
        if self.minibatch.get('pose_err') is None:
            self.random_pose: bool = True
            # Random Pose Error
            self.tr_err_range: float = np.array(tr_err_range, dtype=np.float32)
            self.rot_err_range: float = np.deg2rad(rot_err_range)
            self.minibatch['sparse'][CONFIG_TAG_TF].append(('', False))
            rot_vec: np.ndarray = np.random.rand(self.length, 3)
            rot_vec /= np.linalg.norm(rot_vec, axis=1, keepdims=True)
            rot_abs: float = np.random.rand(self.length) * self.rot_err_range
            self.q_err_list: np.ndarray = self.__vec2quat(rot_vec, rot_abs)
            self.tr_err_list: np.ndarray = np.random.rand(self.length, 3) * self.tr_err_range
        else:
            self.random_pose: bool = False

    def __vec2quat(self, vec: np.ndarray, abs: float) -> np.ndarray:
        # Rotation vector to Quaternion.
        xyz: np.ndarray = vec * \
            np.sin(np.repeat(abs[:, np.newaxis], 3, axis=1) * 0.5)
        return np.append(xyz, np.cos(abs * 0.5)[:, np.newaxis], axis=1)

    def create_pose_from_pose(self, key: str, link_idx: int, minibatch_config: Dict[str, Union[str, Dict[str, str], List[int], bool, List[float]]]) -> np.ndarray:
        """create_pose_from_pose
        "pose"を生成する
        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定
        Returns:
            np.ndarray: [tx, ty, tz, qx, qy, qz, qw]
        """
        translations = [np.array([0.0, 0.0, 0.0], dtype=np.float32)]
        quaternions = [np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)]

        for child_frame_id, invert in minibatch_config[CONFIG_TAG_TF]:
            tf_data: Dict[str, str] = self.tf[CONFIG_TAG_DATA].get(
                child_frame_id)
            if tf_data is None:
                # Random Pose Error
                trns: np.ndarray = self.tr_err
                qtrn: np.ndarray = self.q_err
            else:
                h5_key = tf_data[CONFIG_TAG_KEY]
                if h5_key[0] == '/':
                    h5_key = str(link_idx) + h5_key
                else:
                    h5_key = os.path.join(key, h5_key)
                trns: np.ndarray = self.h5links[h5_key][SUBTYPE_TRANSLATION][()]
                qtrn: np.ndarray = self.h5links[h5_key][SUBTYPE_ROTATION][()]

            if invert is True:
                trns, qtrn = invertTransform(translation=trns, quaternion=qtrn)

            translations.append(trns)
            quaternions.append(qtrn)

        translation, quaternion = combineTransforms(translations=translations, quaternions=quaternions)

        return np.concatenate([translation, quaternion])

    def __getitem__(self, index: int) -> dict:
        if self.random_pose is True:
            self.tr_err = self.tr_err_list[index]
            self.q_err = self.q_err_list[index]

        items: Dict[str, torch.Tensor] = super().__getitem__(index)
        tr_norm: float = 1.0

        if self.minibatch['sparse'][CONFIG_TAG_NORMALIZE] is True:
            tr_norm *= self.minibatch['sparse'][CONFIG_TAG_RANGE][1]

        if self.random_pose is True:
            items['pose_err'] = torch.from_numpy(CONVERT_TORCH[TYPE_POSE](
                DTYPE_TORCH[TYPE_POSE](np.concatenate([self.tr_err / tr_norm, self.q_err]))))
        else:
            items['pose_err'][:3] = items['pose_err'][:3] / tr_norm

        return items
