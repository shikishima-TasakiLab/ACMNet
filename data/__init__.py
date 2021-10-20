from typing import Tuple
import torch
from torch.utils.data.dataloader import DataLoader
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor
from data.datasets import get_dataset
from data.transform import RandomImgAugment

from h5dataloader.pytorch import HDF5Dataset
from data.datasets import H5_Train_Dataset, H5_Test_Dataset

def create_test_dataloader(args):

    joint_transform_list = [
        RandomImgAugment(True,
                        True,
                        Image.BICUBIC)]

    joint_transform = Compose(joint_transform_list)

    dataset = get_dataset(root=args.root, data_file=args.test_data_file, phase='test',
                        dataset=args.dataset, joint_transform=joint_transform)
    loader = DataLoader(
        dataset,
        batch_size=1, shuffle=False,
        num_workers=int(args.nThreads),
        pin_memory=True
    )

    return loader

def create_train_dataloader(args):
    joint_transform_list = [
        RandomImgAugment(args.no_flip,
                        args.no_augment,
                        Image.BICUBIC)]
    joint_transform = Compose(joint_transform_list)

    dataset = get_dataset(root=args.root, data_file=args.train_data_file, phase='train',
                        dataset=args.dataset, joint_transform=joint_transform)

    loader = DataLoader(
        dataset,
        batch_size=args.batchSize, shuffle=True,
        num_workers=int(args.nThreads),
        pin_memory=True
    )

    return loader

def create_dataloader(args):

    if not args.isTrain:
        return create_test_dataloader(args)

    return create_train_dataloader(args)

def create_h5_train_dataloader(args):
    train_use_mods: Tuple[int, int] = None

    if args.block_size > 1:
        train_use_mods = (0, args.block_size - 1)

    tr_err_range = args.tr_err_range if args.prj_pos_aug is True else 0.0
    rot_err_range = args.rot_err_range if args.prj_pos_aug is True else 0.0

    dataset = H5_Train_Dataset(
        h5_paths=args.train_data, config=args.train_dl_config, quiet=True,
        block_size=args.block_size, use_mods=train_use_mods,
        tr_err_range=tr_err_range, rot_err_range=rot_err_range
    )

    return DataLoader(dataset, batch_size=args.batchSize, shuffle=True, pin_memory=True)

def create_h5_test_dataloader(args):
    test_use_mods: Tuple[int, int] = None

    if args.block_size > 1:
        test_use_mods = (args.block_size - 1, args.block_size)

    dataset = H5_Test_Dataset(
        h5_paths=args.train_data, config=args.train_dl_config, quiet=True,
        block_size=args.block_size, use_mods=test_use_mods,
    )

    return DataLoader(dataset, batch_size=args.batchSize, shuffle=False, pin_memory=True)

def create_h5_dataloader(args):
    if not args.isTrain:
        return create_h5_train_dataloader(args)
    else:
        return create_h5_test_dataloader(args)
