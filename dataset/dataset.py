""" module dataset.dataset.py
"""
import cv2
import torch
import random

import numpy as np
import dataset.transform as transform

from typing import List
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from .classes import get_split_classes_pascal
from .utils import make_dataset, process_label


# region CONSTANTS
VALID_TRAIN_SPLIT = [0, 1, 2, 3]
VALID_VAL_SPLIT = VALID_TRAIN_SPLIT + [-1, 'default']
# endregion CONSTANTS


# region Dataset
class StandardData(Dataset):
    pass


class EpisodicData(Dataset):
    def __init__(self, transform: transform.Compose, class_list: List[int], data_list_path: str, shot: int = 1, data_root: str = "dataset/VOC2012"):
        self.shot = shot
        self.transform = transform
        self.data_root = data_root
        self.class_list = class_list
        self.data_file_list, self.class_file_dict = make_dataset(
            self.data_root, data_list_path, self.class_list)

    def __len__(self):
        return len(self.data_file_list)

    def __getitem__(self, index: int):

        # * ===== Read query image + choose a class =====
        query_image_path, query_label_path = self.data_file_list[index]
        query_image = np.float32(cv2.imread(
            query_image_path, cv2.IMREAD_COLOR))
        query_label = cv2.imread(query_label_path, cv2.IMREAD_GRAYSCALE)

        assert query_image.shape[0] == query_label.shape[0] and query_image.shape[1] == query_label.shape[1],\
            f"Query image and label shapes mismatch: image shape {query_image.shape}; label shape {query_label.shape}"

        new_label_classes = process_label(
            query_label, class_list=self.class_list, class_ids_to_be_removed=[0, 255])
        assert len(
            new_label_classes) > 0, f"Got an empty label class list at index {index}"

        # * ===== Randomly draw one class from query label =====
        class_chosen = np.random.choice(new_label_classes)
        # * update the query label based on choice
        ignore_pixel = np.where(query_label == 255)
        target_pixel = np.where(query_label == class_chosen)
        query_label[ignore_pixel] = 255
        query_label[target_pixel] = 1

        # * retrieve the file info
        file_class_chosen = self.class_file_dict[class_chosen]
        num_file = len(file_class_chosen)

        # * ===== Build Support =====

        # * First randomly choose indexes of support images
        support_image_path_list = []
        support_label_path_list = []
        support_idx_list = []

        for _ in range(self.shot):
            support_idx = random.randint(0, num_file - 1)
            # * init the support image, label path
            support_image_path, support_label_path = file_class_chosen[support_idx]

            # * make sure that the support image we drew does not overlap with query image and previously drawn support images
            while ((support_image_path == query_image_path and support_label_path == query_label_path) or support_idx in support_idx_list):
                support_idx = random.randint(0, num_file - 1)
                # * init the support image, label path
                support_image_path, support_label_path = file_class_chosen[support_idx]

            support_idx_list.append(support_idx)
            support_image_path_list.append(support_image_path)
            support_label_path_list.append(support_label_path)

        # * Second, read support images and labels
        support_image_list = []
        support_label_list = []

        for k in range(self.shot):
            support_image = np.float32(cv2.imread(
                support_image_path_list[k], cv2.IMREAD_COLOR))
            support_label = cv2.imread(
                support_label_path_list[k], cv2.IMREAD_GRAYSCALE)

            # * process support label and only retain the chosen class
            ignore_pixel = np.where(support_label == 255)
            target_pixel = np.where(support_label == class_chosen)
            support_label[:, :] = 0
            support_label[ignore_pixel] = 255
            support_label[target_pixel] = 1

            assert support_image.shape[0] == support_label.shape[0] and support_image.shape[1] == support_label.shape[1],\
                f"Support image and label shapes mismatch: image shape {support_image.shape}; label shape {support_label.shape}"

            support_image_list.append(support_image)
            support_label_list.append(support_label)

        assert len(support_label_list) == self.shot and len(
            support_image_list) == self.shot, f"Support image/label retrieve failed and we could not retrieve enough support labels/images"

        # * concatenate support images and labels
        support_images = []
        support_labels = []
        for k in range(self.shot):
            support_images.append(torch.Tensor(support_image_list[k]))
            support_labels.append(torch.Tensor(support_label_list[k]))
        spprt_images_tensor = torch.cat(support_images, dim=0)
        spprt_labels_tensor = torch.cat(support_labels, dim=0)

        # * transform query image and label into torch.Tensor
        qry_img_tensor = torch.Tensor(query_image)
        qry_label_tensor = torch.Tensor(query_label)

        return qry_img_tensor, qry_label_tensor, spprt_images_tensor, spprt_labels_tensor,\
            [query_image, query_label, query_image_path, query_label_path],\
            [support_image_list, support_label_list,
                support_image_path_list, support_label_path_list]
# endregion Dataset


# region DataLoader
def get_pascal_raw_train_dataloader(split: int, episodic: bool = True, train_list: str = "dataset/lists/pascal/train.txt") -> DataLoader:
    """ Get Dataloader for PASCAL-5i dataset 

    The Dataloader here loads raw images with no transformation or augmentation

    Args:
        split {int}: dataset split
        episodic {bool}: whether or not to return the Dataloader in an episodic manner (currently, non-episodic Dataloader is NOT implemented)
        train_list {str}: the relative/absolute path to the train list .txt file

    Returns:
        {torch.utils.data.Dataloader}: a Dataloader that loads raw PASCAL-5i data in the following manner
            (qry_img_tensor, qry_label_tensor, spprt_images_tensor, spprt_labels_tensor, qry_info, spt_info)
    """
    assert split in VALID_TRAIN_SPLIT, f"Valid train splits: {VALID_TRAIN_SPLIT}, got split: {split}"

    split_classes_pascal = get_split_classes_pascal()
    class_list = split_classes_pascal["pascal"][split]['train']

    if episodic:
        raw_train_data_pascal = EpisodicData(
            transform=None,
            class_list=class_list,
            data_list_path=train_list,
        )
    else:
        raw_train_data_pascal = StandardData(
            transform=None,
            class_list=class_list,
            data_list_path=train_list
        )

    train_loader = DataLoader(
        raw_train_data_pascal,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        sampler=None,
        drop_last=True,
    )

    return train_loader

# endregion DataLoader
