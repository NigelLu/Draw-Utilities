"""module dataset.utils.py
"""
import os
import cv2

import numpy as np

from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from collections import defaultdict
from typing import Dict, List, Tuple, Callable, TypeVar, Iterable


# region CONSTANTS
A = TypeVar("A")
B = TypeVar("B")
# endregion CONSTANTS

# region Image Info Preparation


def mmap_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    """ Apply function to an Iterable utilizing multiprocessing

        Args:
            fn {Callable[[A], B]}: a callable function that intakes an argument of type A, and outputs a result of type B
            iter {Iterable[A]}: an Iterable of type A

        Returns:
            {List[B]}: returns a list of mapping results of type B
    """
    return Pool().map(fn, iter)


def process_label(label: np.array, class_list: List[int], class_ids_to_be_removed: List[int], threshold: int = None, supported_classes: List[int] = list(range(1, 81))) -> List[int]:
    """ Preliminary processing for label
        
        Remove the unwanted classes as well as retain the classes that covers an area no smaller than the optional threshold

        Args:
            label {np.array}: the image label, should be of shape [h, w, 1]
            class_list {[List[int]]}: a list of the class ids that we should consider as foreground
            class_ids_to_be_removed {List[int]}: a list of class ids that we should remove from the label
            threshold {int}: optional threshold argument for only retaining classes that cover a large enough area, default to no threshold
            supported_classes {List[int]}: a list of all class ids that we should consider as valid

        Returns:
            {List[int]}: a list of the class ids to be retained in the label
    """
    # * a list of the unique class_ids for the current image
    label_class_list = np.unique(label).tolist()
    for class_id in class_ids_to_be_removed:
        if class_id in label_class_list:
            label_class_list.remove(class_id)

    for label_class in label_class_list:
        assert label_class in supported_classes, f"Supported classes: {supported_classes}\nInvalid class found: {label_class}"

    class_id: int
    new_label_class_list = []
    for class_id in label_class_list:
        if class_id not in class_list:
            continue

        if not threshold:
            new_label_class_list.append(class_id)
            continue

        tmp_label = np.zeros_like(label)
        tmp_label[np.where(label == class_id)] = 1

        if tmp_label.sum() >= threshold:
            new_label_class_list.append(class_id)

    return new_label_class_list


def process_image_line(line: str, data_root: str, class_list: List[int]) -> Tuple[List[Tuple[str, str]], Dict[int, List[Tuple[str, str]]]]:
    ''' Reads and parses a line corresponding to 1 file

        Args:
            line {str} : A line corresponding to 1 file, in the format path_to_image.jpg path_to_image.png
            data_root {str}: Path to the data directory
            class_list {List[int]}: List of classes to keep

        Returns:
            Tuple:
                data_file_list {List[Tuple[str, str]]}: 
                    list containing one element -- Tuple[<image>, <label>]

                class_file_dict {Dict[int, List[Tuple[str, str]]]}: 
                    dict of <cls_id> -> List[Tuple[<image>, <label>]]
    '''
    # * line is in the form of "<image_file> <label_file>"
    line = line.strip()
    line_split = line.split(' ')
    # * image_file_path
    image_path = os.path.join(data_root, line_split[0])
    # * label_file_path
    label_path = os.path.join(data_root, line_split[1])

    item: Tuple[str, str] = (image_path, label_path)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    # * ===== Determine the classes that occupy an area large enough on the image to be considered valid for the image =====
    # * first-step processing
    new_label_class_list = process_label(label, class_list, class_ids_to_be_removed=[
                                         0, 255], threshold=2 * 32 * 32)

    data_file_list: List[Tuple[str, str]] = []
    class_file_dict: Dict[int, List[Tuple[str, str]]] = defaultdict(list)

    if len(new_label_class_list) > 0:
        data_file_list.append(item)

        for class_id in new_label_class_list:
            assert class_id in class_list
            class_file_dict[class_id].append(item)

    return data_file_list, class_file_dict


def make_dataset(data_root: str, data_list: str, class_list: List[int]) -> Tuple[List[Tuple[str, str]], Dict[int, List[Tuple[str, str]]]]:
    ''' Recovers all tupples (img_path, label_path) relevant to the current experiments (class_list is used as filter)

        Args:
            data_root {str}: Path to the data directory
            data_list {str}: Path to the .txt file that contain the train/test split of images
            class_list {List[int]}: List of classes to keep

        Returns:
            Tuple:
                data_file_list {List[Tuple[str, str]]}: 
                    List of (img_path, label_path) that contain at least 1 object of a class in class_list

                class_file_dict {Dict[int, List[Tuple[str, str]]]}: 
                    Dict of all (img_path, label_path that contain at least 1 object of a class in class_list, grouped by classes.
    '''
    assert os.path.isfile(data_list), \
        f"Argument data_list expects a valid file path, {data_list} is invalid"

    data_file_list: List[Tuple[str, str]] = []
    list_read = open(data_list).readlines()

    print(f"Processing data for {class_list}")
    class_file_dict: Dict[int, List[Tuple[str, str]]] = defaultdict(list)

    process_partial = partial(
        process_image_line, data_root=data_root, class_list=class_list)

    for sub_data_file_list, sub_class_file_dict in mmap_(process_partial, tqdm(list_read)):
        data_file_list += sub_data_file_list

        for (class_id, items) in sub_class_file_dict.items():
            class_file_dict[class_id] += items

    return data_file_list, class_file_dict
# endregion Image Info Preparation
