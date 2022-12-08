"""module draw.drawlabel.py

   Summary:
    This module reduces label to "background" versus "foreground" grayscale
    and draws the grayscale mask alongside the original image
"""
import os
import cv2
import torch

import numpy as np

from typing import Dict


def draw_label(draw_info_dict: Dict[str, torch.Tensor],  image_to_mask_map: Dict[str, str], save_dir: str, image_name: str, split: int, should_log: bool = False) -> None:
    """ Save the label as black-white mask image alongside the image

    Save the label as black-white mask image alongside the image,
        the keys of the masks and images are specified by an image_to_mask_map
        the save path and image name are specified by save_dir and image_name

    Args:
        draw_info_dict {dict}: a python dictionary with values being torch.Tensor
        image_to_mask_map {dict}: a python dictionary specifying an image to mask key mapping (corresponding to the keys in draw_info_dict), 
                                  as following
                                    {
                                        <image_key>: <mask_key>,
                                    }
        save_dir {str}: the save dir to save the output images
        image_name {str}: the image name to name the output images
        split {int}: the split of the dataset

    Returns:
        None
    """

    # * double check for image_name and save_dir
    assert type(
        split) == int, f"Expect split argument to be an int, got {type(split)} instead"
    assert type(
        image_name) == str, f"Expect image_name argument to be a str, got {type(image_name)} instead"

    split = str(split)

    try:
        if should_log:
            print(f">>> Start drawing labels for {image_to_mask_map}...")
        for image_key, mask_key in image_to_mask_map.items():
            image = draw_info_dict[image_key]
            label = draw_info_dict[mask_key]
            # * remove ignored pixels (i.e., treat as background)
            label[label == 255] = 0
            # * put foreground pixels as white
            label[label == 1] = 255
            # * remove all other classes
            unique_labels = np.unique(label).tolist()
            unique_labels.remove(0)
            unique_labels.remove(255)
            for irrelevant_class_ids in unique_labels:
                label[label == irrelevant_class_ids] = 0

            cv2.imwrite(os.path.join(save_dir, split, f"{image_key}-label", image_name), np.concatenate(
                (image, cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)), axis=1))
        if should_log:
            print(f">>> Drawing labels finished")
        return
    except:
        print(
            "Something went wrong and we failed to draw the masked images")
        print(">>> draw_label function exited with status code 1")
        return
