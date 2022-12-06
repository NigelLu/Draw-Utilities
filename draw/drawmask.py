"""module drawmask.py

   Summary:
    This module helps apply mask to image(s) and 
    draw the masked image out alongside the original image(s)
"""
import os
import cv2

import numpy as np

from functions.mask import apply_mask
from functions.utils import validate_save_dir


def applymask_and_draw(draw_info_dict: dict, image_to_mask_map: dict, MASK_WEIGHT: float, GAMMA_SCALAR: float, save_dir: str, image_name: str, split: int, should_log: bool = False) -> None:
    """Apply a red mask on top of the images and save the images

    Apply a red mask on top of the images and save the images, 
        the keys of the masks and images are specified by an image_to_mask_map
        the save path and image name are specified by save_dir and image_name

    Args:
        draw_info_dict {dict}: a python dictionary with values being torch.Tensor
        image_to_mask_map {dict}: a python dictionary specifying an image to mask key mapping (corresponding to the keys in draw_info_dict), 
                                  as following
                                    {
                                        <image_key>: <mask_key>,
                                    }
        MASK_WEIGHT {float}: a float in the interval (0, 1) specifying the weight of the mask (image weight will be calculated using 1-MASK_WEIGHT)
        GAMMA_SCALAR {float}: the scalar to apply on the masked image. This is the same argument used as in cv2.addWeighted()
        save_dir {str}: the save dir to save the output images
        image_name {str}: the image name to name the output images (note: this will be prepended with <image_key>)
        split {int}: the split of the dataset

    Returns:
        None
    """

    # * double check for image_name and save_dir
    assert type(split) == int, f"Expect split argument to be an int, got {type(split)} instead"
    assert type(
        image_name) == str, f"Expect image_name argument to be a str, got {type(image_name)} instead"

    split = str(split)

    try:
        masked_img_dict = apply_mask(
            draw_info_dict, image_to_mask_map, MASK_WEIGHT, GAMMA_SCALAR, should_log=should_log)

        for img_key, masked_img in masked_img_dict.items():
            cv2.imwrite(os.path.join(save_dir, split, img_key, image_name), np.concatenate(
                (draw_info_dict[img_key], masked_img), axis=1))

        return
    except:
        print(
            "Something went wrong and we failed to draw the masked images")
        print(">>> applymask_and_draw function exited with status code 1")
        return
