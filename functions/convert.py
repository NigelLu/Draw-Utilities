"""module convert.py
"""
import cv2
import torch
import numpy as np

from einops import rearrange


def detach(draw_info_dict: dict, keys_to_process: list) -> None:
    """Detach tensors in draw_info_dict

    Detach the tensors (in place) in draw_info_dict based on the keys given by keys_to_process

    Args:
        draw_info_dict {dict}: a python dictionary with values being torch.Tensor
        keys_to_process {list}: a python list with values being (typically) strings or integers, 
                                specifying the keys in draw_info_dict to be processed

    Returns:
        None
    """
    # * double check for argument type
    assert type(
        draw_info_dict) == dict, f"Expect draw_info_dict argument to be a dict, got {type(draw_info_dict)} instead"
    assert type(
        keys_to_process) == list, f"Expect keys_to_process argument to be a list, got {type(keys_to_process)} instead"

    # * start detaching
    print(f">>> Start detaching for {keys_to_process}...")
    try:
        for key in keys_to_process:
            draw_info_dict[key] = draw_info_dict[key].detach()
        print(">>> Detaching finished")
    except:
        print(
            "Something went wrong and we failed to detach the corresponding draw info dict")
        print(">>> detach function exited with status code 1")
        return


def to_float32(draw_info_dict: dict, keys_to_process: list, in_place: bool = True) -> dict:
    """Convert tensors to torch.float32 in draw_info_dict

    Convert the tensors to torch.float32 in draw_info_dict based on the keys given by keys_to_process

    Args:
        draw_info_dict {dict}: a python dictionary with values being torch.Tensor
        keys_to_process {list}: a python list with values being (typically) strings or integers, 
                                specifying the keys in draw_info_dict to be processed
        in_place {bool}: True/False argument, specifying whether to convert the tensors in an in-place manner

    Returns:
        A dict with the keys in keys_to_process, with values being
            None (if in_place is True)
                OR
            torch.Tensor (if in_place is False), these tensors of type torch.float32 and are copies of tensors in draw_info_dict
    """
    # * double check for argument type
    assert type(
        in_place) == bool, f"Expect keys_to_process argument to be a bool, got {type(in_place)} instead"
    assert type(
        draw_info_dict) == dict, f"Expect draw_info_dict argument to be a dict, got {type(draw_info_dict)} instead"
    assert type(
        keys_to_process) == list, f"Expect keys_to_process argument to be a list, got {type(keys_to_process)} instead"

    # * start converting to float 32
    print(f">>> Start converting {keys_to_process} to float32...")
    try:
        # * create a new info dict for not in_place conversion
        new_draw_info_dict = dict([(key, None)
                                  for key in keys_to_process])

        for key in keys_to_process:
            if not in_place:
                new_draw_info_dict[key] = new_draw_info_dict[key].clone().type(
                    torch.float32)
            if in_place:
                draw_info_dict[key] = draw_info_dict[key].type(torch.float32)

        print(">>> Converting to float32 finished")
        return new_draw_info_dict
    except:
        print(
            "Something went wrong and we failed to convert the corresponding draw info dict")
        print(">>> to_float32 function exited with status code 1")
        return


def tocpu_and_asNumpy(draw_info_dict: dict, keys_to_process: list, in_place: bool = True) -> dict:
    """Send tensors to CPU and convert them from torch.Tensor to Numpy

    Send the tensors in draw_info_dict to CPU and convert them from torch.Tensor to Numpy based on the keys given by keys_to_process

    Args:
        draw_info_dict {dict}: a python dictionary with values being torch.Tensor
        keys_to_process {list}: a python list with values being (typically) strings or integers, 
                                specifying the keys in draw_info_dict to be processed
        in_place {bool}: True/False argument, specifying whether to convert the tensors in an in-place manner

    Returns:
        A dict with the keys in keys_to_process, with values being
            None (if in_place is True)
                OR
            torch.Tensor (if in_place is False), these tensors are copies of tensors in draw_info_dict,
                which got sent to CPU and converted to Numpy
    """
    # * double check for argument type
    assert type(
        in_place) == bool, f"Expect keys_to_process argument to be a bool, got {type(in_place)} instead"
    assert type(
        draw_info_dict) == dict, f"Expect draw_info_dict argument to be a dict, got {type(draw_info_dict)} instead"
    assert type(
        keys_to_process) == list, f"Expect keys_to_process argument to be a list, got {type(keys_to_process)} instead"

    # * start to CPU and convert to numpy
    print(
        f">>> Start sending {keys_to_process} to CPU and transform to Numpy...")
    try:
        # * create a new info dict for not in_place conversion
        new_draw_info_dict = dict([(key, None)
                                  for key in keys_to_process])

        for key in keys_to_process:
            if not in_place:
                new_draw_info_dict[key] = new_draw_info_dict[key].clone(
                ).cpu().numpy()
            if in_place:
                draw_info_dict[key] = draw_info_dict[key].cpu().numpy()

        print(">>> Sending to CPU and transform to Numpy finished")
        return new_draw_info_dict
    except:
        print(
            "Something went wrong and we failed to convert the corresponding draw info dict")
        print(">>> tocpu_and_asNumpy function exited with status code 1")
        return


def apply_mask(draw_info_dict: dict, image_to_mask_map: dict, MASK_WEIGHT: float, GAMMA_SCALAR: float, rearrange_str: str = None) -> dict:
    """Apply a red mask on top of the images

    Apply a red mask on top of the images, with the keys of masks and images specified by an image_to_mask_map

    Args:
        draw_info_dict {dict}: a python dictionary with values being torch.Tensor
        image_to_mask_map {dict}: a python dictionary specifying an image to mask key mapping (corresponding to the keys in draw_info_dict), 
                                  as following
                                    {
                                        <image_key>: <mask_key>,
                                    }
        MASK_WEIGHT {float}: a float in the interval (0, 1) specifying the weight of the mask (image weight will be calculated using 1-MASK_WEIGHT)
        GAMMA_SCALAR {float}: the scalar to apply on the masked image. This is the same argument used as in cv2.addWeighted()
        rearrange_str {str}: a string in the form of "c h w -> h w c", for rearranging the image tensor using einops.rearrange function

    Returns:
        A dict with the sames keys as in image_mask_map, with values being
            Numpy.Array, these are of shape (h, w, 3) and stores the masked image
    """
    # * double check for argument type
    assert type(
        rearrange_str) == str, f"Expect rearrange_str argument to be a str, got {type(rearrange_str)} instead"
    assert type(
        draw_info_dict) == dict, f"Expect draw_info_dict argument to be a dict, got {type(draw_info_dict)} instead"
    assert type(
        image_to_mask_map) == dict, f"Expect image_to_mask_map argument to be a bool, got {type(image_to_mask_map)} instead"

    # * start applying masks
    print(f">>> Start applying masks for {image_to_mask_map}...")
    try:
        # * create a dict for storing masks, with keys being the keys of image_to_mask_map
        mask_dict = dict([(key, None) for key in image_to_mask_map.keys()])

        for image_key, mask_key in image_to_mask_map.items():
            image = np.ascontiguousarray(
                np.array(draw_info_dict[image_key], dtype=np.uint8, copy=True))
            if rearrange_str:
                image = rearrange(image, rearrange_str)

            mask = np.array(
                draw_info_dict[mask_key], dtype=np.uint8, copy=True) * 255
            G = B = np.zeros(mask.shape)
            RGB_mask = np.concatenate((mask, G, B), axis=2)

            image_with_mask = cv2.addWeighted(
                image, 1-MASK_WEIGHT, RGB_mask, MASK_WEIGHT, GAMMA_SCALAR)

            mask_dict[image_key] = image_with_mask

        print(">>> Mask Applied")
        return mask_dict
    except:
        print(
            "Something went wrong and we failed to apply masks")
        print(">>> apply_mask function exited with status code 1")
        return
