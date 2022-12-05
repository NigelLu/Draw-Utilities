"""module convert.py
"""
import cv2
import torch
import numpy as np

from einops import rearrange


def detach(draw_info_dict: dict, keys_to_process: list) -> None:
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
                                  for key in draw_info_dict.keys()])

        for key in keys_to_process:
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
                                  for key in draw_info_dict.keys()])

        for key in keys_to_process:
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
