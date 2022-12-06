"""module mask.py
"""
import cv2
import numpy as np


def apply_mask(draw_info_dict: dict, image_to_mask_map: dict, MASK_WEIGHT: float, GAMMA_SCALAR: float) -> dict:
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

    Returns:
        A dict with the sames keys as in image_mask_map, with values being
            numpy.Array, these are of shape (h, w, 3) and stores the masked image
    """
    # * double check for argument type
    assert type(
        MASK_WEIGHT) == float, f"Expect MASK_WEIGHT argument to be a float, got {type(MASK_WEIGHT)} instead"
    assert type(
        GAMMA_SCALAR) == float, f"Expect GAMMA_SCALAR argument to be a float, got {type(GAMMA_SCALAR)} instead"
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
