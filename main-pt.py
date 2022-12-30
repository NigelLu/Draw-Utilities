""" Main Drawing Module main-pt.py
    This module uses existing pt files
"""
import pdb
import time
import torch

from tqdm import tqdm
from config import config_pt as cfg
from draw.drawmask import applymask_and_draw
from functions.utils import validate_save_dir
from functions.convert import detach, reshape, to_float32, denormalize, logit_to_mask, combine, tocpu_and_asNumpy


def main():
    # * keys in draw_info_dict that will be pre-processed
    keys_to_process = ['query', 'support', 'base', 'label_q',
                       'label_s', 'label_b', 'pred_q_n', 'pred_q_n_b2', 'pred_q_b']

    # * list of the <prediction_key> that will be translated into binary masks
    masks_to_translate = ['pred_q_n', 'pred_q_n_b2', 'pred_q_b']

    # * a map for images and masks to combine side by side horizontally
    images_and_masks_to_combine = {
        "images": ["base", 'support', 'query', 'query', 'query', 'base'],
        "masks": ['label_b', 'label_s', 'label_q', 'pred_q_n', 'pred_q_n_b2', 'pred_q_b'],
    }

    # * map from <image_key> to <mask_key>
    image_to_mask_map = {'img_combined': 'mask_combined'}

    # * validate save direcotry
    should_proceed = validate_save_dir(
        cfg.SAVE_DIR, image_to_mask_map, cfg.SPLIT, cfg.DRAW_LABEL)
    if not should_proceed:
        print(">>> Main process aborted")
        return

    # * load the .pt file as draw_list
    print(f">>> Loading pt file from {cfg.PT_FILE_PATH}")

    # @note
    # * No need to load the .pt file to CUDA (drawing requires few matrix operations)
    draw_list = torch.load(cfg.PT_FILE_PATH, map_location=torch.device('cpu'))
    print(f">>> pt file loaded")

    for idx in tqdm(range(len(draw_list) if len(draw_list) < cfg.NUM_TO_DRAW else cfg.NUM_TO_DRAW)):
        # * retrieve the draw_info_dict from the draw_list
        draw_info_dict = draw_list[idx]

        # * pre-processing
        detach(draw_info_dict, keys_to_process, should_log=cfg.SHOULD_LOG)
        denormalize(
            draw_info_dict, keys_to_process[0:3], cfg.MEAN, cfg.STD, should_log=cfg.SHOULD_LOG)
        to_float32(draw_info_dict, keys_to_process, should_log=cfg.SHOULD_LOG)
        reshape(draw_info_dict, keys_to_process,
                'c h w -> h w c', should_log=cfg.SHOULD_LOG)
        logit_to_mask(draw_info_dict, masks_to_translate,
                      should_log=cfg.SHOULD_LOG)

        # * by default, "combine" function saves the result to "img_combined" and "mask_combined", respectively
        combine(draw_info_dict, images_and_masks_to_combine,
                should_log=cfg.SHOULD_LOG)
        tocpu_and_asNumpy(
            draw_info_dict, ['img_combined', 'mask_combined'], should_log=cfg.SHOULD_LOG)

        # * apply the mask to the image, then draw
        applymask_and_draw(draw_info_dict, image_to_mask_map,
                           cfg.MASK_WEIGHT, cfg.GAMMA_SCALAR, cfg.SAVE_DIR, f'{idx}.jpg', cfg.SPLIT, should_log=cfg.SHOULD_LOG)


if __name__ == "__main__":
    start_time = time.time()
    print(
        f">>> Main process started at {time.strftime('%H:%M:%S, %m/%d/%Y',time.localtime(start_time))} local time")
    main()
    end_time = time.time()
    print(
        f">>> Main process finished in {'{:.2f}'.format(end_time - start_time)} seconds")
    print(
        f">>> Main process finished at {time.strftime('%H:%M:%S, %m/%d/%Y',time.localtime(end_time))} local time")
