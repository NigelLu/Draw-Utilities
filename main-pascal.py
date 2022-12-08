""" Main Drawing Module main-pascal.py
    This module loads data from PASCAL-VOC2012    
"""
import time

from tqdm import tqdm
from draw.drawlabel import draw_label
from config import config_pascal as cfg
from draw.drawmask import applymask_and_draw
from functions.utils import validate_save_dir
from dataset.dataset import get_pascal_raw_train_dataloader
from functions.convert import detach, reshape, to_float32, tocpu_and_asNumpy


def main():
    keys_to_process = ['query', 'support', 'label_q', 'label_s']
    image_to_mask_map = {'query': 'label_q', 'support': 'label_s'}
    should_proceed = validate_save_dir(
        cfg.SAVE_DIR, image_to_mask_map, cfg.SPLIT)
    if not should_proceed:
        print(">>> Main process aborted")
        return
    print(">>> Building PASCAL train loader...")
    pascal_raw_train_dataloder = iter(
        get_pascal_raw_train_dataloader(cfg.SPLIT, episodic=True))
    print(">>> PASCAL train loader build success")

    for idx in tqdm(range(cfg.NUM_TO_DRAW)):
        qry_img_tensor, qry_label_tensor, spprt_images_tensor, spprt_labels_tensor, qry_info, spt_info = \
            pascal_raw_train_dataloder.next()
        draw_info_dict = {
            'query': qry_info[0],
            'support': spt_info[0][0],
            'label_q': qry_info[1],
            'label_s': spt_info[1][0],
        }
        detach(draw_info_dict, keys_to_process, should_log=cfg.SHOULD_LOG)
        to_float32(draw_info_dict, keys_to_process, should_log=cfg.SHOULD_LOG)
        reshape(draw_info_dict, keys_to_process[0:2],
                'b h w c -> (b h) w c', should_log=cfg.SHOULD_LOG)
        reshape(draw_info_dict,
                keys_to_process[2:], 'c h w -> h w c', should_log=cfg.SHOULD_LOG)
        tocpu_and_asNumpy(draw_info_dict, keys_to_process,
                          should_log=cfg.SHOULD_LOG)
        applymask_and_draw(draw_info_dict, image_to_mask_map,
                           cfg.MASK_WEIGHT, cfg.GAMMA_SCALAR, cfg.SAVE_DIR, f'{idx}.jpg', cfg.SPLIT, should_log=cfg.SHOULD_LOG)
        draw_label(draw_info_dict, image_to_mask_map, cfg.SAVE_DIR,
                   f'{idx}.jpg', cfg.SPLIT, should_log=cfg.SHOULD_LOG)

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
