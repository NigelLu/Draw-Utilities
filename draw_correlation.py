
import os
import cv2
import pdb
import math
import torch
import random
import shutil
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from einops import rearrange

# ! IMPORTANT: please only use batch size of 1,
# !              otherwise the program will not run as expected

# region CONSTANTS
# * used to judge which point on support image to choose
TOP_K_CORRELATION_QUERY = 3600
PICKED_LOCATION = [(250, 170)]
SUPPORT_POINT_MULTIPLICATION_FACTOR = (473/60)**2

# * denormalize
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# * draw related
PT_FILE_DIR = '/scratch/xl3139/PROTR'
SPLIT = 0
SAVE_DIR = '/scratch/xl3139/PROTR/correlation-picked-new'
NUM_TO_DRAW = 100
START_IDX = 0
CIRCLE_COLOR = (0, 255, 0)
CIRCLE_RADIUS = 15
CIRCLE_THICKNESS = 2
SUPPORT_WEIGHT = 0.6
HEATMAP_WEIGHT = 1 - SUPPORT_WEIGHT
GAMMA_SCALAR = 1
HANDPICKED_LOCATION = None
GROUND_TRUTH_WEIGHT = 0.7
# endregion CONSTANTS

# region functions

random.seed(1)

def interpolate(feats, size, mode='bilinear', align_corners=True):
    return F.interpolate(feats, size=size, mode=mode, align_corners=align_corners)


def to_float32(draw_info):
    # * process for query and support
    for key in ['query', 'support', 'label_s', 'label_q']:
        draw_info[key] = draw_info[key].type(torch.float32)
    # * process for attn_dict
    draw_info['attn_dict'][0] = draw_info['attn_dict'][0].type(torch.float32)
    draw_info['attn_dict'][1] = draw_info['attn_dict'][1].type(torch.float32)


def interpolate_and_mask(draw_info):

    # * process for attn_dict
    attn_info_before = draw_info['attn_dict'][1]
    attn_info_after = draw_info['attn_dict'][0]
    b, n_q, n_s = attn_info_after.shape
    h_s = w_s = int(math.sqrt(n_s))

    _, topk_correlation_idx_s_before = torch.topk(
        attn_info_before, TOP_K_CORRELATION_QUERY, dim=2)
    _, topk_correlation_idx_s_after = torch.topk(
        attn_info_after, TOP_K_CORRELATION_QUERY, dim=2)

    # * shape [num_points_available, 2]
    foreground_points_q = (draw_info['label_q'][0] == 1).nonzero()
    # * handpicked location
    if HANDPICKED_LOCATION:
        selected_point_q_x, selected_point_q_y = HANDPICKED_LOCATION
    else:
        # * randomely select from available foreground points
        selected_point_q_y, selected_point_q_x = foreground_points_q[random.randint(
            0, foreground_points_q.shape[0]-1)]
        selected_point_q_x, selected_point_q_y = int(
            selected_point_q_x), int(selected_point_q_y)

    picked_correlation_idx_q = int(
        (selected_point_q_x*473 + selected_point_q_y - 1)/SUPPORT_POINT_MULTIPLICATION_FACTOR)
    # * shape [1, h_q * w_q]
    # * get correlation maps for support
    topk_correlation_map_before = attn_info_before[0,
                                                   picked_correlation_idx_q, :].unsqueeze(0)
    topk_correlation_map_after = attn_info_after[0,
                                                 picked_correlation_idx_q, :].unsqueeze(0)

    # * normalize correlation
    max_correlation_before, _ = torch.max(topk_correlation_map_before, dim=1)
    min_correlation_before, _ = torch.min(topk_correlation_map_before, dim=1)
    max_correlation_after, _ = torch.max(topk_correlation_map_after, dim=1)
    min_correlation_after, _ = torch.min(topk_correlation_map_after, dim=1)

    topk_correlation_map_before = ((topk_correlation_map_before-min_correlation_before)/(
        max_correlation_before-min_correlation_before)).view(1, h_s, w_s)
    topk_correlation_map_after = ((topk_correlation_map_after-min_correlation_after)/(
        max_correlation_after-min_correlation_after)).view(1, h_s, w_s)

    # * generate masks for topk correlation points on support
    topk_correlation_mask_before = torch.zeros((1, n_s))
    topk_correlation_mask_after = torch.zeros((1, n_s))

    topk_correlation_mask_before[:,
                                 topk_correlation_idx_s_before[0, picked_correlation_idx_q, :]] = 1
    topk_correlation_mask_after[:, topk_correlation_idx_s_after[0,
                                                                picked_correlation_idx_q, :]] = 1

    topk_correlation_mask_before = topk_correlation_mask_before.view(
        1, h_s, w_s)
    topk_correlation_mask_after = topk_correlation_mask_after.view(1, h_s, w_s)

    # * transfer everything to GPU and mask the correlation map
    topk_correlation_mask_before, topk_correlation_mask_after, topk_correlation_map_before, topk_correlation_map_after = \
        topk_correlation_mask_before.to('cuda'), topk_correlation_mask_after.to(
            'cuda'), topk_correlation_map_before.to('cuda'), topk_correlation_map_after.to('cuda')

    # * apply masks
    topk_correlation_map_before = torch.mul(
        topk_correlation_map_before, topk_correlation_mask_before)
    topk_correlation_map_after = torch.mul(
        topk_correlation_map_after, topk_correlation_mask_after)

    # * interpolate correlation map to image size
    # * shape [1, 473, 473]
    topk_correlation_map_before = interpolate(
        topk_correlation_map_before.unsqueeze(1), size=(473, 473)).squeeze(1)
    topk_correlation_map_after = interpolate(
        topk_correlation_map_after.unsqueeze(1), size=(473, 473)).squeeze(1)

    draw_info['selected_point_q'], draw_info['topk_correlation_map_before'], draw_info['topk_correlation_map_after'] = (
        selected_point_q_x, selected_point_q_y), topk_correlation_map_before, topk_correlation_map_after


def reshape_and_denormalize(draw_info):
    for key in ['query', 'support']:
        img = draw_info[key][0, 0] if key == 'support' else draw_info[key][0]
        for t, m, s in zip(img, MEAN, STD):
            t.mul_(s).add_(m).mul_(255)

        draw_info[key] = torch.cat(
            [img[2].unsqueeze(0), img[1].unsqueeze(0), img[0].unsqueeze(0)], dim=0)


def detach_and_asnumpy(draw_info):
    # * process for query and support
    for key in ['query', 'support', 'label_s', 'label_q', 'topk_correlation_map_before', 'topk_correlation_map_after']:
        if (type(draw_info[key]) == np.ndarray):
            continue
        draw_info[key] = draw_info[key].detach().cpu().numpy()
    # * process for attn_dict
    draw_info['attn_dict'][0] = draw_info['attn_dict'][0].detach().cpu().numpy()

# * support, query image [b, 1, 3, 473, 473]


def apply_colormap_and_save(draw_info, save_dir=None, idx=None):
    s_img, q_img, correlation_map_before, correlation_map_after, label_s = draw_info['support'], draw_info['query'], \
        np.uint8(draw_info['topk_correlation_map_before'][0] *
                 255), np.uint8(draw_info['topk_correlation_map_after'][0]*255), np.uint8(draw_info['label_s'][0, 0]*255),
    s_img, q_img = \
        np.ascontiguousarray(rearrange(s_img, 'c h w -> h w c'), dtype=np.uint8), \
        np.ascontiguousarray(
            rearrange(q_img, 'c h w -> h w c'), dtype=np.uint8)

    # * circle on query image
    q_img = cv2.circle(
        q_img, draw_info['selected_point_q'], CIRCLE_RADIUS, CIRCLE_COLOR, CIRCLE_THICKNESS)

    # * processing for support
    s_correlation_heatmap_before = cv2.applyColorMap(
        correlation_map_before, cv2.COLORMAP_JET)
    s_correlation_heatmap_after = cv2.applyColorMap(
        correlation_map_after, cv2.COLORMAP_JET)
    s_ground_truth = cv2.applyColorMap(label_s, cv2.COLORMAP_JET)
    s_img_with_heatmap_before = cv2.addWeighted(
        s_img, SUPPORT_WEIGHT, s_correlation_heatmap_before, HEATMAP_WEIGHT, GAMMA_SCALAR)
    s_img_with_heatmap_after = cv2.addWeighted(
        s_img, SUPPORT_WEIGHT, s_correlation_heatmap_after, HEATMAP_WEIGHT, GAMMA_SCALAR)
    s_img_with_mask = cv2.addWeighted(
        s_img, GROUND_TRUTH_WEIGHT, s_ground_truth, 1-GROUND_TRUTH_WEIGHT, GAMMA_SCALAR)

    if not idx:
        idx = ''

    cv2.imwrite(os.path.join(save_dir, f'split-{SPLIT}', f'{idx}.jpg'), np.concatenate(
        (q_img, s_img_with_mask, s_img_with_heatmap_before, s_correlation_heatmap_before, s_img_with_heatmap_after, s_correlation_heatmap_after), axis=1))
# endregion functions


def main():
    # * clear and prepare save dir
    assert os.path.isdir(
        SAVE_DIR), f'SAVE_DIR expects a folder path, {SAVE_DIR} is not a folder'

    for save_path in [os.path.join(SAVE_DIR, f'split-{SPLIT}')]:
        if not os.path.isdir(save_path):
            print(f"Creating directory {save_path}...")
            os.mkdir(save_path)
            continue

        if len(os.listdir(save_path)) <= 0:
            continue

        print(
            f"Warning: the save path '{save_path}' you specified is NOT empty\nIt contains")

        dir_content = os.listdir(save_path)
        dir_content.sort()
        print('\n'.join(dir_content))
        should_proceed = ''
        while should_proceed.lower() != 'yes' and should_proceed.lower() != 'no' and should_proceed.lower() != 'ignore':
            should_proceed = input(
                "Would you like us to empty the directory for you and proceed? (yes/no/ignore)\n> ").lower()
            if should_proceed.lower() == 'no':
                return
            if should_proceed.lower() == 'yes':
                shutil.rmtree(save_path, ignore_errors=True)
                os.mkdir(save_path)

    print(">>> Loading .pt file...")
    draw_info_list = torch.load(os.path.join(
        PT_FILE_DIR, f"drawList-{SPLIT}-temp10-200.pt"))
    print(">>> .pt file loaded...\n")

    print(">>> Example info keys:", draw_info_list[0].keys(), sep="\n")

    if START_IDX + NUM_TO_DRAW > len(draw_info_list):
        print(
            f"WARNING: START_IDX ({START_IDX}) + NUM_TO_DRAW ({NUM_TO_DRAW}) specified is larger than the length of draw info list {len(draw_info_list)}")

    print('>>> Start converting float16 to float32...')
    for draw_info_idx in tqdm(range(START_IDX, START_IDX+NUM_TO_DRAW)):
        to_float32(draw_info_list[draw_info_idx])
    print('>>> Conversion complete...\n')

    print('>>> Start interpolating and masking...')
    for draw_info_idx in tqdm(range(START_IDX, START_IDX+NUM_TO_DRAW)):
        interpolate_and_mask(draw_info_list[draw_info_idx])
    print('>>> Interpolation and masks added to draw info list...\n')

    print('>>> Start reshaping and denormalizing...')
    for draw_info_idx in tqdm(range(START_IDX, START_IDX+NUM_TO_DRAW)):
        reshape_and_denormalize(draw_info_list[draw_info_idx])
    print('>>> Reshaping and denormalizing complete...\n')

    print('>>> Start detaching and transforming draw info to numpy...')
    for draw_info_idx in tqdm(range(START_IDX, START_IDX+NUM_TO_DRAW)):
        detach_and_asnumpy(draw_info_list[draw_info_idx])
    print('>>> Detaching and transforming complete...\n')

    print(">>> Start applying heatmap and save...")
    for draw_info_idx in tqdm(range(START_IDX, START_IDX+NUM_TO_DRAW)):
        apply_colormap_and_save(
            draw_info_list[draw_info_idx], SAVE_DIR, idx=draw_info_idx+1)
    print(">>> Applying heatmap and save complete...\n")


if __name__ == "__main__":
    main()
