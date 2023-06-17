import numpy as np
from scipy import ndimage
from skimage.io import imread, imshow, imsave
from matplotlib import pyplot as plt
import time
#import cv2 as cv
import pandas as pd
from skimage.metrics import mean_squared_error
import os
from nlm import denoise_nlm_with_history_smart, denoise_nlm, create_padded_img


GT_FILEPATH = "/home/xbuco/pv162-project/cropped_sample1024_512x512.png"
SAMPLES_FOLDER_PREFIX = '/home/xbuco/pv162-project/data/samples_100x100_100ns/shifted/'
RESULTS_FOLDER_PREFIX = '/home/xbuco/pv162-project/data/results/'
DF_HISTORY_NLM = './history_nlm_shifted_results.csv'
# parameters determined using IPOL paper, search_window_size should be 35
SD = 0.2632
STRENGTH = 0.35 * SD  # 0.078
COMPARE_WINDOW_SIZE = 9  # ODD
COMPARE_WINDOW_OFFSET = int((COMPARE_WINDOW_SIZE - 1) // 2)


def compare_img_quality(noisy_img, gt_img, denoised_img):
    # quality of the denoised img
    print("# PSNR")
    psnr_noisy = cv.PSNR(noisy_img, gt_img)
    print(" ", psnr_noisy)
    psnr_denoised = cv.PSNR(denoised_img, gt_img)
    print(" ", psnr_denoised)

    print("# SSIM")
    from skimage.metrics import structural_similarity as ssim
    ssim_noisy = ssim(gt_img, noisy_img, data_range=noisy_img.max() - noisy_img.min())
    print(" ", ssim_noisy)
    ssim_noisy = ssim(gt_img, denoised_img, data_range=denoised_img.max() - denoised_img.min())
    print(" ", ssim_noisy)

    print("# MSE")
    from skimage.metrics import mean_squared_error
    mse_noisy = mean_squared_error(gt_img, noisy_img)
    print(" ", mse_noisy)
    mse_denoised = mean_squared_error(gt_img, denoised_img)
    print(" ", mse_denoised)


def shift_gt_img(img_gt, shift):
    width = 100
    height = 100
    left = 150
    top = 170
    return img_gt[top + shift : top + height + shift,
                  left + shift : left + width + shift]


def generate_results_nlm_with_history(samples_folder_prefix, results_folder_prefix, df_filename):
    img0 = (imread(samples_folder_prefix + f"{0}.png", as_gray=True).astype(np.float64)) / 255
    img0 = create_padded_img(img0, COMPARE_WINDOW_OFFSET)

    n = 92
    history = []
    for c in range(1, 1 + n):
        history.append(create_padded_img(
            (imread(samples_folder_prefix + f"{c}.png", as_gray=True).astype(np.float64)) / 255, COMPARE_WINDOW_OFFSET))

    img_gt_full = imread(GT_FILEPATH, as_gray=True).astype(np.float64) / 255
    img_gt = shift_gt_img(img_gt_full, 0)

    if os.path.exists(df_filename):
        df = pd.read_csv(df_filename)
    else:
        df = pd.DataFrame(columns=['sd', 'fs', 'history', 'sw', 'cw', 'MSE', 'time'])

    for search_window_size in [19, 27, 35, 43]:
        print(f"Starting basic nlm with sw: {search_window_size}")
        start = time.time()
        denoised_img = denoise_nlm(img0, search_window_size, COMPARE_WINDOW_SIZE, STRENGTH, SD, True)
        nlm_calc_time = time.time() - start
        mse = mean_squared_error(img_gt, denoised_img)
        df = df.append({
            'sd': SD,
            'fs': STRENGTH,
            'history': 0,
            'sw': search_window_size,
            'cw': COMPARE_WINDOW_SIZE,
            'MSE': mse,
            'time': nlm_calc_time
        }, ignore_index=True)
        df.to_csv(df_filename, index=False)
        imsave(f"{results_folder_prefix}h0_sw{search_window_size}_cw{COMPARE_WINDOW_SIZE}.png",
               (denoised_img * 255).astype(np.uint8))

    for history_len in [1, 2, 3, 4, 5, 8, 10, 16, 32, 50, 64, 90]:
        print(f"Starting nlm with history of len: {history_len}")
        for search_window_size in [19, 27, 35, 43]:
            start = time.time()
            denoised_img = denoise_nlm_with_history_smart(img0, history, search_window_size, COMPARE_WINDOW_SIZE,
                                                          STRENGTH, SD, history_len, True)
            nlm_calc_time = time.time() - start
            mse = mean_squared_error(img_gt, denoised_img)
            df = df.append({
                'sd': SD,
                'fs': STRENGTH,
                'history': history_len,
                'sw': search_window_size,
                'cw': COMPARE_WINDOW_SIZE,
                'MSE': mse,
                'time': nlm_calc_time
            }, ignore_index=True)
            df.to_csv(df_filename, index=False)
            imsave(f"{results_folder_prefix}h{history_len}_sw{search_window_size}_cw{COMPARE_WINDOW_SIZE}.png",
                   (denoised_img * 255).astype(np.uint8))


if __name__ == "__main__":
    generate_results_nlm_with_history(SAMPLES_FOLDER_PREFIX, RESULTS_FOLDER_PREFIX, DF_HISTORY_NLM)
