import numpy as np
from skimage.io import imread, imshow, imsave
import time
import pandas as pd
from skimage.metrics import mean_squared_error
import os
from nlm import denoise_nlm_with_smart_history, denoise_nlm, denoise_history_averaging, \
    denoise_nlm_with_history_averaging, create_padded_img


GT_FILEPATH = "/home/xbuco/pv162-project/cropped_sample1024_512x512.png"
SAMPLES_FOLDER_PREFIX = '/home/xbuco/pv162-project/data/samples_100x100_100ns/'
RESULTS_FOLDER_PREFIX = '/home/xbuco/pv162-project/data/results/'
DF_HISTORY_SHIFTED_NLM = './history_nlm_shifted_results.csv'
DF_HISTORY_NLM = './history_nlm_results.csv'
DF_NLM_AVG = './nlm_history_avg_results.csv'
DF_NLM_AVG_SHIFTED = './nlm_history_avg_results_shifted.csv'
DF_AVG = './history_avg.csv'
DF_AVG_SHIFTED = './history_avg_shifted.csv'

# parameters determined using IPOL paper, search_window_size should be 35
SD = 0.2632
STRENGTH = 0.35 * SD  # 0.078
COMPARE_WINDOW_SIZE = 9  # ODD
COMPARE_WINDOW_OFFSET = int((COMPARE_WINDOW_SIZE - 1) // 2)


def shift_gt_img(img_gt, shift):
    width = 100
    height = 100
    left = 150
    top = 170
    return img_gt[top + shift : top + height + shift,
                  left + shift : left + width + shift]


def generate_results_history_averaging(samples_folder_prefix, results_folder_prefix, df_filename, history_lengths):
    img0 = (imread(samples_folder_prefix + f"{0}.png", as_gray=True).astype(np.float64)) / 255

    n = max(history_lengths) + 1
    history = []
    for c in range(1, 1 + n):
        history.append(
            (imread(samples_folder_prefix + f"{c}.png", as_gray=True).astype(np.float64)) / 255)

    img_gt_full = imread(GT_FILEPATH, as_gray=True).astype(np.float64) / 255
    img_gt = shift_gt_img(img_gt_full, 0)

    if os.path.exists(df_filename):
        df = pd.read_csv(df_filename)
    else:
        df = pd.DataFrame(columns=['sd', 'fs', 'history', 'MSE', 'time'])

    print(f"Starting history averaging")
    for history_len in history_lengths:
        start = time.time()
        denoised_img = denoise_history_averaging(img0, history, history_len)
        nlm_calc_time = time.time() - start
        mse = mean_squared_error(img_gt, denoised_img)
        df = df.append({
            'sd': SD,
            'fs': STRENGTH,
            'history': history_len,
            'MSE': mse,
            'time': nlm_calc_time
        }, ignore_index=True)
        df.to_csv(df_filename, index=False)
        imsave(f"{results_folder_prefix}havg_h{history_len}.png",
               (denoised_img * 255).astype(np.uint8))


def generate_results_nlm_history_averaging(samples_folder_prefix, results_folder_prefix, df_filename, history_lengths,
                                           sw_sizes):
    img0 = (imread(samples_folder_prefix + f"{0}.png", as_gray=True).astype(np.float64)) / 255  # no padding
    # print(img0.shape)
    n = max(history_lengths) + 1
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

    # precalculating all the NLMs for each image,
    # so that for each number in 'history_length' we only need to do averaging
    for search_window_size in sw_sizes:
        denoised_history = []
        for i in range(n):
            denoised_history.append(
                denoise_nlm(history[i], search_window_size, COMPARE_WINDOW_SIZE, STRENGTH, SD, True)
            )
            # print(denoised_history[0].shape)
        for history_len in history_lengths:
            start = time.time()
            # this operation would take O(n^2) time
            # denoised_img = denoise_nlm_with_history_averaging(img0, history, search_window_size, COMPARE_WINDOW_SIZE,
            #                                                   STRENGTH, SD, history_len, True)
            denoised_img = denoise_history_averaging(img0, denoised_history, history_len)
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
            imsave(f"{results_folder_prefix}nlm_havg_h{history_len}_sw{search_window_size}_cw{COMPARE_WINDOW_SIZE}.png",
                   (denoised_img * 255).astype(np.uint8))

    # denoised_history = {}
    # for search_window_size in sw_sizes:
    #     denoised_history[search_window_size] = []
    #     for i in range(max_history):
    #         denoised_history[search_window_size].append(
    #             denoise_nlm(img0, search_window_size, COMPARE_WINDOW_SIZE, STRENGTH, SD, True)
    #         )
    #
    # for history_len in history_lengths:
    #     print(f"Starting nlm with history of len: {history_len}")
    #     for search_window_size in sw_sizes:
    #         start = time.time()
    #         # this operation would take O(n^2) time
    #         # denoised_img = denoise_nlm_with_history_averaging(img0, history, search_window_size, COMPARE_WINDOW_SIZE,
    #         #                                                   STRENGTH, SD, history_len, True)
    #         denoised_img = denoise_history_averaging(img0, denoised_history[search_window_size], history_len)
    #         nlm_calc_time = time.time() - start
    #         mse = mean_squared_error(img_gt, denoised_img)
    #         df = df.append({
    #             'sd': SD,
    #             'fs': STRENGTH,
    #             'history': history_len,
    #             'sw': search_window_size,
    #             'cw': COMPARE_WINDOW_SIZE,
    #             'MSE': mse,
    #             'time': nlm_calc_time
    #         }, ignore_index=True)
    #         df.to_csv(df_filename, index=False)
    #         imsave(f"{results_folder_prefix}nlm_havg_h{history_len}_sw{search_window_size}_cw{COMPARE_WINDOW_SIZE}.png",
    #                (denoised_img * 255).astype(np.uint8))


def generate_results_nlm_with_smart_history(samples_folder_prefix, results_folder_prefix, df_filename, history_lengths,
                                           sw_sizes):
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

    for search_window_size in sw_sizes:
        print(f"Starting nlm with sw: {search_window_size}")
        start = time.time()
        denoised_img_nlm = denoise_nlm(img0, search_window_size, COMPARE_WINDOW_SIZE, STRENGTH, SD, True)
        nlm_calc_time = time.time() - start
        mse = mean_squared_error(img_gt, denoised_img_nlm)
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
        imsave(f"{results_folder_prefix}nlm_h0_sw{search_window_size}_cw{COMPARE_WINDOW_SIZE}.png",
               (denoised_img_nlm * 255).astype(np.uint8))

        for history_len in history_lengths:
            start = time.time()
            denoised_img = denoise_nlm_with_smart_history(img0, history, search_window_size, COMPARE_WINDOW_SIZE,
                                                          STRENGTH, SD, history_len, True, denoised_img_nlm)
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
            imsave(
                f"{results_folder_prefix}nlm_smart_h{history_len}_sw{search_window_size}_cw{COMPARE_WINDOW_SIZE}.png",
                (denoised_img * 255).astype(np.uint8))

    # for history_len in history_lengths:
    #     print(f"Starting nlm with history of len: {history_len}")
    #     for search_window_size in sw_sizes:
    #         start = time.time()
    #         denoised_img = denoise_nlm_with_smart_history(img0, history, search_window_size, COMPARE_WINDOW_SIZE,
    #                                                       STRENGTH, SD, history_len, True)
    #         nlm_calc_time = time.time() - start
    #         mse = mean_squared_error(img_gt, denoised_img)
    #         df = df.append({
    #             'sd': SD,
    #             'fs': STRENGTH,
    #             'history': history_len,
    #             'sw': search_window_size,
    #             'cw': COMPARE_WINDOW_SIZE,
    #             'MSE': mse,
    #             'time': nlm_calc_time
    #         }, ignore_index=True)
    #         df.to_csv(df_filename, index=False)
    #         imsave(f"{results_folder_prefix}nlm_smart_h{history_len}_sw{search_window_size}_cw{COMPARE_WINDOW_SIZE}.png",
    #                (denoised_img * 255).astype(np.uint8))


if __name__ == "__main__":
    history_lengths = [1, 2, 3, 4, 5, 8, 10, 16, 32, 50, 64]
    sw_sizes = [19, 27, 35, 43]  # [19, 35, 51, 67]

    ################################################################################################################
    # Najskor posun tie vygenerovane itemy do shifted a uloz results.csv:
    # cd /home/xbuco/pv162-project
    # mv history_nlm_shifted_results.csv  history_nlm_shifted_results_17_6_BACKUP.csv
    ################################################################################################################
    print("Generating results for NLM with smart history, for shifted input images")
    generate_results_nlm_with_smart_history(SAMPLES_FOLDER_PREFIX + 'shifted/', RESULTS_FOLDER_PREFIX + 'shifted/',
                                            DF_HISTORY_SHIFTED_NLM, history_lengths, sw_sizes)
    print("Generating results for NLM with smart history, for default input images")
    generate_results_nlm_with_smart_history(SAMPLES_FOLDER_PREFIX + 'default/', RESULTS_FOLDER_PREFIX + 'default/',
                                            DF_HISTORY_NLM, history_lengths, sw_sizes)

    print("Generating results for NLM with history averaging, for shifted input images")
    generate_results_nlm_history_averaging(SAMPLES_FOLDER_PREFIX + 'shifted/', RESULTS_FOLDER_PREFIX + 'shifted/',
                                           DF_NLM_AVG_SHIFTED, history_lengths, sw_sizes)
    print("Generating results for NLM with history averaging, for default input images")
    generate_results_nlm_history_averaging(SAMPLES_FOLDER_PREFIX + 'default/', RESULTS_FOLDER_PREFIX + 'default/',
                                           DF_NLM_AVG, history_lengths, sw_sizes)

    print("Generating results for history averaging, for shifted input images")
    generate_results_history_averaging(SAMPLES_FOLDER_PREFIX + 'shifted/', RESULTS_FOLDER_PREFIX + 'shifted/',
                                       DF_AVG_SHIFTED, history_lengths)
    print("Generating results for history averaging, for default input images")
    generate_results_history_averaging(SAMPLES_FOLDER_PREFIX + 'default/', RESULTS_FOLDER_PREFIX + 'default/',
                                       DF_AVG, history_lengths)



