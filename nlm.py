# 23.4. 2023  Martin Bučo
# my own implementation of NLM algorithm
# main source https://www.ipol.im/pub/art/2011/bcm_nlm/article.pdf


import numpy as np
from scipy import ndimage
from skimage.io import imread, imshow, imsave
from matplotlib import pyplot as plt
import time
import cv2 as cv
import pandas as pd


def calculate_distance_squared_between_images(image1, x1, y1, image2, x2, y2, compare_window_size):
    """
    Calculates simple MSE between the compared image windows, each window from a different image
    :param image1: should be a padded noisy image of type np.ndarray
    :param image2: should be a padded noisy image of type np.ndarray
    """
    f = (compare_window_size - 1) // 2

    # np uses shallow copies when slicing
    # maybe try to rewrite using for loops (avoid possible copying)
    return np.sum(np.square(
        image1[y1 - f:y1 + f, x1 - f:x1 + f] - image2[y2 - f:y2 + f, x2 - f:x2 + f])
    ) / (compare_window_size * compare_window_size)

def calculate_distance_squared(image, x1, y1, x2, y2, compare_window_size):
    """
    Calculates simple MSE between the compared image windows in the SAME image
    :param image: should be a padded noisy image of type np.ndarray
    """
    f = (compare_window_size - 1) // 2

    # np uses shallow copies when slicing
    # maybe try to rewrite using for loops (avoid possible copying)
    window_1 = image[y1 - f:y1 + f, x1 - f:x1 + f]
    window_2 = image[y2 - f:y2 + f, x2 - f:x2 + f]

    return np.sum(np.square(window_1 - window_2)) / (compare_window_size * compare_window_size)


# def calculate_distance_mean(image, x1, y1, x2, y2, compare_window_size):
#     f = (compare_window_size - 1) // 2
#     window_1 = image[y1 - f:y1 + f, x1 - f:x1 + f]
#     window_2 = image[y2 - f:y2 + f, x2 - f:x2 + f]

#     return ((np.mean(window_1) - np.mean(window_2)) ** 2) / (compare_window_size * compare_window_size)

# is it (-sd^2) or (+sd^2) ?????
def calculate_weight(d_squared, sd, strength):
    return np.exp(-max(d_squared - sd * sd, 0) / (strength * strength))

# def calculate_weight2(d_squared, sd, strength):
#     return np.exp(-(d_squared))

def create_padded_img(img, compare_window_offset):
    return np.pad(img, compare_window_offset, mode='symmetric')


def denoise_nlm(img, search_window_size, compare_window_size, filter_strength, noise_sd, are_images_padded=False):

    search_window_offset = int((search_window_size - 1) // 2)
    compare_window_offset = int((compare_window_size - 1) // 2)
    start = time.time()
    if not are_images_padded:
        padded_img = create_padded_img(img, compare_window_offset)
        img_height = img.shape[0]
        img_width = img.shape[1]
    else:
        padded_img = img
        img_height = img.shape[0] - 2 * compare_window_offset
        img_width = img.shape[1] - 2 * compare_window_offset

    denoised_img = np.zeros((img_height, img_width))
    for y in range(compare_window_offset, compare_window_offset + img_height):
        for x in range(compare_window_offset, compare_window_offset + img_width):
            value = calculate_pixel_denoised_value(padded_img, img_width, img_height, noise_sd, filter_strength, x, y,
                                                   search_window_offset, compare_window_offset, compare_window_size)
            denoised_img[y - compare_window_offset, x - compare_window_offset] = value

        #print(f"{y - compare_window_offset + 1}/{img_height} rows done, elapsed time: {time.time() - start}s")
    return denoised_img


def calculate_pixel_denoised_value(padded_img, img_width, img_height, noise_sd, filter_strength, x, y,
                         search_window_offset, compare_window_offset, compare_window_size):
    """
    Compares all compare windows (CW) inside one search window (SW)
    :param x, y: center pixel coordinates around which the SW is searched
    """
    weight_sum = 0
    value = 0

    # search window cannot fall outside the boundaries of the original image
    y_search_min = max(compare_window_offset, y - search_window_offset)
    y_search_max = min(compare_window_offset + img_height, y + search_window_offset)
    x_search_min = max(compare_window_offset, x - search_window_offset)
    x_search_max = min(compare_window_offset + img_width, x + search_window_offset)

    for y_search in range(y_search_min, y_search_max):
        for x_search in range(x_search_min, x_search_max):
            if y_search != y and x_search != x:
                d2 = calculate_distance_squared(padded_img, x, y, x_search, y_search, compare_window_size)
                w = calculate_weight(d2, noise_sd, filter_strength)
                value += padded_img[y_search, x_search] * w
                weight_sum += w

    value /= weight_sum
    return value


def calculate_pixel_denoised_value_between_images(padded_img1, padded_img2, img_width, img_height, noise_sd, filter_strength, x, y,
                         search_window_offset, compare_window_offset, compare_window_size):
    """
    Compares all compare windows (CW) inside one search window (SW)
    For each pixel in :padded_img2: compares its surrounding CW with single CW around pixel x,y from :padded_img1:
    :param x, y: center pixel coordinates around which the SW is searched
    :param img_width: width of both :padded_img1: and :padded_img2:
    :param img_height: height of both :padded_img1: and :padded_img2:
    """
    weight_sum = 0
    value = 0

    # search window cannot fall outside the boundaries of the original image
    y_search_min = max(compare_window_offset, y - search_window_offset)
    y_search_max = min(compare_window_offset + img_height, y + search_window_offset)
    x_search_min = max(compare_window_offset, x - search_window_offset)
    x_search_max = min(compare_window_offset + img_width, x + search_window_offset)

    for y_search in range(y_search_min, y_search_max):
        for x_search in range(x_search_min, x_search_max):
            if y_search != y and x_search != x:
                d2 = calculate_distance_squared_between_images(padded_img1, x, y, padded_img2,
                                                               x_search, y_search, compare_window_size)
                w = calculate_weight(d2, noise_sd, filter_strength)
                value += padded_img2[y_search, x_search] * w
                weight_sum += w

    value /= weight_sum
    return value


def show_original_with_denoised(original, denoised):
    plt.figure(figsize=(12, 6))
    plt.subplot(121).title.set_text('Original'), plt.imshow(original, cmap='gray', vmin=0, vmax=1)
    plt.subplot(122).title.set_text('Denoised'), plt.imshow(denoised, cmap='gray', vmin=0, vmax=1)
    plt.show()





# def denoise_nlm_with_history(img, history_images, search_window_size, compare_window_size, filter_strength, noise_sd):
#     """
#     :param history_images: these images are changed inside this function
#     """
#     img_height = img.shape[0]
#     img_width = img.shape[1]
#     search_window_offset = int((search_window_size - 1) // 2)
#     compare_window_offset = int((compare_window_size - 1) // 2)
#
#     for i in range(len(history_images)):
#         history_images[i] = create_padded_img(history_images[i], compare_window_offset)
#     history_images.append(create_padded_img(img, compare_window_offset))
#
#     start = time.time()
#     denoised_img = np.zeros(img.shape)
#     for y in range(compare_window_offset, compare_window_offset + img_height):
#         for x in range(compare_window_offset, compare_window_offset + img_width):
#             for padded_img in history_images:
#                 denoised_img[y - compare_window_offset, x - compare_window_offset] += \
#                     calculate_pixel_denoised_value(padded_img, img_width, img_height, noise_sd, filter_strength, x,y,
#                                                        search_window_offset, compare_window_offset, compare_window_size)
#             denoised_img[y - compare_window_offset, x - compare_window_offset] /= len(history_images)
#     return denoised_img


def nlm_with_history_averaging(img, history, search_window_size, compare_window_size, filter_strength, noise_sd,
                               n=-1, are_images_padded=False, denoised_img=None):
    compare_window_offset = int((compare_window_size - 1) // 2)
    if n == -1:
        n = len(history)

    if not are_images_padded:
        history_padded = []
        for i in range(n):
            history_padded.append(create_padded_img(history[i], compare_window_offset))
        main_img = create_padded_img(img, compare_window_offset)
    else:
        history_padded = history
        main_img = img
    img_height = img.shape[0] - 2 * compare_window_offset
    img_width = img.shape[1] - 2 * compare_window_offset

    if denoised_img is None:
        denoised_img = denoise_nlm(main_img, search_window_size, compare_window_size, filter_strength, noise_sd, True)
    else:
        assert (denoised_img.shape == (img_height, img_width))
    if n == 0:
        return denoised_img

    for i in range(n):
        history_padded += denoise_nlm(history_padded[i], search_window_size, compare_window_size, filter_strength,
                                      noise_sd, True)
    denoised_img /= n + 1

    return denoised_img

def denoise_history_averaging(img, history, n=-1):
    if n == -1:
        n = len(history)
    elif n == 0:
        return img

    denoised_img = np.zeros(img.shape)
    for i in range(n):
        denoised_img += history[i]
    denoised_img += img
    denoised_img /= n + 1

    return denoised_img

# def nlm():
#     pass
# def nlm_with_smart_history():
#     pass


def denoise_nlm_with_history_smart(img, history, search_window_size, compare_window_size, filter_strength, noise_sd,
                                   n=-1, are_images_padded=False, denoised_img=None):
    """
    :param img: noisy input image
    :param history: all images in history
    :param search_window_size:
    :param compare_window_size:
    :param filter_strength: strength of filter (usually based on 'noise_sd')
    :param noise_sd: precalculated estimated standard deviation of the noise
    :param n: number of images to use from 'history' list
    :param are_images_padded: specifies whether the 'history' images and 'img' are already padded
    :param denoised_img: image with precalculated NLM for 'main_img', can be omitted
    :return:
    """
    compare_window_offset = int((compare_window_size - 1) // 2)
    if n == -1:
        n = len(history)

    if not are_images_padded:
        history_padded = []
        for i in range(n):
            history_padded.append(create_padded_img(history[i], compare_window_offset))
        main_img = create_padded_img(img, compare_window_offset)
    else:
        history_padded = history
        main_img = img
    img_height = img.shape[0] - 2 * compare_window_offset
    img_width = img.shape[1] - 2 * compare_window_offset
    history_img_height = history[0].shape[0] - 2 * compare_window_offset
    history_img_width = history[0].shape[1] - 2 * compare_window_offset

    if denoised_img is None:
        denoised_img = denoise_nlm(main_img, search_window_size, compare_window_size, filter_strength, noise_sd, True)
    else:
        assert(denoised_img.shape == (img_height, img_width))

    if n == 0:
        return denoised_img

    return _denoise_nlm_history_smart(denoised_img, img_width, img_height, compare_window_size, search_window_size,
                                      main_img, history_padded, n, history_img_width, history_img_height,
                                      filter_strength, noise_sd)


def _denoise_nlm_history_smart(denoised_img, img_width, img_height, compare_window_size, search_window_size,
                               main_img_padded, history_padded, n, history_img_width, history_img_height,
                               filter_strength, noise_sd):
    """
    :param denoised_img: image with precalculated NLM for 'main_img_padded'
    :param img_width: width of main_img_padded before padding and denoised_img
    :param img_height: height of main_img_padded before padding and denoised_img
    :param compare_window_size:
    :param search_window_size:
    :param main_img_padded: noisy input image, already padded
    :param history_padded: all images in history with padding of half the size of compare_window_size
    :param n: number of images to use from 'history' list
    :param history_img_width: width of all images in history before padding
    :param history_img_height: height of all images in history before padding
    :param noise_sd: precalculated estimated standard deviation of the noise
    :param filter_strength: strength of filter (usually based on 'noise_sd')
    :return:
    """
    search_window_offset = int((search_window_size - 1) // 2)
    compare_window_offset = int((compare_window_size - 1) // 2)
    for y in range(compare_window_offset, compare_window_offset + img_height):
        for x in range(compare_window_offset, compare_window_offset + img_width):
            for i in range(n):
                denoised_img[y - compare_window_offset, x - compare_window_offset] += \
                    calculate_pixel_denoised_value_between_images(main_img_padded, history_padded[i], history_img_width,
                                                                  history_img_height, noise_sd,
                                                                  filter_strength, x, y, search_window_offset,
                                                                  compare_window_offset, compare_window_size)
            # 'n + 1' because denoised_img already contains denoised main img
            denoised_img[y - compare_window_offset, x - compare_window_offset] /= n + 1

    return denoised_img


# def shift_gt_img(img_gt, shift):
#     # left = 200
#     # top = 450
#     width = 100
#     height = 100
#     left = 150
#     top = 170
#     return img_gt[top + shift : top + height + shift,
#                   left + shift : left + width + shift]
#
#
# def test_history_size_vs_window_size():
#     from skimage.metrics import mean_squared_error
#     import os
#
#     sd = 0.05
#     strength = 0.078  # 0.35 * sd #
#     compare_window_size = 9  # ODD
#     compare_window_offset = int((compare_window_size - 1) // 2)
#     # search_window_size = 35
#     samples_folder_prefix = '/home/xbuco/pv162-project/data/100x100_100ns/'
#     results_folder_prefix = '/home/xbuco/pv162-project/data/results/'
#     img0 = (imread(samples_folder_prefix + f"{0}.png", as_gray=True).astype(np.float64)) / 255
#     print(f"Before padding: {img0.shape}")
#     img0 = create_padded_img(img0, compare_window_offset)
#     print(f"After padding: {img0.shape}")
#     max_win_size = 50
#     win_increment = 8
#
#     n = 92
#     history = []
#     for c in range(1, 1 + n):
#         history.append(create_padded_img(
#             (imread(samples_folder_prefix + f"{c}.png", as_gray=True).astype(np.float64)) / 255, compare_window_offset))
#     print(f"History with padding: {history[5].shape}")
#
#     img_gt_full = imread("./sample_1024.png", as_gray=True).astype(np.float64) / 255
#     img_gt = shift_gt_img(img_gt_full, 0)
#     print(f"GT img without padding: {img_gt.shape}")
#
#     df_filename = './results.csv'
#     if os.path.exists(df_filename):
#         df = pd.read_csv(df_filename)
#     else:
#         df = pd.DataFrame(columns=['sd', 'fs', 'history', 'sw', 'cw', 'MSE', 'time'])
#
#     #for search_window_size in range(35, max_win_size, win_increment):
#     start_no_history = time.time()
#     for search_window_size in [19, 27, 35, 43]:
#         start = time.time()
#         denoised_img = denoise_nlm(img0, search_window_size, compare_window_size, strength, sd, True)
#         nlm_calc_time = time.time() - start
#         mse = mean_squared_error(img_gt, denoised_img)
#         df = df.append({
#             'sd': sd,
#             'fs': strength,
#             'history': 0,
#             'sw': search_window_size,
#             'cw': compare_window_size,
#             'MSE': mse,
#             'time': nlm_calc_time
#         }, ignore_index=True)
#         df.to_csv(df_filename, index=False)
#         imsave(f"{results_folder_prefix}h0_sw{search_window_size}_cw{compare_window_size}.png",
#                (denoised_img * 255).astype(np.uint8))
#     print(f"Basic nlm lasted for: {time.time() - start_no_history}s")
#
#
#     # range(2, n, 2):
#     for history_len in [1, 2, 3, 4, 5, 8, 10, 16, 32, 50, 64, 90]:
#         # range(compare_window_size * 2 + 1, max_win_size, win_increment):
#         start_one_history_len = time.time()
#         for search_window_size in [19, 27, 35, 43]:
#             start = time.time()
#             denoised_img = denoise_nlm_with_history_smart(img0, history, search_window_size, compare_window_size,
#                                                           strength, sd, history_len, True)
#             nlm_calc_time = time.time() - start
#             mse = mean_squared_error(img_gt, denoised_img)
#             df = df.append({
#                 'sd': sd,
#                 'fs': strength,
#                 'history': history_len,
#                 'sw': search_window_size,
#                 'cw': compare_window_size,
#                 'MSE': mse,
#                 'time': nlm_calc_time
#             }, ignore_index=True)
#             df.to_csv(df_filename, index=False)
#             imsave(f"{results_folder_prefix}h{history_len}_sw{search_window_size}_cw{compare_window_size}.png",
#                    (denoised_img * 255).astype(np.uint8))
#
#         print(f"Nlm with history of len {history_len} lasted for: {time.time() - start_one_history_len}s")
#
#
#


def main():
    prefix = "/home/martinb/SCHOOL/pv162_project/"
    i = 2
    img = imread(prefix + f"data/input_images/samples_dark/{i}.png", as_gray=True).astype(np.uint8)
    img_float = img.astype(np.float64) / 255

    ################################################################################################################
    ################################################################################################################
    ################################################################################################################
    # PARAMETERS                                                                                                   #
    ################################################################################################################
    ################################################################################################################
    ################################################################################################################
    sd = 0.05
    strength = 0.078#0.35 * sd #
    compare_window_size = 9  # ODD
    search_window_size = 35  # ODD! bigger than compare win
    # sd = 0.05
    # strength = 1  # 0.078#
    # compare_window_size = 5  # ODD
    # search_window_size = 21  # ODD! bigger than compare win
    ################################################################################################################
    ################################################################################################################
    ################################################################################################################

    denoised_img = denoise_nlm(img_float, search_window_size, compare_window_size, strength, sd)
    show_original_with_denoised(img_float, denoised_img)

    # plt.figure(figsize=(12, 6))
    # plt.subplot(121).title.set_text('Diff'), plt.imshow((denoised_img - img_float) > 0.05, cmap='gray', vmin=0, vmax=1)
    # plt.show()

    imsave(f"./data/output_images/dark_200x100/{i}.png", (denoised_img * 255).astype(np.uint8))

if __name__ == "__main__":
    main()
    # test_history_size_vs_window_size()
