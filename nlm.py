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


def compare_img_quality(gt_img, noisy_img, denoised_img):
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


def calculate_weight(d_squared, sd, strength):
    return np.exp(-max(d_squared - sd * sd, 0) / (strength * strength))


def create_padded_img(img, compare_window_offset):
    return np.pad(img, compare_window_offset, mode='symmetric')


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

    return denoised_img


def denoise_nlm_with_history_averaging(img, history, search_window_size, compare_window_size, filter_strength, noise_sd,
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


def denoise_nlm_with_smart_history(img, history, search_window_size, compare_window_size, filter_strength, noise_sd,
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


if __name__ == "__main__":
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
    sd = 0.2632
    strength = 0.35 * sd #
    cw = 9  # ODD
    sw = 35  # ODD! bigger than compare win
    # sd = 0.05
    # strength = 1  # 0.078#
    # compare_window_size = 5  # ODD
    # search_window_size = 21  # ODD! bigger than compare win
    ################################################################################################################
    ################################################################################################################
    ################################################################################################################

    denoised_img = denoise_nlm(img_float, sw, cw, strength, sd)
    show_original_with_denoised(img_float, denoised_img)

    # plt.figure(figsize=(12, 6))
    # plt.subplot(121).title.set_text('Diff'), plt.imshow((denoised_img - img_float) > 0.05, cmap='gray', vmin=0, vmax=1)
    # plt.show()

    imsave(f"./data/output_images/dark_200x100/{i}.png", (denoised_img * 255).astype(np.uint8))
