# 23.4. 2023  Martin Bučo
# my own implementation of NLM algorithm
# main source https://www.ipol.im/pub/art/2011/bcm_nlm/article.pdf


import numpy as np
from scipy import ndimage
from skimage.io import imread, imshow, imsave
from matplotlib import pyplot as plt
import time
import cv2 as cv



def calculate_avg_img(image, patch_size):
    """
    SHOULD BE BASIC BOX FILTER plus normalizing coef
    Calculates average for each patch in the image
    Could be rewritten using FFT
    :param image:
    :param x, y: coordinates
    :param patch_size: should be odd!
    :return:
    """
    # f = (patch_size - 1) / 2
    # kernel = np.ones((f, f))
    # convolved_img = np.convolve(image, kernel)
    # return convolved_img / (patch_size * patch_size)

    # ?? which mode to choose
    return ndimage.uniform_filter(image, size=patch_size) / (patch_size * patch_size)


# which boundary conditions to use???
def calculate_distance_squared(image, x1, y1, x2, y2, compare_window_size):
    """
    Calculates simple MSE between the compared image windows
    :param image: should be a padded noisy image of type np.ndarray
    """
    f = (compare_window_size - 1) // 2

    # np uses shallow copies when slicing
    # maybe try to rewrite using for loops (avoid possible copying)
    window_1 = image[y1 - f:y1 + f, x1 - f:x1 + f]
    window_2 = image[y2 - f:y2 + f, x2 - f:x2 + f]

    # value_sum = 0
    # for y_offset in range(-f, f):
    #     for x_offset in range(-f, f):
    #         value_sum += np.square(image[y1 + y_offset, x1 + x_offset] - image[y2 + y_offset, x2 + x_offset])


    return np.sum(np.square(window_1 - window_2)) / (compare_window_size * compare_window_size)
    #return value_sum / (compare_window_size * compare_window_size)


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


def denoise_nlm(img, search_window_size, compare_window_size, filter_strength, noise_sd):
    img_height = img.shape[0]
    img_width = img.shape[1]
    search_window_offset = int((search_window_size - 1) // 2)
    compare_window_offset = int((compare_window_size - 1) // 2)
    start = time.time()
    padded_img = create_padded_img(img, compare_window_offset)
    denoised_img = np.zeros(img.shape)
    for y in range(compare_window_offset, compare_window_offset + img_height):
        for x in range(compare_window_offset, compare_window_offset + img_width):
            # weight_sum = 0
            # value = 0
            #
            # # search window cannot fall outside the boundaries of the original image
            # y_search_min = max(compare_window_offset, y - search_window_offset)
            # y_search_max = min(compare_window_offset + img_height, y + search_window_offset)
            # x_search_min = max(compare_window_offset, x - search_window_offset)
            # x_search_max = min(compare_window_offset + img_width, x + search_window_offset)
            #
            # for y_search in range(y_search_min, y_search_max):
            #     for x_search in range(x_search_min, x_search_max):
            #         if y_search != y and x_search != x:
            #             d2 = calculate_distance_squared(padded_img, x, y, x_search, y_search, compare_window_size)
            #             w = calculate_weight(d2, noise_sd, filter_strength)
            #             value += padded_img[y_search, x_search] * w
            #             weight_sum += w
            #
            # value /= weight_sum
            value = calculate_pixel_denoised_value(padded_img, img_width, img_height, noise_sd, filter_strength, x, y,
                                                   search_window_offset, compare_window_offset, compare_window_size)
            denoised_img[y - compare_window_offset, x - compare_window_offset] = value

        print(f"{y - compare_window_offset + 1}/{img_height} rows done, elapsed time: {time.time() - start}s")
    return denoised_img


def calculate_pixel_denoised_value(padded_img, img_width, img_height, noise_sd, filter_strength, x, y,
                         search_window_offset, compare_window_offset, compare_window_size):
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
    return value # x, y, value


# def denoise_nlm_multithread(img, search_window_size, compare_window_size, filter_strength, noise_sd):
#     img_height = img.shape[0]
#     img_width = img.shape[1]
#     search_window_offset = int((search_window_size - 1) // 2)
#     compare_window_offset = int((compare_window_size - 1) // 2)
#
#     start = time.time()
#     padded_img = create_padded_img(img, compare_window_offset)
#     denoised_img = np.zeros(img.shape)
#
#     import concurrent.futures
#     with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
#         futures = [executor.submit(calculate_pixel_denoised_value, padded_img, img_width, img_height,
#                                    noise_sd, filter_strength, x, y, search_window_offset,
#                                    compare_window_offset, compare_window_size)
#                    for y in range(compare_window_offset, compare_window_offset + img_height)
#                    for x in range(compare_window_offset, compare_window_offset + img_width)]
#         for i, future in enumerate(concurrent.futures.as_completed(futures)):
#             x, y, value = future.result()
#             print("finished")
#             denoised_img[y - compare_window_offset, x - compare_window_offset] = value
#
#     print(f"Elapsed time: {time.time() - start}s")
#     return denoised_img


def show_original_with_denoised(original, denoised):
    plt.figure(figsize=(12, 6))
    plt.subplot(121).title.set_text('Original'), plt.imshow(original, cmap='gray', vmin=0, vmax=1)
    plt.subplot(122).title.set_text('Denoised'), plt.imshow(denoised, cmap='gray', vmin=0, vmax=1)
    plt.show()
    
    
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


def denoise_nlm_with_history(img, history_images, search_window_size, compare_window_size, filter_strength, noise_sd):
    img_height = img.shape[0]
    img_width = img.shape[1]
    search_window_offset = int((search_window_size - 1) // 2)
    compare_window_offset = int((compare_window_size - 1) // 2)

    for i in range(len(history_images)):
        history_images[i] = create_padded_img(history_images[i], compare_window_offset)
    history_images.append(create_padded_img(img, compare_window_offset))
    
    start = time.time()
    denoised_img = np.zeros(img.shape)
    for y in range(compare_window_offset, compare_window_offset + img_height):
        for x in range(compare_window_offset, compare_window_offset + img_width):
            for padded_img in history_images:
                denoised_img[y - compare_window_offset, x - compare_window_offset] += \
                    calculate_pixel_denoised_value(padded_img, img_width, img_height, noise_sd, filter_strength, x,y,
                                                       search_window_offset, compare_window_offset, compare_window_size)
            denoised_img[y - compare_window_offset, x - compare_window_offset] /= len(history_images)
        print(f"{y - compare_window_offset + 1}/{img_height} rows done, elapsed time: {time.time() - start}s")
    return denoised_img


def denoise_nlm_with_history_smart(img, history_images, search_window_size, compare_window_size, filter_strength, noise_sd):
    img_height = img.shape[0]
    img_width = img.shape[1]
    search_window_offset = int((search_window_size - 1) // 2)
    compare_window_offset = int((compare_window_size - 1) // 2)

    for i in range(len(history_images)):
        history_images[i] = create_padded_img(history_images[i], compare_window_offset)
    history_images.append(create_padded_img(img, compare_window_offset))

    start = time.time()
    denoised_img = np.zeros(img.shape)
    for y in range(compare_window_offset, compare_window_offset + img_height):
        for x in range(compare_window_offset, compare_window_offset + img_width):
            value = 0
            for padded_img in history_images:
                weight_sum = 0

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
            denoised_img[y - compare_window_offset, x - compare_window_offset] /= len(history_images)
        print(f"{y - compare_window_offset + 1}/{img_height} rows done, elapsed time: {time.time() - start}s")
    return denoised_img


def denoise_nlm_incomplete_with_history(img, mask, history_images, search_window_size, compare_window_size,
                                        filter_strength, noise_sd):
    """
    :param mask: binary mask, that specifies which pixels were loaded correctly
    """
    img_height = img.shape[0]
    img_width = img.shape[1]
    search_window_offset = int((search_window_size - 1) // 2)
    compare_window_offset = int((compare_window_size - 1) // 2)

    inverted_mask = (np.ones(mask.shape) - mask)
    for i in range(len(history_images)):
        history_images[i] = img * mask + history_images[i] * inverted_mask
        history_images[i] = create_padded_img(history_images[i], compare_window_offset)

    start = time.time()
    denoised_img = np.zeros(img.shape)
    for y in range(compare_window_offset, compare_window_offset + img_height):
        for x in range(compare_window_offset, compare_window_offset + img_width):
            for padded_img in history_images:
                denoised_img[y - compare_window_offset, x - compare_window_offset] += \
                    calculate_pixel_denoised_value(padded_img, img_width, img_height, noise_sd, filter_strength, x, y,
                                                   search_window_offset, compare_window_offset, compare_window_size)
            denoised_img[y - compare_window_offset, x - compare_window_offset] /= len(history_images)
        print(f"{y - compare_window_offset + 1}/{img_height} rows done, elapsed time: {time.time() - start}s")
    return denoised_img


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