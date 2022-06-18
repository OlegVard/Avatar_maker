import numpy as np
import cv2


def big_pixels(img_path, mask):
    img = np.array(cv2.imread(img_path))
    new_h = (img.shape[0] // mask - 1) * mask
    new_w = (img.shape[1] // mask - 1) * mask
    img_out = np.zeros_like(img)
    for i in range(0, img.shape[0] - mask, mask):
        for j in range(0, img.shape[1] - mask, mask):
            r = np.mean(img[i:i + mask, j:j + mask, 0])
            g = np.mean(img[i:i + mask, j:j + mask, 1])
            b = np.mean(img[i:i + mask, j:j + mask, 2])
            img_out[i:i + mask, j:j + mask] = [r, g, b]
    cv2.imwrite('pix_av.jpg', img_out[0:new_h, 0:new_w, :])


def reduce_pic(img_path, mask):
    img = np.array(cv2.imread(img_path))
    new_h = (img.shape[0] // mask - 1) * mask
    new_w = (img.shape[1] // mask - 1) * mask
    img = img[0:new_h, 0:new_w, :]
    img_out = np.zeros_like(img)
    for i in range(img.shape[0] // mask):
        for j in range(img.shape[1] // mask):
            img_out[i, j] = img[i * mask, j * mask]
    cv2.imwrite('small_img.jpg', img_out[0:int(new_h / mask), 0:int(new_w / mask), :])


def upscale_img(img_path, mask):
    img = np.array(cv2.imread(img_path))
    new_h = img.shape[0] * mask
    new_w = img.shape[1] * mask
    img = img[0:new_h, 0:new_w, :]
    img_out = np.zeros([new_h, new_w, 3])
    for i in range(img_out.shape[0]):
        for j in range(img_out.shape[1]):
            img_out[i, j] = img[i // mask, j // mask]
    cv2.imwrite('big_img.jpg', img_out)


def acid(img_path, koef):
    img = np.array(cv2.imread(img_path))
    img_out = np.zeros_like(img)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            img_out[i, j] = (koef * img[i, j] + img[i - 1, j] + img[i + 1, j] + img[i, j - 1] + img[i, j + 1])
    cv2.imwrite('acid_img2.jpg', img_out)


def bound(img_path):
    img = np.array(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY))
    img_out = np.zeros_like(img)
    prev_matrix_1 = np.array([[-1, -1, -1],
                              [0, 0, 0],
                              [1, 1, 1]])
    prev_matrix_2 = np.array([[-1, 0, 1],
                              [-1, 0, 1],
                              [-1, 0, 1]])
    for i in range(1, img_out.shape[0] - 1):
        for j in range(1, img_out.shape[1] - 1):
            img_out[i, j] = (np.abs(np.sum(img[i - 1:i + 2, j - 1:j + 2] * prev_matrix_1)) +
                             np.abs(np.sum(img[i - 1:i + 2, j - 1:j + 2] * prev_matrix_2))) / 2
    cv2.imwrite('bound_img.jpg', img_out[0:img_out.shape[0]-2, 0:img_out.shape[1]-2])
