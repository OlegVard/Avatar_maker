import numpy as np
import cv2


def big_pixels(img_path, mask):
    img = np.array(cv2.imread(img_path))
    img_out = np.zeros_like(img)
    for i in range(0, img.shape[0] - mask[0], mask[0]):
        for j in range(0, img.shape[1] - mask[1], mask[1]):
            r = np.mean(img[i:i+mask[0], j:j+mask[1], 0])
            g = np.mean(img[i:i+mask[0], j:j+mask[1], 1])
            b = np.mean(img[i:i+mask[0], j:j+mask[1], 2])
            img_out[i:i+mask[0], j:j+mask[1]] = [r, g, b]
    cv2.imwrite('pix_av.jpg', img_out)


def bound(img_path):
    mask = np.array([[-1, 1, -1],
                    [1, -1, 1],
                    [-1, 1, -1]])
    img = np.array(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY))
    if cv2.countNonZero(np.where(img > 128, 1, 0)) > cv2.countNonZero(np.where(img > 128, 0, 1)):
        img = np.where(img > 128, 0, 1)
    else:
        img = np.where(img < 128, 0, 1)

    while cv2.countNonZero(img) != 0:
        count_of_zero = cv2.countNonZero(img)
        obr_img = ~hit_miss(img, mask)
        img = img & obr_img
        if count_of_zero == cv2.countNonZero(img):
            cv2.imwrite('Avatar.jpg', img*255)
            break


def hit_miss(img_, mask):
    subset = erosion(img_, mask)
    subset2 = erosion(1 - img_, 1 - mask)
    return subset & subset2


def erosion(img_, kernel):
    ker_t = np.array([[0, 0, 0],
                      [0, 1, 0],
                      [0, 0, 0]])
    ker_f = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]])
    ret_img = np.zeros_like(img_)
    for i in range(img_.shape[0] - 2):
        for j in range(img_.shape[1] - 2):
            if array_eq(img_[i:i + 3, j:j + 3], kernel):
                ret_img[i:i + 3, j:j + 3] = ret_img[i:i + 3, j:j + 3] | ker_t
            else:
                ret_img[i:i + 3, j:j + 3] = ret_img[i:i + 3, j:j + 3] | ker_f
    return ret_img


def array_eq(img_shape, kernel):
    for i in range(img_shape.shape[0]):
        for j in range(img_shape.shape[1]):
            if img_shape[i, j] == kernel[i, j]:
                continue
            elif kernel[i, j] == -1 or kernel[i, j] == 2:
                continue
            else:
                return False
    return True


big_pixels('eg.jpg', [100, 50])
