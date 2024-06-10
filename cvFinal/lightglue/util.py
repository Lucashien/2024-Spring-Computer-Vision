import cv2
import scipy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

prc_ord = np.array([
    [0, 0, -1,0], [1, 32, -1,0], [2, 16, 0, 32], [3, 8, 0, 16], [4, 4, 0, 8],
    [5, 2, 0, 4], [6, 1, 0, 2], [7, 3, 2, 4], [8, 6, 4, 8], [9, 5, 4, 6],
    [10, 7, 6, 8], [11, 12, 8, 16], [12, 10, 8, 12], [13, 9, 8, 10],
    [14, 11, 10, 12], [15, 14, 12, 16], [16, 13, 12, 14], [17, 15, 14, 16],
    [18, 24, 16, 32], [19, 20, 16, 24], [20, 18, 16, 20], [21, 17, 16, 18],
    [22, 19, 18, 20], [23, 22, 20, 24], [24, 21, 20, 22], [25, 23, 22, 24],
    [26, 28, 24, 32], [27, 26, 24, 28], [28, 25, 24, 26], [29, 27, 26, 28],
    [30, 30, 28, 32], [31, 29, 28, 30], [32, 31, 30, 32], [33, 64, -1,0],
    [34, 48, 32, 64], [35, 40, 32, 48], [36, 36, 32, 40], [37, 34, 32, 36],
    [38, 33, 32, 34], [39, 35, 34, 36], [40, 38, 36, 40], [41, 37, 36, 38],
    [42, 39, 38, 40], [43, 44, 40, 48], [44, 42, 40, 44], [45, 41, 40, 42],
    [46, 43, 42, 44], [47, 46, 44, 48], [48, 45, 44, 46], [49, 47, 46, 48],
    [50, 56, 48, 64], [51, 52, 48, 56], [52, 50, 48, 52], [53, 49, 48, 50],
    [54, 51, 50, 52], [55, 54, 52, 56], [56, 53, 52, 54], [57, 55, 54, 56],
    [58, 60, 56, 64], [59, 58, 56, 60], [60, 57, 56, 58], [61, 59, 58, 60],
    [62, 62, 60, 64], [63, 61, 60, 62], [64, 63, 62, 64], [65, 96, -1,0],
    [66, 80, 64, 96], [67, 72, 64, 80], [68, 68, 64, 72], [69, 66, 64, 68],
    [70, 65, 64, 66], [71, 67, 66, 68], [72, 70, 68, 72], [73, 69, 68, 70],
    [74, 71, 70, 72], [75, 76, 72, 80], [76, 74, 72, 76], [77, 73, 72, 74],
    [78, 75, 74, 76], [79, 78, 76, 80], [80, 77, 76, 78], [81, 79, 78, 80],
    [82, 88, 80, 96], [83, 84, 80, 88], [84, 82, 80, 84], [85, 81, 80, 82],
    [86, 83, 82, 84], [87, 86, 84, 88], [88, 85, 84, 86], [89, 87, 86, 88],
    [90, 92, 88, 96], [91, 90, 88, 92], [92, 89, 88, 90], [93, 91, 90, 92],
    [94, 94, 92, 96], [95, 93, 92, 94], [96, 95, 94, 96], [97, 128, -1,0],
    [98, 112, 96, 128], [99, 104, 96, 112], [100, 100, 96, 104], [101, 98, 96, 100],
    [102, 97, 96, 98], [103, 99, 98, 100], [104, 102, 100, 104], [105, 101, 100, 102],
    [106, 103, 102, 104], [107, 108, 104, 112], [108, 106, 104, 108], [109, 105, 104, 106],
    [110, 107, 106, 108], [111, 110, 108, 112], [112, 109, 108, 110], [113, 111, 110, 112],
    [114, 120, 112, 128], [115, 116, 112, 120], [116, 114, 112, 116], [117, 113, 112, 114],
    [118, 115, 114, 116], [119, 118, 116, 120], [120, 117, 116, 118], [121, 119, 118, 120],
    [122, 124, 120, 128],  [123, 122, 120, 124], [124, 121, 120, 122], [125, 123, 122, 124],
    [126, 126, 124, 128], [127, 125, 124, 126], [128, 127, 126, 128]
])

def itr_dilation(image, itr):
    img = np.copy(image)
    kernel = np.ones((3, 3))
    img = cv2.dilate(img, kernel, iteration=itr)

    return img

def blk_dilation(image):
    img = np.copy(image)
    for h in range(0, 2160, 16):
        for w in range(0, 3840, 16):
            if (image[h, w] == 255):
                img[h:h+16, w:w+16] = 255
            else:
                img[h:h+16, w:w+16] = 0

def sum_blk(image, block=16):
    h, w = image.shape
    b_h = h // block
    b_w = w // block

    img_blk = np.zeros((b_h, b_w))

    for y in range(b_h):
        for x in range(b_w):
            img_blk[y, x] = np.sum(image[y*block:(y+1)*block, x*block:(x+1)*block])

    return img_blk

def gen_selection_map(img0, img1, block=16):
    h, w = img0.shape
    b_h = h // block
    b_w = w // block

    # calculate the error between two reference frame of each pixel
    err = np.abs(img1 - img0).astype(int)

    # calculate the error between two reference frame of each block
    err_blk = sum_blk(err, block)

    # select 13000 blocks based on block error
    # set up block map, 1 for evaluation, 0 for skip
    blk_srt = np.argsort(err_blk.flatten())
    blk_map = np.zeros((b_h * b_w, 1), dtype=np.uint8)
    blk_map[blk_srt[:13000]] = 1
    blk_map = blk_map.reshape((b_h, b_w))

    return blk_map

def jbf(src, gdc, var_s, kernel_size):
        # check if the size of image and guidance are matched
        src_h, src_w = src.shape
        gdc_h, gdc_w = gdc.shape
        if src_h != gdc_h or src_w != gdc_w:
            raise ValueError('the size of image and guidance are not matched')
        
        # set parameters
        pad_size = kernel_size // 2
        
        # pad the image and guidance with reflecting boundary
        border_type = cv2.BORDER_REFLECT
        pad_src = cv2.copyMakeBorder(
            src, pad_size, pad_size, pad_size, pad_size, border_type
        ).astype(np.uint8)
        pad_gdc = cv2.copyMakeBorder(
            gdc, pad_size, pad_size, pad_size, pad_size, border_type
        ).astype(np.uint8)

        # generate the spatial kernel
        dst = np.linspace(-pad_size, pad_size, kernel_size)
        dst_x, dst_y = np.meshgrid(dst, dst)
        gs = np.exp(-(dst_x ** 2 + dst_y ** 2) / 2 / var_s)
        
        # implement the convolution of bilateral filter
        output = np.zeros_like(src).astype(np.float16)
        wgt_sum = np.zeros((gdc_h, gdc_w))
        for i in range(kernel_size):
            for j in range(kernel_size):
                gr = (gdc == pad_gdc[i:i+gdc_h, j:j+gdc_w])
                wgt = gs[i, j] * gr
                output += pad_src[i:i+src_h, j:j+src_w] * wgt
                wgt_sum += wgt
        
        output /= wgt_sum

        return output

def softmax(x0, x1):
    x = np.array([x0, x1])
    return scipy.special.softmax(x, axis=0)

def mask_by_id(seg_map, id):
    mask = np.isin(seg_map, id).astype(np.bool_)
    return mask

def extract_features(img):
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(img, None)
    return kp, des

def bf_match(des1, des2, thr=0.45):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good_matches = [m for m in matches if m.distance < thr]
    return good_matches

def homography(kp_src, kp_dst):
    if len(kp_src) < 4:
        raise ValueError("Not enough matches to compute homography")
    H, _ = cv2.findHomography(kp_src, kp_dst, cv2.RANSAC, 5.0, maxIters=50000)
    return H

def backwarp(src, H, mask):
    h_src, w_src = src.shape
    h_dst, w_dst = mask.shape
    dst = np.zeros((h_dst, w_dst))
    x = np.linspace(0, w_dst, w_dst)
    y = np.linspace(0, h_dst, h_dst)
    intrp = scipy.interpolate.RegularGridInterpolator(
        (x, y), src.T, method='linear')
    n = np.sum(mask)
    H_inv = np.linalg.inv(H)
    x, y = np.meshgrid(range(w_dst), range(h_dst))
    xy_mesh = np.array([x, y])
    xy_mesh = np.moveaxis(xy_mesh, 0, 2)
    xy_mesh = xy_mesh[mask]
    xy_mesh = np.concatenate((xy_mesh, np.ones((n, 1))), axis=1)
    uv_mesh = H_inv @ xy_mesh.T
    uv_mesh = (uv_mesh / uv_mesh[2, :]).T
    vld = (uv_mesh[:, 0] >= 0) & (uv_mesh[:, 0] < w_src) \
          & (uv_mesh[:, 1] >= 0) & (uv_mesh[:, 1] < h_src)

    dst[y[mask][vld], x[mask][vld]] = intrp(uv_mesh[vld][:, :2])
    return np.clip(dst, 0, 255).astype(np.uint8)


def iou(src, gt):
    intersection = np.logical_and(src, gt).sum()
    union = np.logical_or(src, gt).sum()
    iou = intersection / union if union != 0 else 0
    return iou

def mask_se(img0, img1, mask):
    img0 = img0.astype(np.float32)
    img1 = img1.astype(np.float32)
    se = ((img0 - img1) ** 2).astype(np.float32)
    se = se * mask
    return se


if __name__ == '__main__':
    read_mode = cv2.IMREAD_GRAYSCALE
    src = cv2.imread('dataset/000.png', read_mode)
    dst = np.zeros((2160, 3840))
    mask = np.zeros((2160, 3840), dtype=np.bool_)
    for i in range(2160):
        for j in range(3840):
            if (2 * i - j >= -300) and (2 * i - j <= 300):
                mask[i, j] = 1
    
    H =  np.array([
        [0.5, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    out = backwarp(src, dst, H, mask)

    matplotlib.use('Agg')
    fig = plt.figure()
    plt.imshow(out, cmap='gray')
    plt.savefig('test.png')

    