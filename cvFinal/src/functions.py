import numpy as np
from scipy import stats
import math
from tqdm import tqdm

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def calculate_mse(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    return mse

def filtered(homography, threshold):
    homography = np.array(homography)
    # 對每一列做標準化
    homography_std = stats.zscore(homography, axis=0)
    # 創建一個遮罩,標記出極端值的索引
    extreme_mask = np.any(abs(homography_std) > threshold, axis=1)
    # 使用遮罩過濾homography列表
    homography = homography[~extreme_mask]
    return homography

def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """
    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    x = np.arange(xmin, xmax, 1)
    y = np.arange(ymin, ymax, 1)
    one = np.ones(((xmax - xmin) * (ymax - ymin), 1), dtype=np.int32)

    mesh_x, mesh_y = np.meshgrid(x, y)
    mesh_x, mesh_y = mesh_x.flatten(), mesh_y.flatten()
    mesh_matrix = np.concatenate((mesh_x.reshape(-1, 1), mesh_y.reshape(-1, 1), one), axis=1)
    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    # dst = dst.reshape((h_dst * w_dst), ch)

    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        trans_dst = np.transpose(np.dot(H_inv, np.transpose(mesh_matrix)))
        trans_dst /= trans_dst[:, 2].reshape(-1, 1)
        # print(trans_dst)
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        mask = np.where((trans_dst[:, 0] < 0) | (trans_dst[:, 0] >= w_src) | (trans_dst[:, 1] < 0) | (trans_dst[:, 1] >= h_src))[0].tolist()

        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        trans_dst = np.delete(trans_dst, mask, axis=0)
        mesh_x = np.delete(mesh_x, mask, axis=0)
        mesh_y = np.delete(mesh_y, mask, axis=0)

        # TODO: 6. assign to destination image with proper masking
        dst_x, dst_y = mesh_x, mesh_y
        # src_x, src_y = np.floor(trans_dst[:, 0]).astype(np.int32), np.floor(trans_dst[:, 1]).astype(np.int32)
        # dst[dst_y, dst_x] = src[src_y, src_x]

        x1, y1 = np.floor(trans_dst[:, 0]).astype(np.int32), np.floor(trans_dst[:, 1]).astype(np.int32)
        x2, y2 = np.clip(x1 + 1, None, w_src - 1), np.clip(y1 + 1, None, h_src - 1)
        weight_x, weight_y = trans_dst[:, 0] - x1, trans_dst[:, 1] - y1

        dst[dst_y, dst_x] = ((1.0 - weight_x) * (1.0 - weight_y)).reshape(-1, 1) * src[y1, x1] + \
                            ((weight_x) * (1.0 - weight_y)).reshape(-1, 1) * src[y1, x2] + \
                            ((1.0 - weight_x) * (weight_y)).reshape(-1, 1) * src[y2, x1] + \
                            ((weight_x) * (weight_y)).reshape(-1, 1) * src[y2, x2]

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        trans_src = np.transpose(np.dot(H, np.transpose(mesh_matrix)))
        trans_src /= trans_src[:, 2].reshape(-1, 1)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        mask = np.where((trans_src[:, 0] < 0) | (trans_src[:, 0] >= w_dst) | (trans_src[:, 1] < 0) | (trans_src[:, 1] >= h_dst))[0].tolist()

        # TODO: 5.filter the valid coordinates using previous obtained mask
        trans_src = np.delete(trans_src, mask, axis=0)
        mesh_x = np.delete(mesh_x, mask, axis=0)
        mesh_y = np.delete(mesh_y, mask, axis=0)

        # TODO: 6. assign to destination image using advanced array indicing
        dst_x, dst_y = np.floor(trans_src[:, 0]).astype(np.int32), np.floor(trans_src[:, 1]).astype(np.int32)
        x1, y1 = mesh_x, mesh_y
        x2, y2 = np.clip(x1 + 1, None, xmax - 1), np.clip(y1 + 1, None, ymax - 1)
        weight_x, weight_y = trans_src[:, 0] - dst_x, trans_src[:, 1] - dst_y

        dst[dst_y, dst_x] = ((1.0 - weight_x) * (1.0 - weight_y)).reshape(-1, 1) * src[y1, x1] + \
                            ((weight_x) * (1.0 - weight_y)).reshape(-1, 1) * src[y1, x2] + \
                            ((1.0 - weight_x) * (weight_y)).reshape(-1, 1) * src[y2, x1] + \
                            ((weight_x) * (weight_y)).reshape(-1, 1) * src[y2, x2]

    return dst

def backward_wrapping(ref1, ref2, tar, H1_list, H2_list):
    h_frame, w_frame = tar.shape[:2]
    sum_mse = 0

    temp_frame = np.zeros_like(tar)
    tar_frame = np.zeros_like(tar)
    # 定義區塊大小
    block_size = 16

    # 計算區塊數量
    h_blocks = h_frame // block_size
    w_blocks = w_frame // block_size

    # 初始化 best_mse_map
    best_mse_blk = np.full((h_blocks, w_blocks), np.inf, dtype=np.float32)
    model_map = np.zeros((h_blocks, w_blocks), dtype=np.uint8)

    for index, H in tqdm(enumerate(H1_list)):
        # 對區塊進行逆向投影變換
        temp_frame = warping(ref1, temp_frame, H, 0, temp_frame.shape[0], 0, temp_frame.shape[1], direction='b')
        
        temp_mse_map = (temp_frame.astype(np.float32) - tar.astype(np.float32))**2
        
        for i in range(h_blocks):
            x_start, x_end = i * block_size, (i + 1) * block_size
            for j in range(w_blocks):
                y_start, y_end = j * block_size, (j + 1) * block_size
                mse = np.mean(temp_mse_map[x_start:x_end, y_start:y_end])
                if mse < best_mse_blk[i, j]:
                    best_mse_blk[i, j] = mse
                    tar_frame[x_start:x_end, y_start:y_end] = temp_frame[x_start:x_end, y_start:y_end]
                    model_map[i, j] = index

    temp_frame = np.zeros_like(tar)
    for index, H in tqdm(enumerate(H2_list)):
        # 對區塊進行逆向投影變換
        temp_frame = warping(ref2, temp_frame, H, 0, temp_frame.shape[0], 0, temp_frame.shape[1], direction='b')
        
        temp_mse_map = (temp_frame.astype(np.float32) - tar.astype(np.float32))**2
        
        for i in range(h_blocks):
            x_start, x_end = i * block_size, (i + 1) * block_size
            for j in range(w_blocks):
                y_start, y_end = j * block_size, (j + 1) * block_size
                mse = np.mean(temp_mse_map[x_start:x_end, y_start:y_end])
                if mse < best_mse_blk[i, j]:
                    best_mse_blk[i, j] = mse
                    tar_frame[x_start:x_end, y_start:y_end] = temp_frame[x_start:x_end, y_start:y_end]
                    model_map[i, j] = index + len(H1_list)

    return tar_frame, model_map, np.sum(best_mse_blk)


def choose_model(ref1, ref2, tar, H1_list, H2_list):
    h_frame, w_frame = tar.shape[:2]
    block_size = 16
    h_blocks = h_frame // block_size
    w_blocks = w_frame // block_size
    # 初始化
    tar_frame = np.zeros_like(tar)
    temp_frame = np.zeros_like(tar)
    model_list1 = []
    model_list2 = []
    mse_map_list1 = []
    mse_map_list2 = []
    best_mse_blk = np.full((h_blocks, w_blocks), np.inf, dtype=np.float32)

    temp = [np.eye(3)]
    for H in H1_list:
        temp.append(H)
    H1_list = temp
    
    temp = [np.eye(3)]
    for H in H2_list:
        temp.append(H)
    H2_list = temp

    for H in tqdm(H1_list):
        temp_frame = warping(ref1, temp_frame, H, 0, temp_frame.shape[0], 0, temp_frame.shape[1], direction='b')
        
        temp_mse_map = (temp_frame.astype(np.float32) - tar.astype(np.float32))**2

        temp_mse_map = np.sum(temp_mse_map, axis=2, keepdims=True)

        # 重塑 temp_mse_map 為區塊大小的形狀
        reshaped = temp_mse_map.reshape(h_blocks, block_size, w_blocks, block_size)
        
        # 計算每個區塊的均方誤差
        block_mse = reshaped.mean(axis=(1, 3))

        mse_map_list1.append(block_mse)

    for H in tqdm(H2_list):
        temp_frame = warping(ref2, temp_frame, H, 0, temp_frame.shape[0], 0, temp_frame.shape[1], direction='b')
        
        temp_mse_map = (temp_frame.astype(np.float32) - tar.astype(np.float32))**2

        temp_mse_map = np.sum(temp_mse_map, axis=2, keepdims=True)

        # 重塑 temp_mse_map 為區塊大小的形狀
        reshaped = temp_mse_map.reshape(h_blocks, block_size, w_blocks, block_size)
        
        # 計算每個區塊的均方誤差
        block_mse = reshaped.mean(axis=(1, 3))
        mse_map_list2.append(block_mse)


    while(len(model_list1)+len(model_list2) < 12):
        best_index = 0
        bset_mse = np.float32(np.inf)
        for index, mse_blk in enumerate(mse_map_list1):
            union_mse_blk = np.minimum(mse_blk, best_mse_blk)
            mse = np.sum(np.sort(union_mse_blk.flatten())[:13000])
            if mse < bset_mse:
                bset_mse = mse
                best_index = index

        for index, mse_blk in enumerate(mse_map_list2):
            union_mse_blk = np.minimum(mse_blk, best_mse_blk)
            mse = np.sum(np.sort(union_mse_blk.flatten())[:13000])
            if mse < bset_mse:
                bset_mse = mse
                best_index = index + len(mse_map_list1)

        if best_index < len(mse_map_list1):
            best_mse_blk = np.minimum(best_mse_blk, mse_map_list1[best_index])
            model_list1.append(H1_list[best_index])
            mse_map_list1.pop(best_index)
            H1_list.pop(best_index)

        else:
            best_index -= len(mse_map_list1)
            best_mse_blk = np.minimum(best_mse_blk, mse_map_list2[best_index])
            model_list2.append(H2_list[best_index])
            mse_map_list2.pop(best_index)
            H2_list.pop(best_index)

    return model_list1, model_list2