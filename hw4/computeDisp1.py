import numpy as np
import cv2.ximgproc as xip


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    # labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)
    window_size = 3
    pad_size = window_size // 2

    pad_Il = np.zeros(((h + pad_size * 2), (w + pad_size * 2), ch), dtype=np.float32)
    pad_Ir = np.zeros(((h + pad_size * 2), (w + pad_size * 2), ch), dtype=np.float32)
    pad_Il[1: h + 1, 1: w + 1, :] = Il
    pad_Ir[1: h + 1, 1: w + 1, :] = Ir

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency
    cost_l = np.zeros((h, w, max_disp), dtype=np.float32)
    cost_r = np.zeros((h, w, max_disp), dtype=np.float32)
    bin_l = np.empty((h, w, ch * window_size * window_size), dtype=np.int)
    bin_r = np.empty((h, w, ch * window_size * window_size), dtype=np.int)
    # Census cost
    for row in range(h):
        for col in range(w):
            local_bin_l, local_bin_r = np.array([], dtype=np.int), np.array([], dtype=np.int)
            for channel in range(ch):
                # Local binary pattern
                local_Il = pad_Il[row: row + window_size, col: col + window_size, channel]
                local_Ir = pad_Ir[row: row + window_size, col: col + window_size, channel]
                mid_l = local_Il[pad_size, pad_size]
                mid_r = local_Ir[pad_size, pad_size]
                local_bin_l = np.concatenate((local_bin_l, (local_Il > mid_l).astype(np.int).flatten()))
                local_bin_r = np.concatenate((local_bin_r, (local_Ir > mid_r).astype(np.int).flatten()))
            # Hamming distance
            bin_l[row, col, :] = local_bin_l
            bin_r[row, col, :] = local_bin_r

    for row in range(h):
        for col in range(w):
            # left
            if col < max_disp - 1:
                local_bin_l = np.tile(bin_l[row, col], (col + 1, 1))
                local_bin_r = np.flip(bin_r[row, 0: col + 1], axis=0)
                cost_l[row, col, 0: col + 1] = np.sum(np.abs(local_bin_l - local_bin_r), axis=1)
                # Set costs of out-of-bound pixels = cost of closest valid pixel
                cost_l[row, col, col + 1:] = cost_l[row, col, col]
            else:
                local_bin_l = np.tile(bin_l[row, col], (max_disp, 1))
                local_bin_r = np.flip(bin_r[row, col - max_disp + 1: col + 1], axis=0)
                cost_l[row, col] = np.sum(np.abs(local_bin_l - local_bin_r), axis=1)
            # right
            if col + max_disp > w:
                local_bin_r = np.tile(bin_r[row, col], (w - col, 1))
                local_bin_l = bin_l[row, col: w]
                cost_r[row, col, 0: w - col] = np.sum(np.abs(local_bin_r - local_bin_l), axis=1)
                # Set costs of out-of-bound pixels = cost of closest valid pixel
                cost_r[row, col, w - col :] = cost_r[row, col, w - col - 1]
            else:
                local_bin_r = np.tile(bin_r[row, col], (max_disp, 1))
                local_bin_l = bin_l[row, col: col + max_disp]
                cost_r[row, col] = np.sum(np.abs(local_bin_r - local_bin_l), axis=1)

    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    for dis in range(max_disp):
        cost_l[:, :, dis] = xip.jointBilateralFilter(Il, cost_l[:, :, dis], 19, 9.5, 9.5)
        cost_r[:, :, dis] = xip.jointBilateralFilter(Ir, cost_r[:, :, dis], 19, 9.5, 9.5)

    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    winner_l = np.argmin(cost_l, axis=2)
    winner_r = np.argmin(cost_r, axis=2)

    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    # Left-right consistency check
    for row in range(h):
        for col in range(w):
            if winner_l[row, col] != winner_r[row, col - winner_l[row, col]]:
                winner_l[row, col] = -1

    # Hole filling
    for row in range(h):
        for col in range(w):
            if winner_l[row, col] == -1:
                FL_list = winner_l[row, :col][winner_l[row, :col] != -1]
                if FL_list.size > 0:
                    FL = FL_list[-1]
                else:
                    FL = 10000

                FR_list = winner_l[row, col + 1: w][winner_l[row, col + 1: w] != -1]
                if FR_list.size > 0:
                    FR = FR_list[0]
                else:
                    FR = 10000

                winner_l[row, col] = min(FL, FR)

    # Weighted median filtering        
    labels = xip.weightedMedianFilter(Il.astype(np.uint8), winner_l.astype(np.uint8), 17, 1)
    return labels.astype(np.uint8)