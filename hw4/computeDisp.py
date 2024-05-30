import numpy as np
import cv2.ximgproc as xip

def census_cost(local_L, local_R):
    disparity = np.sum(np.abs(local_L - local_R), axis=1)

    return disparity

def apply_threshold(Image_window, middle_value):
    for i in range(Image_window.shape[0]):
        for j in range(Image_window.shape[1]):
            if Image_window[i, j] >= middle_value:
                Image_window[i, j] = 0
            else:
                Image_window[i, j] = 1
                
    return Image_window


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)
    window_size = 4
    pad_size = window_size // 2

    ############### Start ###############
    
    # Doing image padding
    img_L = np.zeros((h + pad_size * 2, w + pad_size * 2, ch), dtype=np.float32)
    img_R = np.zeros((h + pad_size * 2, w + pad_size * 2, ch), dtype=np.float32)
    img_L[pad_size:-pad_size, pad_size:-pad_size, :] = Il
    img_R[pad_size:-pad_size, pad_size:-pad_size, :] = Ir
    
    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both "Il to Ir" and "Ir to Il" for later left-right consistency
    
            
    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    
    # Initial
    cost_L, cost_R = (np.zeros((h, w, max_disp), dtype=np.float32) for _ in range(2))
    local_binary_IL, local_binary_IR = (np.zeros((h, w, window_size, window_size, ch), dtype=np.float32) for _ in range(2))
    
    for i in range(h):
        for j in range(w):
            IL_window = img_L[i : i + window_size, j : j + window_size, :].copy()
            IR_window = img_R[i : i + window_size, j : j + window_size, :].copy()
            
            for c in range(ch):
                apply_threshold(IL_window[:, :, c],IL_window[pad_size, pad_size, c])
                apply_threshold(IR_window[:, :, c],IR_window[pad_size, pad_size, c])
            
            local_binary_IL[i, j] = IL_window
            local_binary_IR[i, j] = IR_window

    # reshape to 3d
    local_binary_IL = local_binary_IL.reshape(h, w, -1) # (h, w, 27)
    local_binary_IR = local_binary_IR.reshape(h, w, -1) # (h, w, 27)
    
    
    for i in range(h):
        for j in range(w):
            local_L = np.expand_dims(local_binary_IL[i, j].copy(), axis=0)
            
            # left -> right
            if j < max_disp - 1: 
                local_R = np.flip(local_binary_IR[i, : j+1].copy(), 0)
                disparity = census_cost(local_L, local_R)
                cost_L[i, j, :j+1] = disparity
                cost_L[i, j, j+1:] = disparity[-1]
            else:
                local_R = np.flip(local_binary_IR[i, (j - max_disp + 1): j + 1].copy(), 0)
                disparity = census_cost(local_L, local_R)
                cost_L[i, j, :] = disparity
            
            local_R = np.expand_dims(local_binary_IR[i, j].copy(), axis=0)
            # right -> left
            if j + max_disp > w:
                disparity = census_cost(local_L, local_R)
                cost_R[i, j, :w - j] = disparity
                cost_R[i, j, w - j:] = disparity[-1]
            else:
                local_L = local_binary_IL[i, j : j + max_disp].copy()
                disparity = census_cost(local_L, local_R)
                cost_R[i, j, :] = disparity
    
    for d in range(max_disp):
        cost_L[:, :, d] = xip.jointBilateralFilter(Il, cost_L[:, :, d], 16, 12, 12)
        cost_R[:, :, d] = xip.jointBilateralFilter(Ir, cost_R[:, :, d], 16, 12, 12)  

    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Winner-take-all
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    win_L = np.argmin(cost_L, axis=2)
    win_R = np.argmin(cost_R, axis=2)
    for i in range(h):
        for j in range(w):
            if win_L[i, j] != win_R[i, j - win_L[i, j]]:
                win_L[i, j]=-1                
    
    for i in range(h):
        for j in range(w):
            if win_L[i, j] == -1:
                idx_L = j - 1
                while idx_L >= 0 and win_L[i, idx_L] == -1:
                    idx_L -= 1
                
                FL = win_L[i, idx_L] if idx_L >= 0 else float('inf')

                idx_R = j + 1
                while idx_R < w and win_L[i, idx_R] == -1:
                    idx_R += 1

                FR = win_L[i, idx_R] if idx_R < w else float('inf')
                
                win_L[i, j] = min(FL, FR)

    labels = xip.weightedMedianFilter(Il.astype(np.uint8), win_L.astype(np.uint8), 11, 1)
    return labels.astype(np.uint8)
    