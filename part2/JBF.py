import numpy as np
import cv2
import time

np.set_printoptions(threshold=np.inf)


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6 * sigma_s + 1
        self.pad_w = 3 * sigma_s
        self.offset = self.wndw_size // 2
        self.sigma_r_dash = 1 / (-2 * self.sigma_r**2)
        self.sigma_s_dash = 1 / (-2 * self.sigma_s**2)

        xq, yq = np.meshgrid(np.arange(self.wndw_size), np.arange(self.wndw_size))
        self.Gs = np.exp(
            self.sigma_s_dash
            * (((self.wndw_size // 2 - xq) ** 2) + (self.wndw_size // 2 - yq) ** 2)
        )

    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(
            img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE
        ).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(
            guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE
        ).astype(np.int32)

        # normalize
        guidance = (guidance / 255).astype(np.float64)
        padded_guidance = (padded_guidance / 255).astype(np.float64)

        w, h, c = img.shape
 
        Ip_dash = img.astype(np.int32)
        numerator, denominator = 0, 0
        for row in range(self.wndw_size):
            for col in range(self.wndw_size):
                # (Tp - Tq) ** 2
                I_diff = (guidance - padded_guidance[row : row + h, col : col + w]) ** 2
                Iq = padded_img[row : row + h, col : col + w]

                if len(I_diff.shape) == 3:
                    Gr = np.exp(
                        self.sigma_r_dash
                        * (I_diff[:, :, 0] + I_diff[:, :, 1] + I_diff[:, :, 2])
                    )
                else:
                    Gr = np.exp(self.sigma_r_dash * I_diff)

                mul = self.Gs[row][col] * Gr
                numerator += np.stack(
                    [
                        mul * Iq[:, :, 0],
                        mul * Iq[:, :, 1],
                        mul * Iq[:, :, 2],
                    ],
                    axis=-1,
                )

                denominator += np.stack([mul, mul, mul], axis=-1)

        Ip_dash = (numerator / denominator).astype(np.int32)
        print(Ip_dash)

        return np.clip(Ip_dash, 0, 255).astype(np.uint8)
