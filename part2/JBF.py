import numpy as np
import cv2

np.set_printoptions(threshold=np.inf)


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6 * sigma_s + 1
        self.pad_w = 3 * sigma_s

    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(
            img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE
        ).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(
            guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE
        ).astype(np.int32)

        guidance  = (guidance / 255).astype(np.float64)
        self.img = img
        self.guidance = guidance

        self.padded_img = padded_img
        self.padded_guidance = padded_guidance

        w, h, c = img.shape
        Ip_dash = self.img.astype(np.int32)
        for row in range(w):
            for col in range(h):
                Ip_dash[row][col] = self.Bilateral_Filter(row, col)

        return np.clip(Ip_dash, 0, 255).astype(np.uint8)

    def Gs(self, xp, xq, yp, yq):
        return np.exp(((xp - xq) ** 2) + (yp - yq) ** 2 / (-2 * self.sigma_s**2))

    # tp tq 為 guidence 中 pq位置的強度值
    def Gr(self, Tp, Tq):
        if self.guidance.shape == 3:
            Tp_r = Tp[0]
            Tp_g = Tp[1]
            Tp_b = Tp[2]

            Tq_r = Tq[0]
            Tq_g = Tq[1]
            Tq_b = Tq[2]

            return np.exp(
                (((Tp_r - Tq_r) ** 2) + ((Tp_g - Tq_g) ** 2) + ((Tp_b - Tq_b) ** 2))
                / (-2 * self.sigma_r**2)
            )
        else:
            return np.exp(((Tp - Tq) ** 2) / (-2 * self.sigma_r**2))

    def Bilateral_Filter(self, xp, yp):
        # xp, yp 是原圖
        offset = self.wndw_size // 2
        xp_offset = xp + self.wndw_size // 2
        yp_offset = yp + self.wndw_size // 2
        numerator = 0
        denominator = 0

        # 在padding 的圖上
        for xq in range(xp, xp + self.wndw_size):
            for yq in range(yp, yp + self.wndw_size):
                Iq = self.padded_img[xq][yq]
                Tp = self.padded_guidance[xp_offset][yp_offset]
                Tq = self.padded_guidance[xq][yq]
                numerator += (
                    self.Gs(xp_offset, xq, yp_offset, yq) * self.Gr(Tp, Tq) * Iq
                )

                denominator += self.Gs(xp_offset, xq, yp_offset, yq) * self.Gr(Tp, Tq)

        if not np.any(numerator):
            output = (numerator / 1).astype(np.int32)
        else:
            output = (numerator / denominator).astype(np.int32)

        return output
