import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter


def load_settings(setting_path):
    """Load settings from a file."""
    weights = []
    sigma_s, sigma_r = None, None
    with open(setting_path, "r") as file:
        for idx, line in enumerate(file):
            info = line.strip().split(",")
            if 0 < idx < 6:
                weights.append([float(x) for x in info])
            elif idx == 6:
                sigma_s, sigma_r = int(info[1]), float(info[3])
    return weights, sigma_s, sigma_r


def calculate_guidance_images(img_rgb, weights, guidance_images):
    """Calculate guidance images based on weights."""
    for weight in weights:
        guidance = (
            img_rgb[:, :, 0] * weight[0]
            + img_rgb[:, :, 1] * weight[1]
            + img_rgb[:, :, 2] * weight[2]
        )
        guidance_images.append(guidance)


def evaluate_costs(img_rgb, guidance_images, sigma_s, sigma_r, weights):
    """Evaluate and print the cost for each guidance image."""
    jbf = Joint_bilateral_filter(sigma_s, sigma_r)
    bf_out = jbf.joint_bilateral_filter(img_rgb, img_rgb)

    cost_list = []
    jbf_images = []
    for i, guidance_image in enumerate(guidance_images):
        gf_out = jbf.joint_bilateral_filter(img_rgb, guidance_image)
        jbf_images.append(gf_out)
        cost = np.sum(np.abs(bf_out.astype("int32") - gf_out.astype("int32")))
        cost_list.append(cost)
        if i == 0:
            print("cv2.COLOR_BGR2GRAY cost:", cost)
        else:
            print(
                f"R*{weights[i-1][0]:.1f}+G*{weights[i-1][1]:.1f}+B*{weights[i-1][2]:.1f}: {cost}"
            )

    cv2.imwrite("./gray_max.jpg", guidance_images[cost_list.index(max(cost_list))])
    cv2.imwrite("./rgb_max.jpg", cv2.cvtColor(jbf_images[cost_list.index(max(cost_list))],cv2.COLOR_RGB2BGR))
    cv2.imwrite("./gray_min.jpg", guidance_images[cost_list.index(min(cost_list))])
    cv2.imwrite("./rgb_min.jpg", cv2.cvtColor(jbf_images[cost_list.index(min(cost_list))],cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(
        description="main function of joint bilateral filter"
    )
    parser.add_argument(
        "--image_path", default="./testdata/1.png", help="path to input image"
    )
    parser.add_argument(
        "--setting_path",
        default="./testdata/1_setting.txt",
        help="path to setting file",
    )
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ### TODO ###
    guidance_images = [img_gray]
    weights, sigma_s, sigma_r = load_settings(args.setting_path)
    calculate_guidance_images(img_rgb, weights, guidance_images)
    evaluate_costs(img_rgb, guidance_images, sigma_s, sigma_r, weights)


if __name__ == "__main__":
    main()
