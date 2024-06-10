import os, argparse
import numpy as np
import cv2
from PIL import Image
from functions import choose_model, backward_wrapping

def gen_frame(args):
	refs_0 = [ 0,  0,  0,  0,  0,  0,  2,  4,
			   4,  6,  8,  8,  8, 10, 12, 12,
			  14, 16, 16, 16, 16, 18, 20, 20,
			  22, 24, 24, 24, 26, 28, 28, 30]
	refs_1 = [ 0, 32, 16,  8,  4,  2,  4,  8,
			   6,  8, 16, 12, 10, 12, 16, 14,
			  16, 32, 24, 20, 18, 20, 24, 22,
			  24, 32, 28, 26, 28, 32, 30, 32]
	tars = [32, 16,  8,  4,  2,  1,  3,  6,
			 5,  7, 12, 10,  9, 11, 14, 13,
			15, 24, 20, 18, 17, 19, 22, 21,
			23, 28, 26, 25, 27, 30, 29, 31]
	
	for i in range(args.start_frame, args.end_frame):
		frame = i % 32
		if frame == 0:
			continue
		
		it = i // 32
		ref0_frame_num = refs_0[frame] + (32 * it)
		ref1_frame_num = refs_1[frame] + (32 * it)
		tar_frame_num = tars[frame] + (32 * it)

		print('generating frame... %3d' % tar_frame_num)
		ref0 = cv2.imread(os.path.join(args.gt_path, '%03d.png' % ref0_frame_num))
		ref1 = cv2.imread(os.path.join(args.gt_path, '%03d.png' % ref1_frame_num))
		target = cv2.imread(os.path.join(args.gt_path, '%03d.png' % tar_frame_num))
		
		H_seg_big_L = np.load(os.path.join(args.homo_path, '%03d_L.npy' % tar_frame_num))
		H_seg_big_R = np.load(os.path.join(args.homo_path, '%03d_R.npy' % tar_frame_num))
		
  		# H1
		H1_list = H_seg_big_L
		H1_list_inv = np.linalg.inv(H1_list)

		# H2
		H2_list = H_seg_big_R
		H2_list_inv = np.linalg.inv(H2_list)

		# inv
		H1_list = np.concatenate((H1_list, H2_list_inv), axis=0)
		H2_list = np.concatenate((H2_list, H1_list_inv), axis=0)
		

		H1_result, H2_result= choose_model(ref0, ref1, target, H1_list, H2_list)
		target, model_map, _ = backward_wrapping(ref0, ref1, target, H1_result, H2_result)

		target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
		cv2.imwrite(os.path.join(so_path, '%03d.png' % tar_frame_num),target)
		
		# model map
		with open(os.path.join(so_path, 'm_%03d.txt' % tar_frame_num), 'w') as f:
			for row in model_map:
				for element in row:
					f.write(f"{element}\n")
				
		# model
		np.save(os.path.join(so_path, 'H1_%03d.npy' % tar_frame_num), H1_result)
		np.save(os.path.join(so_path, 'H2_%03d.npy' % tar_frame_num), H2_result)

def compute_psnr(so_path, gt_path):
	image_name = ['%03d.png'% i for i in range(args.start_frame, args.end_frame) if i not in [0, 32, 64, 96, 128]]
	txt_name   = ['s_%03d.txt'% i for i in range(args.start_frame, args.end_frame) if i not in [0, 32, 64, 96, 128]]

	so_img_paths = [os.path.join(so_path, name) for name in image_name]
	so_txt_paths = [os.path.join(so_path, name) for name in txt_name]
	gt_img_paths = [os.path.join(gt_path, name) for name in image_name]

	for so_img_path, so_txt_path, gt_img_path in zip(so_img_paths, so_txt_paths, gt_img_paths):
		
		try:
			s = np.array(Image.open(so_img_path).convert('L'))
			g = np.array(Image.open(gt_img_path).convert('L'))
		except Exception as e:
			print(f"No {so_img_path}")
			continue

		s = s.reshape(2160//16, 16, 3840//16, 16).swapaxes(1, 2).reshape(-1, 16, 16).astype(int)
		g = g.reshape(2160//16, 16, 3840//16, 16).swapaxes(1, 2).reshape(-1, 16, 16).astype(int)

		mse = np.sum(((s - g) ** 2) / (16 * 16), axis=(1, 2))
		mse[mse == 0] = 6.5025e-06
		psnr = (10 * np.log10(255 ** 2 / mse))

		selected_psnr = np.argsort(psnr)[::-1][:13000]
		mask = np.zeros(32400, dtype=np.uint8)
		mask[selected_psnr] = 1			

		with open(so_txt_path, 'w') as f:
			f.write('\n'.join(map(str, mask)))
 
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--so_path', type=str, default="solution/")
	parser.add_argument('-g', '--gt_path', type=str, default="gt/")
	parser.add_argument('-st', '--start_frame', type=int, default=0)
	parser.add_argument('-end', '--end_frame', type=int, default=128)
	parser.add_argument('-ho','--homo_path', type=str, default="homography/")
	args = parser.parse_args()

	so_path = args.so_path
	gt_path = args.gt_path
	
	gen_frame(args)

	compute_psnr(so_path, gt_path)
	