###################################
##### DO NOT MODIFY THIS FILE #####
###################################

import argparse

from utils import read_csv, read_json

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', help='predicted csv file path', type=str, default='./output/pred.csv')
    parser.add_argument('--annos_path', help='ground truth json file path', type=str, default='../hw2_data/p2_data/val/annotations.json')
    args = parser.parse_args()

    pred_files, pred_labels = read_csv(args.csv_path)
    gt_files, gt_labels = read_json(args.annos_path)

    test_correct = 0.0
    for i, filename in enumerate(pred_files):
        if gt_labels[gt_files.index(filename)] == pred_labels[i]:
            test_correct += 1
    
    test_acc = test_correct / len(pred_files)
    print(f'Accuracy = {test_acc}\n')

if __name__ == '__main__':
    main()