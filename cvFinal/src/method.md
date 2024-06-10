# Brief description of models and your method

## 計算Homography
- 使用 [LightGlue](https://github.com/cvg/LightGlue) 尋找兩張照片對應的特徵點，並利用RANSAC算出homography matrix

## 生成homography
- 使用不同大小的 window 掃過 reference 和 target 的照片，計算對應 window 的 homography matrix，再利用 Kmeans 分群找出群集中心點
- 使用 [OneFormer](https://github.com/SHI-Labs/OneFormer) 對照片做 segmentation，並計算對應物件的 homography matrix
- 將 reference 0 到 target 的 inverse homography matrix 加入 reference 1 到 target 的 homography matrix，反之亦然

## 挑選Homography
- 每輪選一個 homography matrix 加入 model 中，選擇方式為加入該 homography matrix 能下降最多MSE者，總共挑選 12 輪
