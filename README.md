


# *MoDe*
The PyTorch implementation for the MoDe: Synergizing Motion and Depth: A Distilled Dual-Branch Network for Moving Infrared Small Target Detection. 


## 1. Pre-request
### 1.1. Environment
```bash
conda create -n MoDe python=3.10.11
conda activate MoDe
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

git clone --depth=1 https://github.com/xiaomingxige/MoDe
cd MoDe
pip install -r requirements.txt
```

### 1.2. Datasets

Our experiments are conducted on two datasets: **IRDST-H**,  **ITSDT-15K**, and **DAUB-R**.
We would like to thank [MoPKL](https://github.com/UESTC-nnLab/MoPKL) for providing the datasets download links.


## 2. Pre-trained DAM2 model
The pre-trained DAM2 model can be downloaded from [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2).
## 3. Train
Taking the **DAUB_R** dataset as an example, you can use the following command:
```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u train_DAUB_R.py > nohup_DUAB_R.out & 
```

For other datasets, you can proceed in a similar manner after modifying the corresponding file paths.
## 4. Test
We utilize 1 NVIDIA GeForce RTX 4090D GPU for test. For the **DAUB_R** dataset：
```bash
CUDA_VISIBLE_DEVICES=0 python vid_DAUB_R.py  
```
## 5. Visualization
For the **DAUB_R** dataset：
```bash
python predict_DAUB_R.py
```
## Citation
If you find this project is useful for your research, please cite:
```bash
@article{LUO2026184,
title = {Synergizing motion and depth: A distilled dual-branch network for moving infrared small target detection},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {242},
pages = {184-197},
year = {2026},
author = {Dengyan Luo and Yanping Xiang and Hu Wang and Luping Ji and Song Tang and Mao Ye},
}
```



## Acknowledgements
This work is based on  [MoPKL](https://github.com/UESTC-nnLab/MoPKL) and  [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2). Thank them for sharing the codes or datasets.

