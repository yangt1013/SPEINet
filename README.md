# Video Deblurring by Sharpness Prior Detection and Edge Information
[Paper](https://arxiv.org/abs/2501.12246)    Paper is under review.
# Video visulization
<iframe src="https://drive.google.com/file/d/1d7F9RDtkRZ0NQ701mw-izlyxvNz1UZnc/preview" width="640" height="480"></iframe>

[Video Deblurring](https://drive.google.com/file/d/1d7F9RDtkRZ0NQ701mw-izlyxvNz1UZnc/view?usp=drive_link)




# Environment
Python >= 3.8, PyTorch >= 1.1.0

Requirements: opencv-python, numpy, matplotlib, imageio, scikit-image, tqdm

Platforms: Ubuntu 22.04, cuda-12.4, 3*A100(40G)

# Datasets
**GoProRS organiazation Form**
```bash
|--dataset
    |--blur  
        |--video 1
            |--frame 1
            |--frame 2
                ：  
        |--video 2
            :
        |--video n
    |--gt
        |--video 1
            |--frame 1
            |--frame 2
                ：  
        |--video 2
         :
        |--video n
    |--label
        |--video 1
        |--video 2
         :
        |--video n
```
**Download**

Please download the testing datasets and training datasets (GoProO, GoProS, GoProRS, BSD) from [BaiduYun](https://pan.baidu.com/s/1HCtfDtz35fl-ihlvhRFugQ?pwd=xtj7) password：`xtj7`

And pretrained model can be downloaded Here [Google Drive](https://drive.google.com/drive/folders/17gJkfAGVcBiLU50wBfueLrQHMj9J2Fev?dmr=1&ec=wgc-drive-globalnav-goto)

Our SPEINet model trained on random ratio GoProRS dataset can be download Here [Google Drive](https://drive.google.com/drive/folders/1AfAH4Fmj1DE0tcxOCMssCqCWuIcptkBI?dmr=1&ec=wgc-drive-globalnav-goto)

if you want only generate GoProRS dataset not include the SPEINet model, you can run
```bash
python LD_detector/choice_dataset_train.py
```
# Sharpness frame detector 
if you want generate the dataset and training sharpness frame detector, you can run 

```bash
python LD_detector/run_detector.sh
```
if you only want train and test the detector without generate dataset, you can run

```bash
python /LD_detector_gopros_train.py
```
```bash
python /test_detector.py
```
# SPEINet
## 1)Test
```bash
python inference_SPEINet.py
```
## 2)Train
```bash
python train.sh
```
# Acknowledgements
This code is built on [D2Net](https://github.com/shangwei5/D2Net?tab=readme-ov-file#prerequisites). We thank the authors for sharing the codes.
