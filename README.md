# Video Deblurring by Sharpness Prior Detection and Edge Information
[PDF](https://arxiv.org/abs/2501.12246)    Paper is under review.

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
    |--Event
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
weights and data will be available soon.
