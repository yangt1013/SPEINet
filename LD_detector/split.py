import os
import shutil
import random

def split_dataset(base_dir, train_ratio=0.9):
    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    train_dir = os.path.join(base_dir, 'train_split')
    test_dir = os.path.join(base_dir, 'test_split')
    
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    for cls in classes:
        cls_dir = os.path.join(base_dir, cls)
        images = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]
        
        random.shuffle(images)
        
        train_count = int(len(images) * train_ratio)
        train_images = images[:train_count]
        test_images = images[train_count:]
        
        cls_train_dir = os.path.join(train_dir, cls)
        cls_test_dir = os.path.join(test_dir, cls)
        
        if not os.path.exists(cls_train_dir):
            os.makedirs(cls_train_dir)
        if not os.path.exists(cls_test_dir):
            os.makedirs(cls_test_dir)
        
        for img in train_images:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(cls_train_dir, img))
        
        for img in test_images:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(cls_test_dir, img))

if __name__ == '__main__':
    base_directory = '/home/yangt/ssd1/dataset/GOPRO_Large_all/train'  
    split_dataset(base_directory)
