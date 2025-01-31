# YOLOv8 Next-Gen Object Detection using PyTorch

### _Achieve SOTA results with just 1 line of code!_
# 🚀 Demo

https://github.com/user-attachments/assets/b41621ae-ebdf-453c-b2bc-9ffe01a14d08


https://github.com/user-attachments/assets/1b5d4323-64ee-4514-95f0-6a1080672753


### ⚡ Installation (30 Seconds Setup)

```
conda create -n YOLO python=3.9
conda activate YOLO
pip install thop
pip install tqdm
pip install PyYAML
pip install opencv-python
conda install pytorch torchvision torchaudio cudatoolkit=12.2 -c pytorch-lts
```

### 🏋 Train

* Configure your dataset path in `main.py` for training
* Run `bash main.sh $ --train` for training, `$` is number of GPUs

### 🧪 Test/Validate

* Configure your dataset path in `main.py` for testing
* Run `python main.py --Validate` for validation

### 🔍 Inference (Webcam or Video)

* Run `python main.py --inference` for inference

### 📊 Performance Metrics & Pretrained Checkpoints

| Model                                                                            | mAP<sup>val<br>50-95 | mAP<sup>val<br>50 | Precision | Recall | params<br><sup>(M) | FLOPs<br><sup>@640 (B) |
|----------------------------------------------------------------------------------|----------------------|-------------------|-----------|--------|--------------------|------------------------|
| [YOLOv8n](https://github.com/Shohruh72/YOLOv8/releases/download/v1.0.0/v8_n.pt)  | 37.2                 | 52.1              | 63.3      | 47.5   | **3.2**            | **8.7**                |
| [YOLOv8s](https://github.com/Shohruh72/YOLOv8/releases/download/v1.0.0/v8_s.pt)  | 44.8                 | 61.3              | 68.2      | 56.3   | 11.2               | 28.6                   |
| [YOLOv8m](https://github.com/Shohruh72/YOLOv8/releases/download/v1.0.0/v8_m.pt)  | 50.3                 | 66.8              | 71.6      | 61.0   | 25.9               | 78.9                   |
| [YOLOv8l](https://github.com/Shohruh72/YOLOv8/releases/download/v1.0.0/v8_l.pt)  | 53.1                 | 69.5              | 74.0      | 63.4   | 43.7               | 165.2                  | 50.7                 | 68.9              | 86.7               | 205.7                  |
| [YOLOv8x](https://github.com/Shohruh72/YOLOv8/releases/download/v1.0.0/v8_x.pt)  | 54.1                 | 70.7              | 73.7      | 64.7   | 68.2               | 257.8                  | 50.7                 | 68.9              | 86.7               | 205.7                  |

### 📈 Additional Metrics
### 📂 Dataset structure

    ├── COCO 
        ├── images
            ├── train2017
                ├── 1111.jpg
                ├── 2222.jpg
            ├── val2017
                ├── 1111.jpg
                ├── 2222.jpg
        ├── labels
            ├── train2017
                ├── 1111.txt
                ├── 2222.txt
            ├── val2017
                ├── 1111.txt
                ├── 2222.txt

⭐ Star the Repo!

If you find this project helpful, give us a star ⭐ 

#### 🔗 Reference

* https://github.com/ultralytics/ultralytics
