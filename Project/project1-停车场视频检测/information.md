# Project1
## Filder information
- cnn_data: cnn network input image，由[ark_test.py对with_parking.jpg进行裁剪处理完成
- test_images: 图像预处理数据
- train_images: 训练模型数据，包括train files and validation files
  - train: 训练数据 train data 381
    - empty: 空车位图像 96 
    - occupied: 非空车位图像 285
  - test: 验证数据 validation data 164
    - empty: 空车位图像 38
    - occupied: 非空车位图像 126
- others
    - park_test.py: main script
    - Parking.py: functions
    - train.py: 训练脚本
    - other data:
      - car1.h5: cnn训练后预测所需权重参数
      - parking_video.mp4: 原始视频数据
      - spot_dict.pickle
      - with_parking.jpg: 停车位检测结果

## 项目执行流程
1. 首先对原始帧进行图像处理操作，提取每个停车位特征，即是否有停车。数据存放在cnn_data folder中。执行park_test.py中image_process
2. 手动将cnn_data归纳到train_images和test_images，分类为空和非空车位作为Model训练，验证，测试数据
3. 训练train.py训练模型，保存模型权值car1.h5
4. 基于模型权值预测图像数据和视频数据
## 训练结果

  