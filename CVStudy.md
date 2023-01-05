# 斯坦福-李飞飞课程
图像识别的问题：语义映象
图像识别的难点：视点变化、光影变化、变形、遮挡、复杂背景、组内变异（相同目标不同类别）
## 图像分类
### Nearest Neighbor Classifier
train output - test output - L1 distance metric(曼哈顿距离)判断差异
缺点：测试预测时间复杂度O(N)大于训练复杂度O(1)，但理想状态应是预测时间短语训练时间
### K-Nearest Neighbors
train output - test output - L1 distance metric(曼哈顿距离)/L2 distance metric(欧式距离)判断差异
缺点：在测试数据集上实验耗时，distance metric无意义
**超级参数选择**：取决于：数据集
将总数据集分割出一定比例的validation验证数据集可以有效验证分割比例超参
### Linear Classifier
线性分类、二分类