# 斯坦福-李飞飞课程
图像识别的问题：语义映象
图像识别的难点：视点变化、光影变化、变形、遮挡、复杂背景、组内变异（相同目标不同类别）
## 图像分类
### 挑战
由于识别视觉概念（例如猫）的任务对于人类来说相对微不足道，因此值得从计算机视觉算法的角度考虑所涉及的挑战。当我们在下面提出（不详尽的）挑战列表时，请记住图像的原始表示为亮度值的 3-D 数组：
1. 视点变化：对象的单个实例可以相对于相机以多种方式定向。
2. 规模变化：视觉类通常表现出其大小的变化（现实世界中的大小，而不仅仅是它们在图像中的范围）。
3. 变形：许多感兴趣的物体不是刚体，可以以极端方式变形。
4. 遮挡：感兴趣的对象可能会被遮挡。有时，只有对象的一小部分（只有几个像素）是可见的。
5. 照明条件：照明在像素级别上的影响是巨大的。
6. 背景混乱：感兴趣的对象可能会融入其环境，使其难以识别。
7. 类内差异：感兴趣的类别通常可以相对广泛，例如椅子。这些对象有许多不同类型的，每个对象都有自己的外观。
###  图像分类管道
1. 输入：我们的输入由一组N个图像组成，每个图像都标有K个不同的类中的一个。我们将这些数据称为训练集。
2. 学习：我们的任务是使用训练集来学习每个类的样子。我们将此步骤称为训练分类器或学习模型。
3. 评估：最后，我们通过要求分类器预测一组以前从未见过的新图像的标签来评估分类器的质量。然后，我们将这些图像的真实标签与分类器预测的标签进行比较。直觉上，我们希望很多预测与真实答案（我们称之为基本事实）相匹配。
### Nearest Neighbor Classifier
train output - test output - L1 distance metric(曼哈顿距离)判断差异
缺点：测试预测时间复杂度O(N)大于训练复杂度O(1)，但理想状态应是预测时间短语训练时间
### K-Nearest Neighbors
train output - test output - L1 distance metric(曼哈顿距离)/L2 distance metric(欧式距离)判断差异
缺点：在测试数据集上实验耗时，distance metric无意义
分类器必须记住所有训练数据，并将其存储起来，以便将来与测试数据进行比较。这是空间效率低下的，因为数据集的大小可能很容易达到千兆字节。
对测试图像进行分类是昂贵的，因为它需要与所有训练图像进行比较。
**超级参数选择**：取决于：数据集
将总数据集分割出一定比例的validation验证数据集可以有效验证分割比例超参
## Linear Classifier 线性分类
### 线性分类器
线性分类器将类的分数计算为其所有3个颜色通道中所有像素值的加权和。根据我们为这些权重设置的精确值，该函数能够在图像中的某些位置喜欢或不喜欢（取决于每个权重的符号）某些颜色。例如，您可以想象，如果图像两侧有很多蓝色（可能对应于水），则“船”类的可能性更大。您可能会期望“船舶”分类器在其蓝色通道权重中具有大量正权重（蓝色的存在会增加船舶的分数），而红色/绿色通道中的负权重（红色/绿色的存在会降低船舶的分数）。
### 数据预处理
所有图像都是使用的原始像素值（从0到255）。在机器学习中，对于输入的特征做**归一化（normalization）**处理是常见的套路。而在图像分类的例子中，图像上的每个像素可以看做一个特征。在实践中，对每个特征减去平均值来中心化数据是非常重要的。在这些图片的例子中，该步骤意味着根据训练集中所有的图像计算出一个平均图像值，然后每个图像都减去这个平均值，这样图像的像素值就大约分布在[-127, 127]之间了。下一个常见步骤是，让所有数值分布的区间变为[-1, 1]。零均值的中心化是很重要的，等我们理解了梯度下降后再来详细解释。
### Loss function 损失函数
### Softmax分类器
### 最优化
## 神经网络
## 卷积神经网络
## 练习1 Assignment1
# GNN
python version 3.7.6
torch version 1.8.1+cpu
## 0-python工具包安装 - 不能直接使用pip install
pip wheels 装依赖
直接 pip install torch-geometric
## 1-pytorch_geometric基本使用
GNN(Graph Neural Networks)致力于解决不规则数据结果，其迭代更新主要基于图中每个节点及其邻居的信息
```python
# 画图
%matplotlib inline
import torch
import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(G, color):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False, node_color=color, cmap="Set2")
    plt.show()
    
def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()
```
数据集 - KarateClub Dataset - States: nodes-34, edges-156, features-34, classes-4
```python
# 导入数据
from torch_geometric.datasets import KarateClub

datasets = KarateClub()
print(f'Dataset:{datasets}:')
print('=====================')
print(f'Number of graphs: {len(datasets)}')
print(f'Number of features: {datasets.num_features}')
print(f'Number of classes: {datasets.num_classes}')

data = datasets[0] # Get the first graph object
print(data)
```
edge_index
edge_index: 表示图的连结关系（start, end两个序列） node features: 每个点特征 node labels: 每个点的标签 train_mask: 点标签，有的node无标签（用来表示哪些节点要计算损失）
```python
edge_index = data.edge_index
print(edge_index.t())

# 节点可视化
from torch_geometric.utils import to_networkx

G = to_networkx(data, to_undirected=True)
visualize_graph(G, color=data.y)
```
### GNN构建
```python
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        # 三层卷积
        self.conv1 = GCNConv(datasets.num_features, 4) # 只需要定义输入特征和输出特征即可
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, datasets.num_classes) # 全连接
        
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index) # 输入特征与邻接矩阵（注意格式）
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h , edge_index)
        h = h.tanh()
        
        # 分类层
        out = self.classifier(h)
        
        return out, h

model = GCN()
print(model) # 打印网络模型

model = GCN()

_, h = model(data.x, data.edge_index)
print(f'Embedding shape: {list(h.shape)}')

visualize_embedding(h, color=data.y) # 可视化
```
#### 训练
```python
# 训练模型(semi-supervised)
import time

model = GCN()
criterion = torch.nn.CrossEntropyLoss() # Define Loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # Define optimizer

def train(data):
    optimizer.zero_grad()
    out, h = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask]) # semi-supervised：只关注有标签训练节点
    loss.backward() # 反向传播
    optimizer.step()
    return loss, h

for epoch in range(401):
    loss, h = train(data)
    if epoch % 10 == 0:
        visualize_embedding(h, color=data.y, epoch=epoch, loss=loss)
        time.sleep(0.3)


# 可视化out
# 训练模型(semi-supervised)
import time

model = GCN()
criterion = torch.nn.CrossEntropyLoss() # Define Loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # Define optimizer

def train(data):
    optimizer.zero_grad()
    out, h = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask]) # semi-supervised：只关注有标签训练节点
    loss.backward() # 反向传播
    optimizer.step()
    return out

for epoch in range(401):
    out = train(data)
#     if epoch % 10 == 0:
#         visualize_embedding(out, color=data.y, epoch=epoch, loss=loss)
#         time.sleep(0.3)
# #     if epoch % 100 == 0:
# #         visualize_embedding(out, color=data.y, epoch=epoch, loss=loss)
# #         time.sleep(0.3)
visualize_embedding(out, color=data.y, epoch=epoch, loss=loss)
```
## 2-点分类任务
### 可视化方法
```python
# 可视化
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])
    
    plt.scatter(z[:, 0], z[:, 1], s=70,  c=color, cmap="Set2")
    plt.show()
```
### Dataset - Cora
- 该数据集是论文引用数据集，每个点有1433维向量；
- 最终要对每个点进行7分类任务(每个类别只有20个点有标记)
``` python
# 数据
from torch_geometric.datasets import Planetoid # 用于下载数据到本地
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures()) #transform预处理

#打印数据信息
print()
print(f'Dataset: {dataset}: ')
print('=====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0] # Get the first graph object

print()
print(data)
print('===============================')

# Gether some statistics about the graph
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')
```
### 对比实验
使用torch的全连接层Linear和GNN的GCN模型进行对比实验
#### 全连接层 Multi-layer Perception Network
``` python
# 构建模型
import torch
from torch.nn import Linear
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(dataset.num_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, dataset.num_classes)
        
    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

model = MLP(hidden_channels=16)
print(model) # 输出模型结构

# 训练
# 训练 - 注：只考虑有标签的点
model = MLP(hidden_channels=16)
criterion = torch.nn.CrossEntropyLoss() # Define Loss Function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4) # Define Optimizer

def train():
    model.train()
    optimizer.zero_grad() # 梯度初始化清零Clear gradients
    out = model(data.x) # 执行单次向前Perform a single forward pass
    # 注：只考虑有标签的点
    loss = criterion(out[data.train_mask], data.y[data.train_mask]) # 仅基于训练节点计算损失Compute the loss solely based on the training nodes
    loss.backward() # 导出梯度Derive gradients
    optimizer.step() # 基于梯度更新参数Update parameters based on gradients
    return loss

def test():
    model.eval()
    out = model(data.x)
    pred = out.argmax(dim=1) # 使用概率最高的类Use the class with highest probability
    test_correct = pred[data.test_mask] == data.y[data.test_mask] # 对照实际情况标签进行检查Check against ground-truth labels
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum()) # 导出正确预测比Derive ratio of correct predictions
    return test_acc

for epoch in range(1, 201):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

# 准确率计算 7分类
test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')
```
#### Graph Neural Network(GNN)
```python
# 构建模型
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GCN(hidden_channels=16)
print(model) # 输出模型结构

# 可视化时由于输出7维向量，因此需要降维进行展示
model = GCN(hidden_channels=16)
model.eval()

out = model(data.x, data.edge_index)
visualize(out, color=data.y)

# 训练
# 训练GCN模型
# 训练 - 注：只考虑有标签的点
model = GCN(hidden_channels=16)
criterion = torch.nn.CrossEntropyLoss() # Define Loss Function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4) # Define Optimizer

def train():
    model.train()
    optimizer.zero_grad() # 梯度初始化清零Clear gradients
    out = model(data.x, data.edge_index) # 执行单次向前Perform a single forward pass
    # 注：只考虑有标签的点
    loss = criterion(out[data.train_mask], data.y[data.train_mask]) # 仅基于训练节点计算损失Compute the loss solely based on the training nodes
    loss.backward() # 导出梯度Derive gradients
    optimizer.step() # 基于梯度更新参数Update parameters based on gradients
    return loss

def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1) # 使用概率最高的类Use the class with highest probability
    test_correct = pred[data.test_mask] == data.y[data.test_mask] # 对照实际情况标签进行检查Check against ground-truth labels
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum()) # 导出正确预测比Derive ratio of correct predictions
    return test_acc

for epoch in range(1, 101):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

# 准确率计算
test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')

# 可视化效果
model.eval()

out = model(data.x, data.edge_index)
visualize(out, color=data.y)
```
# OpenCV
## OpenCV with python 3h Learning
Github Link: https://github.com/murtazahassan/Learn-OpenCV-in-3-hours
B站：【3h精通Opencv-Python】https://www.bilibili.com/video/BV16K411W7x9?vd_source=cf518f0e157700ce8a169afae9bf19ea
### Learning information
#### 1-读取数据
```python
# 读取展示图片
import cv2
print('Package Imported')

dataDir = 'Data/whdTest01.jpg'
img = cv2.imread(dataDir)
height, width = img.shape[0:2]
img = cv2.resize(img, (int(width / 10), int(height / 10))) # 等比缩小

cv2.imshow('Output', img)
cv2.waitKey(0)

# 保存图像
cv.imwrite('Data/whdTest02.jpg', img)

# 读取视频
import cv2
print('Package Imported')

dataDir = 'Data/videoTest01.mp4'
cap = cv2.VideoCapture(dataDir)
while True:
    success, img = cap.read()
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows() # 关闭所有窗口
        break

# 摄像机
import cv2 as cv
print('Package Imported')

cap = cv.VideoCapture(0)
cap.set(3, 640) # width
cap.set(4, 480) # height
cap.set(10, 100) # brighthness
# cap.set(cv.CV_CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 640)
# cap.set(cv.CAP_PROP_BRIGHTNESS, 500)

while True:
    success, img = cap.read()
    cv.imshow("Video", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv.destroyAllWindows()
        break
```
#### 2-图像基本处理
##### 灰度图像、模糊图像、图像边缘、图像膨胀、图像腐蚀functions
```python
# 灰度图像、模糊图像、图像边缘、图像膨胀、图像腐蚀functions
import cv2 as cv
import numpy as np
print('Package Imported')

dataDir = "Data/whdTest01.jpg"
kernel = np.ones((2,2), np.uint8)

img = cv.imread(dataDir)
height, width = img.shape[0:2]
img = cv2.resize(img, (int(width / 10), int(height / 10)))
# Basic image processing functions
imgGray = cv.cvtColor(img, cv2.COLOR_BGR2GRAY)# 灰度图像 Gray Image
imgBlur = cv.GaussianBlur(imgGray, (7,7), 0) # 模糊图像 Blur Image
imgCanny = cv.Canny(img,100,200) # 边缘图像 Canny Image
imgDilation = cv.dilate(imgCanny, kernel, iterations=1) # 膨胀处理 Dilation Image 扩张
imgEroded = cv.erode(imgDilation, kernel, iterations=10) # 腐蚀处理 Eroded Image

# cv.imshow("Original Image", img)
cv.imshow("Gray Image", imgGray)
cv.imshow("Blur Image", imgBlur)
cv.imshow("Canny Image", imgCanny)
cv.imshow("Dilation Image", imgDilation)
cv.imshow("Eroded Image", imgEroded)

cv.waitKey(0)
```
#### 3-图像大小及裁剪
##### 重置图像大小、裁剪图片
```python
import cv2 as cv
print("Package Imported")

dataDir = "Data/cljTest02.jpg"
imgOri = cv.imread(dataDir)
print(imgOri.shape) # 输出：height:750, width:474, channel: 3
height, width = imgOri.shape[0:2]
print(f'Orignal Image height: {height}, width: {width}')

imgResize = cv.resize(imgOri, (int(width/2), int(height/2))) # (width, height)
print(imgResize.shape)
reHeight, reWidth = imgResize.shape[0:2]
print(f'Resize Image height: {reHeight}, width: {reWidth}')

# 裁剪 - 注：y从上到下，箭头朝下，x从左到右，箭头朝右
imgCropped = imgOri[100:300, 0:200] # [y, x]: height, width 

cv.imshow("Original Image", imgOri)
# cv.imshow("Resize Image", imgResize)
cv.imshow("Cropped Image", imgCropped)

cv.waitKey(0)
```
#### 4-画图
##### 画出矩形、线条、圆形、文本
```python
# 画出矩形、线条、圆形、文本
import cv2 as cv
import numpy as np
print("Package Imported")

# 0-black, channel=3时才能画出彩色图像
imgCanvas = np.zeros((512,512,3), np.uint8) # 黑色底色画布 Canvas
# print(imgCanvas)
imgCanvas[100:200,200:400] = 255,255,0 # 0,0,0-black, 255,0,0-Blue, 0,255,0-Green, 0,0,255-red, 255,255,255-white
# line
cv.line(imgCanvas, (0,0), (300,300), (0,0,255), 3)
cv.line(imgCanvas, (0,0), (imgCanvas.shape[1], imgCanvas.shape[0]), (255,0,0), 1)
# rectangle 矩形
cv.rectangle(imgCanvas, (0,0), (250, 300), (0,255,0), 3)
cv.rectangle(imgCanvas, (0,0), (250, 300), (200,10,0), cv.FILLED)
# circle 圆形
cv.circle(imgCanvas, (400, 50), 30, (255,255,255), 3) # 30半径
# cv.circle(imgCanvas, (400, 50), 30, (255,255,255), cv.FILLED) # 30半径
# Text 文本
cv.putText(imgCanvas, "OpenCV", (300,400), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,100), 3)

cv.imshow("Canvas Image", imgCanvas)

cv.waitKey(0)
```
#### 5-透视
##### warp prespective
翘曲透视。对图像进行透视变换。简单来说，就是有这么一副图像，它的拍摄视角不是从正面拍摄的，而是带有一定的角度，我们希望能得到从正面观察的视角
```python
import cv2 as cv
import numpy as np
print('Package Imported')

dataDir = "Data/cards.jpg"
imgOri = cv.imread(dataDir)
width, height = 250, 350 # 所需图像大小

# pts1 = np.float32([[111,219],[287,188],[154,482],[352,440]])
# pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
# matrix = cv.getPerspectiveTransform(pts1,pts2)
# result = cv.warpPerspective(imgOri, matrix, (width,height))

# 找k
pts1K = np.float32([[527,144],[772,192],[404,396],[677,457]])  #所需图像部分四个顶点的像素点坐标
pts2K = np.float32([[0,0],[width,0],[0,height],[width,height]]) #定义对应的像素点坐标
matrixK = cv.getPerspectiveTransform(pts1K, pts2K)  #使用getPerspectiveTransform()得到转换矩阵
imgK = cv.warpPerspective(imgOri, matrixK, (width,height))  #使用warpPerspective()进行透视变换

#找Q
pts1_Q = np.float32([[63,325],[340,279],[89,634],[403,573]])
pts2_Q = np.float32([[0,0],[width,0],[0,height],[width,height]])
matrixQ = cv.getPerspectiveTransform(pts1_Q,pts2_Q)
imgQ = cv.warpPerspective(imgOri,matrixQ,(width,height))

#找J
pts1_J = np.float32([[777,107],[1019,84],[842,359],[1117,332]])
pts2_J = np.float32([[0,0],[width,0],[0,height],[width,height]])
matrixJ = cv.getPerspectiveTransform(pts1_J,pts2_J)
imgJ = cv.warpPerspective(imgOri,matrixJ,(width,height))

# cv.imshow("Original Image",imgOri)
# cv.imshow("Result Image", result)
cv.imshow("img K",imgK)
cv.imshow("img Q",imgQ)
cv.imshow("img J",imgJ)

cv.waitKey(0)
```
#### 6-Joining Image 连接图像
```python
# 拼接图像在同一个窗口展示
import cv2 as cv
import numpy as np
print('Package Imported')

horDataDir = 'Data/cljTest02.jpg'
verDataDir = 'Data/whdTest02.jpg'

imgForHor = cv.imread(horDataDir)
imgForVar = cv.imread(verDataDir)

imgHor = np.hstack((imgForHor, imgForHor)) # Horizontal 水平拼接
imgVer = np.vstack((imgForVar, imgForVar)) # Vertical 垂直拼接

# cv.imshow("imgForHor", imgForHor)
# cv.imshow("imgForVer", imgForVar)
cv.imshow("Horizontal", imgHor)
cv.imshow("Vertical", imgVer)

cv.waitKey(0)
```
#### 堆叠图片 Stack Image Function
``` python
# 堆叠函数
import cv2 as cv
import numpy as np
print('Package Imported')

def stackImages(scale,imgArray):
    # & 输出一个 rows * cols 的矩阵（imgArray）
    rows = len(imgArray)
    cols = len(imgArray[0])
    print(rows,cols)
    # & 判断imgArray[0] 是不是一个list
    rowsAvailable = isinstance(imgArray[0], list)
    # & imgArray[0][0]就是指[0,0]的那个图片（我们把图片集分为二维矩阵，第一行、第一列的那个就是第一个图片）
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range (0, rows):
            for y in range(0, cols):
                # & 判断图像与后面那个图像的形状是否一致，若一致则进行等比例放缩；否则，先resize为一致，后进行放缩
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                # & 如果是灰度图，则变成RGB图像（为了弄成一样的图像）
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv.cvtColor( imgArray[x][y], cv.COLOR_GRAY2BGR)
        # & 设置零矩阵
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    # & 如果不是一组照片，则仅仅进行放缩 or 灰度转化为RGB
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

stackDataDir = 'Data/whdTest02.jpg'
imgForStack = cv.imread(stackDataDir)
imgGray = cv.cvtColor(imgForStack, cv.COLOR_BGR2GRAY)
stackImage = stackImages(0.5, ([imgForStack, imgForStack, imgGray],[imgForStack, imgForStack, imgForStack]))

cv.imshow('Stack Image', stackImage)

cv.waitKey(0)
```
#### 7-Color Detecion 颜色检测
##### Trackbar
滑动条（Trackbar）是一种可以动态调节参数的工具，它依附于窗口而存在。 namedWindow()函数的作用是通过指定的名字，创建一个可以作为图像和进度条的容器窗口。
```python
import cv2 as cv
print('Package Imported')
dataDir = 'Data/whdTest02.jpg'
# TrackBar
def empty(a):
    pass
cv.namedWindow("TrackBars") # create new window
cv.resizeWindow("TrackBars", 640, 240)
# 色相/饱和度/明度（Hue, Saturation, Value）
# Hue 色相
cv.createTrackbar("Hue Min", "TrackBars", 0, 179, empty) 
cv.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
# Saturation 饱和度
cv.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
cv.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
# Value 明度
cv.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
cv.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

imgOri = cv.imread(dataDir)
imgHSV = cv.cvtColor(imgOri, cv.COLOR_BGR2HSV)

cv.imshow('Original Image', imgOri)
cv.imshow('HSV Image', imgHSV)

cv.waitKey(0)

import cv2 as cv
import numpy as np
print('Package Imported')
dataDir = 'Data/whdTest02.jpg'
# TrackBar
def empty(a):
    pass
cv.namedWindow("TrackBars") # create new window
cv.resizeWindow("TrackBars", 640, 240)
# 色相/饱和度/明度（Hue, Saturation, Value）
# Hue 色相
cv.createTrackbar("Hue Min", "TrackBars", 14, 179, empty)  # 0
cv.createTrackbar("Hue Max", "TrackBars", 143, 179, empty) # 179
# Saturation 饱和度
cv.createTrackbar("Sat Min", "TrackBars", 128, 255, empty) # 0  
cv.createTrackbar("Sat Max", "TrackBars", 200, 255, empty) # 255
# Value 明度
cv.createTrackbar("Val Min", "TrackBars", 55, 255, empty) # 0
cv.createTrackbar("Val Max", "TrackBars", 255, 255, empty) # 255

while True:
    imgOri = cv.imread(dataDir)
    imgHSV = cv.cvtColor(imgOri, cv.COLOR_BGR2HSV)
    h_min = cv.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv.getTrackbarPos("Val Max", "TrackBars")
    print(h_min, h_max, s_min, s_max, v_min, v_max)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv.inRange(imgHSV, lower, upper) # 变化蒙版
    
    imgResult = cv.bitwise_and(imgOri, imgOri, mask=mask)
    imgStack = stackImages(0.5, ([imgOri, imgHSV],[mask, imgResult]))
    

#     cv.imshow('Original Image', imgOri)
#     cv.imshow('HSV Image', imgHSV)
#     cv.imshow("Mask", mask)
#     cv.imshow("Result Image", imgResult)
    cv.imshow("Result Image", imgStack)
    
    cv.waitKey(1)
```
#### 8-Contours/ Shape Detection 轮廓\形状 检测
⭐⭐**getContours(img) Function 获取轮廓函数 重要**
```python
# 获取轮廓函数 getContours(img) Function
import cv2 as cv
print("Package Imported")

def getContours(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
#         print(area) # 打印区域面积
#         cv.drawContours(imgContour, cnt, -1, (255,0,0),3)
        if area > 500:
            cv.drawContours(imgContour, cnt, -1, (255,0,0),3)
            peri = cv.arcLength(cnt,True) # 计算封闭轮廓的周长或曲线的长度
#             print(peri)
            approx = cv.approxPolyDP(cnt, 0.02*peri, True) # 对图像轮廓点进行多边形拟合
#             print(approx)
#             print(len(approx)) # 判断3，4边形
            objCor = len(approx)
            x, y, w, h = cv.boundingRect(approx) # 形状边框
            if objCor == 3:
                objectType = "Triangle" # 三角形
            elif objCor == 4:
                aspRatio = w / float(h) # Aspect ratio 纵横比
                if aspRatio > 0.95 and aspRatio < 1.05:
                    objectType = "Square" # 正方形
                else:
                    objectType = "Rectangle"
            elif objCor > 4: 
                objectType = "Circles"
            else:
                objectType = "None"
            cv.rectangle(imgContour, (x,y), (x+w, y+h), (0,0,255), 2)
            
            cv.putText(imgContour, objectType, (x+(w//2)-10, y+(h//2)-10), cv.FONT_HERSHEY_COMPLEX, 0.7, (0,0,0), 2)

import cv2 as cv
import numpy as np
print("Package Imported")

dataDir = 'Data/shapes.png'
imgOri = cv.imread(dataDir)
imgContour = imgOri.copy()
imgGray = cv.cvtColor(imgOri, cv.COLOR_BGR2GRAY)
imgBlur = cv.GaussianBlur(imgGray, (7,7), 1)
imgCanny = cv.Canny(imgOri, 50, 50) 
imgBlank = np.zeros_like(imgOri)
getContours(imgCanny)
imgStack = stackImages(0.8, ([imgOri, imgGray, imgBlur],[imgCanny, imgContour, imgBlank]))

# cv.imshow("Original Image", imgOri)
# cv.imshow("Gray Image", imgGray)
# cv.imshow("Blur Image", imgBlur)
# cv.imshow("Canny Image", imgCanny)
cv.imshow("Stack Image Result", imgStack)

cv.waitKey(0)       
```
#### 9-Face Detection  ⭐⭐⭐⭐
CascadeClassifier(): 级联分类器 detectMultiScale(): 检测出图片中所有的人脸，并将人脸用vector保存各个人脸的坐标、大小（用矩形表示）
参数1：image -- 待检测图片，一般为灰度图像加快检测速度；
参数2：objects -- 被检测物体的矩形框向量组；
参数3：scaleFactor - 表示在前后两次相继的扫描中，搜索窗口的比例系数。默认为1.1即每次搜索窗口依次扩大10%;
参数4：minNeighbors 表示构成检测目标的相邻矩形的最小个数(默认为3个)。
  如果组成检测目标的小矩形的个数和小于 min_neighbors - 1 都会被排除。
  如果min_neighbors 为 0, 则函数不做任何操作就返回所有的被检候选矩形框，
  这种设定值一般用在用户自定义对检测结果的组合程序上；
参数5：flags 要么使用默认值，要么使用CV_HAAR_DO_CANNY_PRUNING，如果设置为CV_HAAR_DO_CANNY_PRUNING，那么函数将会使用Canny边缘检测来排除边缘过多或过少的区域，因此这些区域通常不会是人脸所在区域；
参数6、7：minSize和maxSize用来限制得到的目标区域的范围。
```python
# 图片人脸检测-原始code
import cv2 as cv
import numpy as np
print("Package Imported")

imageDir = 'Data/lena.png'
# "C:\\Users\10959\anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
# 注：使用库中的分类器文件
classifierDir = r"C:\\Users\10959\anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"

faceCascade= cv.CascadeClassifier(classifierDir) # 级联分类器
imgOri = cv.imread(imageDir)
imgGray = cv.cvtColor(imgOri, cv.COLOR_BGR2GRAY)

# faces = faceCascade.detectMultiScale(imgGray,1.1,4) 
faces = faceCascade.detectMultiScale(imgGray)
# 画出识别框
for (x, y, w, h) in faces:
#     print(x, y, w, h)
    cv.rectangle(imgOri, (x, y), (x+w, y+h), (255,0,0), 2)
    
# cv.imshow("Original Image", imgOri)
cv.imshow("Gray Image", imgGray)
cv.imshow("Result", imgOri)

cv.waitKey(0)
```
```python
# face detection函数
import cv2 as cv
print("Package Imported")

def catchFaces(ClassifierDir, imgOri):
    imgGray = cv.cvtColor(imgOri, cv.COLOR_BGR2GRAY)
    faceCascade = cv.CascadeClassifier(classifierDir)
    faces = faceCascade.detectMultiScale(imgGray)
    for (x, y, w, h) in faces:
        cv.rectangle(imgOri, (x, y), (x+w, y+h), (255,0,0), 2)
    cv.imshow("catachFaces Func Result",imgOri)
    cv.waitKey(0)

import cv2 as cv
print("Package Imported")

dataDir = 'Data/lena.png'
classifierDir = r"C:\\Users\10959\anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
imgOri = cv.imread(dataDir)
# cv.imshow("Orignal Image", imgOri)
catchFaces(classifierDir, imgOri)
# cv.waitKey(0)

# 测试其他图像
import cv2 as cv
print("Package Imported")
dataDir = 'Data/whdTest02.jpg'
imgTest01 = cv.imread(dataDir)
classifierDir = r"C:\\Users\10959\anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
# cv.imshow("Original Image", imgTest01)
# cv.waitKey(0)
catchFaces(classifierDir, imgTest01)

import cv2 as cv
print("Package Imported")
dataDir = 'Data/cljTest01.jpg'
imgTest02 = cv.imread(dataDir)
classifierDir = r"C:\\Users\10959\anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
# cv.imshow("Original Image", imgTest02)
# cv.waitKey(0)
catchFaces(classifierDir, imgTest02)
```
#### 摄像头实时检测人脸、眼睛、微笑
```python
# 摄像头实时人脸、眼睛、微笑检测
import cv2 as cv
print("Package Imported")

# 数据地址
faceDir = r"C:\\Users\10959\anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
eyeDir = r"C:\\Users\10959\anaconda3\Lib\site-packages\cv2\data\haarcascade_eye.xml"
smileDir = r"C:\\Users\10959\anaconda3\Lib\site-packages\cv2\data\haarcascade_smile.xml"

# 级联
faceCascade = cv.CascadeClassifier(faceDir)
eyeCascade = cv.CascadeClassifier(eyeDir)
smileCascade = cv.CascadeClassifier(smileDir)

# 打开摄像头
cap = cv.VideoCapture(0)

while True:
    # 读取帧画面
    ret, img = cap.read()
    # 灰度处理
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 人脸检测
    face = faceCascade.detectMultiScale(imgGray, 1.1, 3, 0, (120, 120))
    
    for (x, y, w, h) in face:
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 3)
        faceArea =  img[y:y+h, x:x+w]
        # 用人眼级联分类器引擎在人脸区域进行人眼识别，返回的eyes为眼睛坐标列表
        eyes = eyeCascade.detectMultiScale(faceArea, 1.3, 10)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(faceArea, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 1)
        # 用人眼级联分类器引擎在人脸区域进行人眼识别，返回的eyes为眼睛坐标列表
        smile = smileCascade.detectMultiScale(faceArea, scaleFactor=1.16, minNeighbors=50, minSize=(50, 50),flags=cv.CASCADE_SCALE_IMAGE)
        for (sx, sy, sw, sh) in smile:
            cv.rectangle(faceArea, (sx, sy), (sx+sw, sy+sh), (255, 0, 0), 1)
            cv.putText(img, 'Smile', (x, y - 7), 3, 1.2, (0, 0, 255), 2, cv.LINE_AA)

    # 展示结果
    cv.imshow("Detection Result", img)
    
    if cv.waitKey(5) & 0xFF == ord("q"):
        # 释放资源
        cap.release()
        # 销毁窗口
        cv.destroyAllWindows()
        break
```
#### Project1-Virtual Paint 虚拟画图
##### Original Code
```python
# 主流程
# 基于视频文件或摄像实时提取
import cv2
import numpy as np
import pytesseract
from PIL import Image
print("Package Imported")

#######################
imgWidth = 480
imgHeight = 640
count = 0
cap = cv2.VideoCapture(1)
cap.set(3, 480)
cap.set(4, 640)
cap.set(10, 150)
#######################

while True:
    success, img = cap.read()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 灰度化图像
    imgCopyForContour = img.copy()
    imgCopyForBiggest = img.copy()
    imgThres = preProcessing(img) # 对原图像预处理
    drawThresInImage(imgThres, imgCopyForContour) # Thres Contour
    biggestPoints = getBiggest(imgThres, imgCopyForBiggest) # 获取文本轮廓角点 (4, 1, 2) 4个1行2列的数组
    
    # 判断是否有角点，即是否有文本出现在检测框中
    if biggestPoints.size != 0:
        imgWarped = getWarp(img, biggestPoints) # 鸟瞰转换 透视转换
        warpGray = cv2.cvtColor(imgWarped, cv2.COLOR_BGR2GRAY)
        result = cv2.adaptiveThreshold(warpGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 3) # 自适应阈值结果
        resultImageArray = [[img, imgGray, imgThres, imgCopyForContour], [imgCopyForBiggest, imgWarped, warpGray, result]]
    else:
        resultImageArray = [[img, imgThres], [img, img]]
        
    # 图片堆叠显示
    resultImage = stackImages(0.5, resultImageArray)
    cv2.imshow("Result", resultImage)
        
    if cv2.waitKey(1) & 0xFF == ord("s"):
        saveImageToText(result, count)
        cv2.rectangle(img, (0, 200), (640, 300), (0,255,0), cv2.FILLED)
        cv2.putText(img, "Scan Saved", (150, 265), cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,255), 2)
        cv2.imshow("Scan State", img)
        cv2.waitKey(10)
        count += 1
        continue

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        break
        
cap.release()
cv2.destroyAllWindows()
```
##### Improved Code
#### Project2-Document Scanner 文件扫描
项目要求：对图片文件、视频数据、实时摄像数据进行文件提取和文件内容扫描
##### 原代码
```python
import cv2
import numpy as np


###################################
widthImg=540
heightImg =640
#####################################

cap = cv2.VideoCapture(1)
cap.set(10,150)


def preProcessing(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,200,200)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=2)
    imgThres = cv2.erode(imgDial,kernel,iterations=1)
    return imgThres

def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>5000:
            #cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            if area >maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
    return biggest

def reorder (myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)
    #print("add", add)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1]= myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    #print("NewPoints",myPointsNew)
    return myPointsNew

def getWarp(img,biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    imgCropped = imgOutput[20:imgOutput.shape[0]-20,20:imgOutput.shape[1]-20]
    imgCropped = cv2.resize(imgCropped,(widthImg,heightImg))

    return imgCropped


def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

while True:
    success, img = cap.read()
    img = cv2.resize(img,(widthImg,heightImg))
    imgContour = img.copy()

    imgThres = preProcessing(img)
    biggest = getContours(imgThres)
    if biggest.size !=0:
        imgWarped=getWarp(img,biggest)
        # imageArray = ([img,imgThres],
        #           [imgContour,imgWarped])
        imageArray = ([imgContour, imgWarped])
        cv2.imshow("ImageWarped", imgWarped)
    else:
        # imageArray = ([img, imgThres],
        #               [img, img])
        imageArray = ([imgContour, img])

    stackedImages = stackImages(0.6,imageArray)
    cv2.imshow("WorkFlow", stackedImages)

    if cv.waitKey(1) & 0xFF == ord("q"):
        cap.release()
        cv.destroyAllWindows()
        break
cap.release()
cv.destroyAllWindows()
```
##### 改进代码
##### Related Functions
``` python
# 图像堆叠函数
def stackImages(scale, imgArray):
    # & 输出一个 rows * cols 的矩阵（imgArray）
    rows = len(imgArray)
    cols = len(imgArray[0])
    print(rows,cols)
    # & 判断imgArray[0] 是不是一个list
    rowsAvailable = isinstance(imgArray[0], list)
    # & imgArray[0][0]就是指[0,0]的那个图片（我们把图片集分为二维矩阵，第一行、第一列的那个就是第一个图片）
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range (0, rows):
            for y in range(0, cols):
                # & 判断图像与后面那个图像的形状是否一致，若一致则进行等比例放缩；否则，先resize为一致，后进行放缩
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                # & 如果是灰度图，则变成RGB图像（为了弄成一样的图像）
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        # & 设置零矩阵
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    # & 如果不是一组照片，则仅仅进行放缩 or 灰度转化为RGB
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

# 其他相关算法

# 等比缩小图片 func
def resizeImages(image, scale):
    result = cv2.resize(image, (int(image.shape[1]/scale), int(image.shape[0]/scale)))
    return result

# 图像预处理
def preProcessing(image):
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 灰度图
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1) # 高斯模糊
    imgCanny = cv2.Canny(imgBlur, 200, 200) # 边缘检测
    kernel = np.ones((5,5)) # 结构元素
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2) # 膨胀
    imgThres = cv2.erode(imgDial, kernel, iterations=1) # 腐蚀
    return imgThres

# 绘制Thres在原图的显示
def drawThresInImage(image, imageCopy):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image=imageCopy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

# 获取最大轮廓角点
def getBiggest(image, imageCopy):
    biggest = np.array([])
    maxArea = 0
    # 轮廓检测
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt) # 轮廓内面积
        if area > 500:
            peri = cv2.arcLength(cnt, True) # 周长 计算封闭轮廓的周长或曲线的长度
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True) # 角点 对图像轮廓点进行多边形拟合
            if area > maxArea and len(approx) == 4:
                biggest = approx # 获取最大矩阵框
                maxArea = area # 获取最大面积
    
    # 定位文本四个点
    cv2.drawContours(imageCopy, biggest, -1, (255,0,0), 30) # 定位文本的四个点
    return biggest

def reorder(points):
    pointsReshaped = points.reshape((4,2)) # 四个角点 (4, 2)
    newBiggest = np.zeros((4,1,2), np.int32)
    
    # 点按照一定的顺序重新排列
    add = pointsReshaped.sum(1) # 计算x+y
    newBiggest[0] = pointsReshaped[np.argmin(add)] # 和最小的点是左上角点 left_up
    newBiggest[3] = pointsReshaped[np.argmax(add)] # 和最大的点是右下角点 right_right
    
    diff = np.diff(pointsReshaped, axis=1) # 将点进行x-y差异计算
    newBiggest[1] = pointsReshaped[np.argmin(diff)] # 差异最小的点为右上 right_up
    newBiggest[2] = pointsReshaped[np.argmax(diff)] # 差异最大的点为左下 left_down
    
    return newBiggest

# 鸟瞰转换 透视转换
def getWarp(image, biggest):
    '''
    鸟瞰转换 透视转换
    image -> 图像
    biggest -> 角点坐标 np.array
    '''
    newBiggestPoints = reorder(biggest) # 矩阵角点处理，对角点进行统一排序
    # 点1
    pts1 = np.float32(newBiggestPoints)
    # 点2
    pts2 = np.float32([[0,0], [imgWidth,0], [0,imgHeight],[imgWidth, imgHeight]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2) # 转换矩阵
    imgOutput = cv2.warpPerspective(image, matrix, (imgWidth, imgHeight))
    
    # 裁剪边缘其他背景，将裁剪后的图像重新调整为原理窗口
    imgCropped = imgOutput[20:imgOutput.shape[0]-20, 20:imgOutput.shape[1]-20]
    imgCropped = cv2.resize(imgCropped, (imgWidth,imgHeight))
    
    return imgCropped

# 保存扫描结果并识别文本内容
def saveImageToText(image, count):
    imageDir = "Result/Document Detection Result_" + str(count) + ".jpg"
    textDir = "Result/TextResult_" + str(count) + ".text"
    cv2.imwrite(imageDir, image)
    text = pytesseract.image_to_string(Image.open(imageDir))
    print(text)
    
    with open(textDir, 'w', encoding='utf-8') as f:
        f.write(text)
        f.close
```
##### 基于图片数据扫描文档
标注可以省略，非强制
```python
# 主流程
# Data/book.jpg
# 基于图片数据进行文本扫描
import cv2
import numpy as np
import pytesseract
from PIL import Image
print("Package Imported")

#######################
imgWidth = 480
imgHeight = 640
dataDir = "Data/paper.jpg"
count = 1
#######################

img = cv2.imread(dataDir)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 灰度化处理

imgCopyForContour = img.copy()
imgCopyForBiggest = img.copy()
imgThres = preProcessing(img) # 对原图像预处理
drawThresInImage(imgThres, imgCopyForContour) # Thres Contour
# cv2.imshow("test01", imgCopyForContour)
biggestPoints = getBiggest(imgThres, imgCopyForBiggest) # 获取文本轮廓角点 (4, 1, 2) 4个1行2列的数组
# 标注
cv2.rectangle(img, (0, 0), (1000,200), (255,255,255), cv2.FILLED) # (minX, minY) (maxX, maxY)
cv2.putText(img, "Orignal Image", (30, 130), cv2.FONT_HERSHEY_DUPLEX, 4, (0,0,0), 7)
cv2.rectangle(imgGray, (0, 0),(1000,200), (255,255,255), cv2.FILLED)
cv2.putText(imgGray, "Gray Image", (120, 130), cv2.FONT_HERSHEY_DUPLEX, 4, (0,0,0), 7)
cv2.rectangle(imgThres, (0, 0),(1000,200), (255,255,255), cv2.FILLED)
cv2.putText(imgThres, "Threshold", (180, 130), cv2.FONT_HERSHEY_DUPLEX, 4, (0,0,0), 7)
cv2.rectangle(imgCopyForContour, (0, 0),(1000,200), (255,255,255), cv2.FILLED)
cv2.putText(imgCopyForContour, "Contours", (200, 130), cv2.FONT_HERSHEY_DUPLEX, 4, (0,0,0), 7)
cv2.rectangle(imgCopyForBiggest, (0, 0),(1000,200), (255,255,255), cv2.FILLED)
cv2.putText(imgCopyForBiggest, "Biggest Points", (30, 130), cv2.FONT_HERSHEY_DUPLEX, 4, (0,0,0), 7)

# 判断是否有角点，即是否有文本出现在检测框中
if biggestPoints.size != 0:
    imgWarped = getWarp(img, biggestPoints) # 鸟瞰转换 透视转换
    cv2.rectangle(imgWarped, (0, 0),(175,30), (255,255,255), cv2.FILLED)
    cv2.putText(imgWarped, "Warped Image", (5,20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,0), 2)
    warpGray = cv2.cvtColor(imgWarped, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(warpGray, (0, 0),(175,30), (255,255,255), cv2.FILLED)
    cv2.putText(warpGray, "Warped Gray", (15,20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,0), 2)
    result = cv2.adaptiveThreshold(warpGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 3) # 自适应阈值结果
    cv2.rectangle(result, (0, 0),(175,30), (255,255,255), cv2.FILLED)
    cv2.putText(result, "Scan Result", (20,20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,0), 2)
#     result = cv2.threshold(warpGray, 120, 255, cv2.THRESH_BINARY)[1] # 二值化
    resultImageArray = [[img, imgGray, imgThres, imgCopyForContour], [imgCopyForBiggest, imgWarped, warpGray, result]]
    saveImageToText(result, count)
else:
    resultImageArray = [[img, imgThres], [img, img]]

# 图片堆叠显示
resultImage = stackImages(0.1, resultImageArray)
cv2.imshow("Result", resultImage)
print("Completed!!!")
cv2.waitKey(0)
```
##### 基于视频数据或实时摄像机扫描文档
注意：imgHeight imgWidth
```python
# 主流程
# 基于视频文件或摄像实时提取
import cv2
import numpy as np
import pytesseract
from PIL import Image
print("Package Imported")

#######################
imgWidth = 480
imgHeight = 640
count = 0
cap = cv2.VideoCapture(1)
cap.set(3, 480)
cap.set(4, 640)
cap.set(10, 150)
#######################

while True:
    success, img = cap.read()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 灰度化图像
    imgCopyForContour = img.copy()
    imgCopyForBiggest = img.copy()
    imgThres = preProcessing(img) # 对原图像预处理
    drawThresInImage(imgThres, imgCopyForContour) # Thres Contour
    biggestPoints = getBiggest(imgThres, imgCopyForBiggest) # 获取文本轮廓角点 (4, 1, 2) 4个1行2列的数组
    
    # 判断是否有角点，即是否有文本出现在检测框中
    if biggestPoints.size != 0:
        imgWarped = getWarp(img, biggestPoints) # 鸟瞰转换 透视转换
        warpGray = cv2.cvtColor(imgWarped, cv2.COLOR_BGR2GRAY)
        result = cv2.adaptiveThreshold(warpGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 3) # 自适应阈值结果
        resultImageArray = [[img, imgGray, imgThres, imgCopyForContour], [imgCopyForBiggest, imgWarped, warpGray, result]]
    else:
        resultImageArray = [[img, imgThres], [img, img]]
        
    # 图片堆叠显示
    resultImage = stackImages(0.5, resultImageArray)
    cv2.imshow("Result", resultImage)
        
    if cv2.waitKey(1) & 0xFF == ord("s"):
        saveImageToText(result, count)
        cv2.rectangle(img, (0, 200), (640, 300), (0,255,0), cv2.FILLED)
        cv2.putText(img, "Scan Saved", (150, 265), cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,255), 2)
        cv2.imshow("Scan State", img)
        cv2.waitKey(10)
        count += 1
        continue

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        break
        
cap.release()
cv2.destroyAllWindows()
```
#### Project3-Number Plate Detection 号牌检测
应用脸部识别的haar级联分类器，画框
##### 基于图片数据
函数化
```python
# scanNumberPlate Func
def scanNumberPlate(image, imgCount, objectDir):
    haarRootDir = r"C:\\Users\10959\anaconda3\Lib\site-packages\cv2\data"
    haarType = "\haarcascade_russian_plate_number.xml"
    haarDir = haarRootDir + haarType
    nPlateCascade = cv2.CascadeClassifier(haarDir) # 级联分类器
    minArea = 300
    color = (255,0,255)
    
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 682, 1023 height,width
    numberPlates = nPlateCascade.detectMultiScale(imgGray, 1.1, 10)
    for (x, y, w, h) in numberPlates:
        area = w * h
        if area > minArea:
            cv2.rectangle(image, (x,y), (x+w, y+h), color, 2)
            cv2.putText(image, "Number Plate", (x, y-1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
            imgRoi = imgOri[y:y+h, x:x+w]
            cv2.imwrite(objectDir + str(imgCount) + ".jpg", imgRoi)
    print("Completed!!!")

import cv2
print("Package Imported")

dataDir = "Data/p3.jpg"
imgOri = cv2.imread(dataDir)
savedDir = "Data/nPlate_"

scanNumberPlate(imgOri, 3, savedDir)
```
原代码
```python
import cv2
print("Package Imported")
########################################################################
dataDir = "Data/p3.jpg"
haarRootDir = r"C:\\Users\10959\anaconda3\Lib\site-packages\cv2\data"
haarType = "\haarcascade_russian_plate_number.xml"
haarDir = haarRootDir + haarType
nPlateCascade = cv2.CascadeClassifier(haarDir) # 级联分类器
minArea = 300
color = (255,0,255)
count = 3
########################################################################
imgOri = cv2.imread(dataDir) #682,1023,3 height,width,channel
imgGray = cv2.cvtColor(imgOri, cv2.COLOR_BGR2GRAY) # 682, 1023 height,width

numberPlates = nPlateCascade.detectMultiScale(imgGray, 1.1, 4)
# 画出识别框
for (x, y, w, h) in numberPlates:
    area = w * h
    if area > minArea:
        cv2.rectangle(imgOri, (x, y), (x+w, y+h), color, 2)
        cv2.putText(imgOri, "Number Plate", (x, y-1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
        imgRoi = imgOri[y:y+h, x:x+w]
        cv2.imshow("Number Plate Image Result", imgRoi)

cv2.imshow("Result in Original Image", imgOri)
cv2.waitKey(0)
```
##### 基于视频、摄像头实时视频数据
```python
import cv2
print("Package Imported")
########################################################################
frameWidth = 640
frameHeight = 480
haarRootDir = r"C:\\Users\10959\anaconda3\Lib\site-packages\cv2\data"
haarType = "\haarcascade_russian_plate_number.xml"
haarDir = haarRootDir + haarType
nPlateCascade = cv2.CascadeClassifier(haarDir) # 级联分类器
minArea = 300
color = (255,0,255)
count = 0
########################################################################
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 灰度化图像
    numberPlates = nPlateCascade.detectMultiScale(imgGray, 1.1, 10) # 识别
    for (x, y, w, h) in numberPlates:
        area = w*h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, "Number Plate", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
            imgRoi = img[y:y+h,x:x+w]
            cv2.imshow("ROI", imgRoi)
    
    cv2.imshow("Result", img)
    
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Data/NoPlate_"+ str(count) + ".jpg",imgRoi)
        cv2.rectangle(img, (0,200), (640,300), (0,255,0), cv2.FILLED)
        cv2.putText(img,"Scan Saved", (150,265), cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,255), 2)
        cv2.imshow("Result",img)
        cv2.waitKey(500)
        count +=1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
        
cap.release()
cv2.destroyAllWindows()
```
## OpenCV with python 补充学习 supplement
### 基本操作 Basic Operation
#### Open Data
``` python
import cv2
print('Package Imported')
# 检查视频是否打开
dataDir = "Data/objectDetectionTest.mp4"
vc = cv2.VideoCapture(dataDir)

if vc.isOpened():
    open, frame = vc.read()
else:
    open = False
    
while open:
    ret, frame = vc.read()
    if frame is None:
        break
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("result", gray)
        if cv2.waitKey(0) & 0xFF == 27: # esc退出键  0 表示视频单帧， 
            break
            
vc.release()
cv2.destroyAllWindows()
```
**show image 封装**
``` python
# 打开图像操作 Func 封装
def showImage(image):
    cv2.imshow("Result", image)
    cv2.waitKey(0) # 点击任意键
    cv2.destroyAllWindows()
```
#### 颜色通道提取
OpenCV 通道排序 BGR
``` python
# 颜色通道提取
import cv2

dataDir = "Data/whdTest02.jpg"
img = cv2.imread(dataDir)

b, g, r = cv2.split(img)
print(type(r))
print(r)
print(r.shape)
print(img_r[:,:,2])

# 重新恢复image - 颜色通道merge
imgMerged = cv2.merge((b,g,r))
img.shape
```
**颜色通道分离Image Func 封装**
``` python
import cv2
print("Package Imported")

dataDir = "Data/whdTest02.jpg"
img = cv2.imread(dataDir)

# 只保留R单通道的图像展示
def imgToR(image):
    img_r = image.copy()
    img_r[:,:,0] = 0 # B channel 0
    img_r[:,:,1] = 0 # G channel 0
    return img_r

# 只保留B单通道的图像展示
def imgToB(image):
    img_b = image.copy()
    img_b[:,:,1] = 0 # G channel 0
    img_b[:,:,2] = 0 # R channel 0
    return img_b

# 只保留G单通道的图像展示
def imgToG(image):
    img_g = image.copy()
    img_g[:,:,0] = 0 # B channel 0
    img_g[:,:,2] = 0 # R channel 0
    return img_g

result_r = imgToR(img)
showImage(result_r)
result_b = imgToB(img)
showImage(result_b)
result_g = imgToG(img)
showImage(result_g)
```
#### 数值计算
``` python
# 相当于 % 255
(img + img2)[:5,:,0]

cv2.add(img, img2)[:5,:,0]
```
#### 图像融合
**注**：图像shape值不同不可融合
``` python
img2_fusion = cv2.addWeighted(img1, 0.6, img2_resize, 0.2, 0)
```
**cv2.resize 可实现图像等比例缩放**
``` python
img1_resize = cv2.resize(img1, (0,0), fx=0.5, fy=0.5)
```
#### 图像阈值处理
ret, dst = cv2.threshold(src, thresh, maxval, type)
- src: input image 注：只能输入单通道，一般为gray image
- dst: output image
- thresh: threshold number
- maxval: 当像素超过阈值（或小于阈值 type决定）则赋予的值
- type: 二值化操作类型，cv2.THRESH_BINARY、cv2.THRESH_BINARY_INV、cv2.THRESH_TRUNC、cv2.THRESH_TOZERO、cv2.THRESH_TOZERO_INV
    - cv2.THRESH_BINARY：超过阈值部分取maxval（最大值），否则为0
    - cv2.THRESH_BINARY_INV：cv2.THRESH_BINARY 反转
    - cv2.THRESH_TRUNC：大于阈值部分设为阈值，否则不变
    - cv2.THRESH_TOZERO： 大于阈值部分不变，否则为0
    - cv2.THRESH_TOZERO_INV： cv2.THRESH_TOZERO 反转
``` python
import cv2
import matplotlib.pyplot as plt
print("Package Imported")

dataDir2 = "Data/whdTest03.jpg"
img = cv2.imread(dataDir2)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 阈值变化
ret1, thresh1 = cv2.threshold(imgGray, 127, 255, cv2.THRESH_BINARY) # 二值处理
ret2, thresh2 = cv2.threshold(imgGray, 127, 255, cv2.THRESH_BINARY_INV) 
ret3, thresh3 = cv2.threshold(imgGray, 127, 255, cv2.THRESH_TRUNC) 
ret4, thresh4 = cv2.threshold(imgGray, 127, 255, cv2.THRESH_TOZERO) 
ret5, thresh5 = cv2.threshold(imgGray, 127, 255, cv2.THRESH_TOZERO_INV) 

titles = ["Original Image", "Binary", "Binary_inv", "Trunc", "Tozero", "Tozero_inv"]
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], "gray")
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

# showImage(img)
# showImage(thresh1)
# print("threshold1: ", ret1)
```
#### 图像平滑处理
首先给图像添加椒盐噪声，作为噪声图像
``` python
# 给图像加椒盐噪声
import cv2
import numpy as np
print("Package Imported")
    
dataDir = "Data/whdTest03.jpg"
img = cv2.imread(dataDir)
noise = np.random.randint(0, 256, size=img.shape)##生成随机噪声 注意这个函数是下闭上开的
noise = np.where(noise > 250, 255, 0) #设定一个阈值，大于的取255，小于的取0

noise = noise.astype('float')
img = img.astype("float")
img = img + noise
#读入的图像的数据类型是uint8，相加的话不会截取，而是自动对256取余，所以我们需要转换为float后再相加
#这时候图像的数据都是float，并且有的是大于255的，对于大于255的，我们进行截取
img = np.where(img>255,255,img)
img = img.astype('uint8')
showImage(img)
cv2.imwrite("Data/whdNoise.jpg", img)
```
各种滤波去噪
``` python
import cv2
print("Package Imported")

# 均值滤波 简单的平均卷积操作
blur = cv2.blur(img, (3,3))

# 方框滤波 当归一化时normalize=True结果与均值滤波一致，否则像素越界像素值为255
box_T = cv2.boxFilter(img, -1, (3,3), normalize=True) # 结果与均值滤波一致
box_F = cv2.boxFilter(img, -1, (3,3), normalize=False)

# 高斯滤波 高斯模糊的卷积核中数值需要满足高斯分布，相当于更重视中间数值
aussian = cv2.GaussianBlur(img, (5,5), 1)

# 中值滤波 相当于中值替代
median = cv2.medianBlur(img, 5)

# 拼接结果 hstack 横向拼接 vstack 纵向拼接
import numpy as np
result = np.hstack((img, blur, aussian, median))
showImage(result)
```
#### 图像处理
补充梯度处理、顶帽、黑帽
##### 梯度处理
``` python
# 梯度运算=膨胀-腐蚀
import cv2
import numpy as np
print("Package Imported")

dataDir = "Data/whdTest02.jpg"
img = cv2.imread(dataDir)
kernel = np.ones((7,7), np.uint8)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dilate = cv2.dilate(imgGray,kernel,iterations=5)
erosion = cv2.erode(imgGray, kernel, iterations=5)

result = np.hstack((dilate, erosion))
showImage(result)

gradient = cv2.morphologyEx(imgGray, cv2.MORPH_GRADIENT, kernel)
showImage(gradient)
```
##### 顶帽
顶帽：原始输入-开运算结果
``` python
# 顶帽：原始输入-开运算结果
import cv2
print("Package Imported")

dataDir = "Data/whdTest02.jpg"
img = cv2.imread(dataDir)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = np.ones((7,7), np.uint8)

tophat = cv2.morphologyEx(imgGray, cv2.MORPH_TOPHAT, kernel)
showImage(tophat)
```
##### 黑帽
黑帽：闭运算-原始输入
``` python
# 黑帽：闭运算-原始输入
import cv2
print("Package Imported")

dataDir = "Data/whdTest02.jpg"
img = cv2.imread(dataDir)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = np.ones((7,7), np.uint8)

blackhat = cv2.morphologyEx(imgGray, cv2.MORPH_BLACKHAT, kernel)
showImage(blackhat)
```
#### 图像梯度处理
1. Sobel算子
dst = cv2.Sobel(src, ddepth, dx, dy, ksize)
- ddepth：图像深度
- dx、dy：分别表示水平、垂直方向
- ksize：Sobel算子大小
2. Scharr算子
结果敏感，含有较多细节
3. laplacian算子
对噪声敏感
``` python
# 不同算子的差异
import cv2
import numpy as np
print("Package Imported")

dataDir = "Data/whdTest02Gray.jpg"
img = cv2.imread(dataDir, cv2.IMREAD_GRAYSCALE)
showImage(img)

# sobel
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
# showImage(sobelx)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobely = cv2.convertScaleAbs(sobely)
# showImage(sobely)
# 再求xy轴
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
# showImage(sobelxy)

# scharr
scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
scharrx = cv2.convertScaleAbs(scharrx)
scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
scharry = cv2.convertScaleAbs(scharry)
scharrxy = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)
# showImage(scharrxy)

# laplacian
laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

result = np.hstack((sobelxy, scharrxy, laplacian))
showImage(result)
```
#### Canny边缘检测
- 1. 使用高斯滤波器平滑图像，滤除噪音
- 2. 计算图中每个像素点的梯度强度和方向
- 3. 应用非极大值（Non-Maximum Suppression）抑制，以消除边缘检测带来的杂散响应
- 4. 应用双阈值（Double-Threshold）检测来确定真实和潜在边缘
- 5. 通过抑制孤立的弱边缘最终完成边缘检测
``` python
import cv2
import numpy as np
print("Package Imported")

dataDir = "Data/whdTest02Gray.jpg"
img = cv2.imread(dataDir, cv2.IMREAD_GRAYSCALE)

canny1 = cv2.Canny(img, 80, 150) #minVal,maxVal
canny2 = cv2.Canny(img, 50, 100)

result = np.hstack((img, canny1, canny2))
showImage(result)
```
#### 图像金字塔
- 高斯金字塔
- 拉普拉斯金字塔
##### 高斯金字塔
1. 向下采样方法（缩小）
    - 1.图像与高斯内核卷积
    - 2.去除所有偶数行和列
2. 向上采用方法（放大）
    - 1.将图像在每个方向扩大为原来的两倍，新增行和列以0填充
    - 2.使用先前同样的内核（乘以4）与放大后的图像卷积，获得近似值
``` python
import cv2
print("Package Imported")

dataDir = "Data/whdTest03.jpg"
imgOri = cv2.imread(dataDir)
print(imgOri.shape) # (445, 520, 3)
showImage(imgOri)

# 上采样 放大
upImage = cv2.pyrUp(imgOri)
print(upImage.shape) # (890, 1040, 3)
showImage(upImage)

# 下采样 缩小
downImage = cv2.pyrDown(imgOri)
print(downImage.shape) # (223, 260, 3)
showImage(downImage) 

# 原图与采样后对比，图像清晰度下降
import cv2
import numpy as np
print("Package Imported")

dataDir = "Data/whdTest02.jpg"
imgOri = cv2.imread(dataDir)

imgUp = cv2.pyrUp(imgOri)
upDown = cv2.pyrDown(imgUp)

result = np.hstack((imgOri, upDown))
showImage(result)
```
##### 拉普拉斯金字塔
先down后up，原图-downUp
``` python
import cv2
import numpy as np
print("Package Imported")

dataDir = "Data/whdTest02.jpg"
imgOri = cv2.imread(dataDir)

# 先down后up，原图-downUp
imgDown = cv2.pyrDown(imgOri)
downUp = cv2.pyrUp(imgDown)
laplas = imgOri - downUp

result = np.hstack((imgOri, laplas))
showImage(result)
```
#### 图像轮廓
**cv2.findContours(img, mode, method)**
- mode: 轮廓检测模式
    - RETR_EXTERNAL：只检测最外面的轮廓
    - RETR_LIST：检测所有的轮廓，并将其保存到一条链表中
    - RETR_CCOMP：检测所有轮廓，将其组织为两层，顶部是各部分的外部边界，第二层是空洞边界
    - RETR_TREE：检测所有轮廓，并重构嵌套轮廓的整个层次
- method：轮廓逼近方法
    - CHAIN_APPROX_NONE：以Freeman链码的方式输出轮廓，所有其他方法输出多边形（顶点序列）
    - CHAIN_APPROX_SIMPLE：压缩水平的、垂直的和斜的，函数只保留终点部分
``` python
# 注：需要对原始数据进行预处理-灰度化，二值化
import cv2
import numpy as np
print("Package Imported")

dataDir = "Data/whdTest02.jpg"
img = cv2.imread(dataDir)
# 图像预处理
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 灰度化，消除通道
ret, thresh = cv2.threshold(imgGray, 127, 255, cv2.THRESH_BINARY) # 二值化
# showImage(thresh)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # 注：不能使用多通道图像，必须单通道图片
# 绘制轮廓
# 传入绘制图像，轮廓，轮廓索引，颜色模式，线条厚度
imgCopy = img.copy() # 对原图像无影响
imgCon = cv2.drawContours(imgCopy, contours, -1, (0,0,255), 1)
# showImage(result)
result = np.hstack((img, imgCon))
showImage(result)
```
##### 轮廓特征
``` python
cnt = contours[0]
# 面积
cv2.contourArea(cnt)
# 周长
cv2.arcLength(cnt, True) # True 表示闭合周长
```
##### 轮廓近似
``` python
import cv2
import numpy as np
print("Package Imported")

dataDir = "Data/whdTest02.jpg"
imgOri = cv2.imread(dataDir)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 灰度化，消除通道
ret, thresh = cv2.threshold(imgGray, 127, 255, cv2.THRESH_BINARY) # 二值化
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # 注：不能使用多通道图像，必须单通道图片
cnt = contours[0]

imgDraw1 = img.copy()
result1 = cv2.drawContours(imgDraw1, [cnt], -1, (0,0,255), 2)

epsilon = 0.1 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)

imgDraw2 = img.copy()
result2 = cv2.drawContours(imgDraw2, [approx], -1, (0,0,255), 2)
```
#### 模板匹配
模板匹配和卷积原理相似，模板在原图像上从原点开始滑动，计算模板与（图像被模板覆盖的地方）的差别程度，这个差别程度的计算方法在opencv里有6种，然后将每次计算的结果放入矩阵种，作为输出结果。若原图A*B，模板a*b，则结果矩阵（A-a+1）*(B-b+1)
- TM_SQDIFF：计算平方不同，计算出来的值越小，越相关
- TM_CCORR：计算相关性，结果值越大，越相关
- TM_CCOEFF：计算相关系数，结果值越大，越相关
- TM_SQDIFF_NORMED：计算归一化平方不同，结果值越接近0，越相关
- TM_CCORR_NORMED：计算归一化相关性，结果值越接近1，越相关
- TM_CCOEFF_NORMED：计算归一化相关系数，结果越接近1，越相关
``` python
# 模板匹配
import cv2
import matplotlib.pyplot as plt
print("Package Imported")

#######################################################################################################################################
methods = ["cv2.TM_SQDIFF", "cv2.TM_CCORR", "cv2.TM_CCOEFF", "cv2.TM_SQDIFF_NORMED", "cv2.TM_CCORR_NORMED", "cv2.TM_CCOEFF_NORMED"]
imgOriDri = "Data/whdTest02.jpg"
imgOri = cv2.imread(imgOriDri)
templateDri = "Data/whdTest02.jpg"
template = cv2.imread(templateDri)
h, w = template.shape[0:2]
#######################################################################################################################################

for meth in methods:
    img2 = img.copy()
    
    # 匹配方法真值
    method = eval(meth)
    print(method)
    result = cv2.matchTemplate(img, template, method)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)

    # 若平方差匹配TM_SQDIFF或归一化平方差匹配TM_SQDIFF_NORMED，取最小值
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        topLeft = minLoc
    else:
        topLeft = maxLoc
    bottomRight = (topLeft[0] + w, topLeft[1] + h)
    
    # 画矩阵
    cv2.rectangle(img2, topLeft, bottomRight, 255, 2)
    
    plt.subplot(121), plt.imshow(result, cmap="gray")
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img2, cmap="gray")
    plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()

# 匹配多个对象
import cv2
import numpy as np

img_rgb = cv2.imread("Data/whdTest02.jpg")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 灰度化，消除通道
template = cv2.imread("Data/whdTest02.jpg")
h, w = template.shape[0:2]

result = cv2.matchTemplate(imgGray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
# 取匹配程度大于%80的坐标
loc = np.where(result >= threshold)
for pt in zip(*loc[::-1]): # * 表示可选参数
    bottomRight = (pt[0] + w, pt[1] + h)
    cv2.rectangle(img_rgb, pt, bottomRight, (0,0,255), 2)
    
showImage(img_rgb)
```
#### 图像直方图
**cv2.calcHist(images, channels, mask, histSize, ranges)**
- images: 原图像 格式unit8/float32, [img]
- channels: [0]-灰度图 [1][2][3]-BGR
- mask: 掩膜图像，整幅图像None, 某一部分则需要制作相关掩膜
- histSize: Bin数目
- ranges: 像素值范围[0,256]
``` python
import cv2
import matplotlib.pyplot as plt
print("Package Imported")

##############################
imagePath = "Data/whdTest03.jpg"
imgBGR = cv2.imread(imagePath)
imgGray = cv2.imread(imagePath, 0) # 直接读成灰度图
##############################

# 灰度图 histogram
imgHist = cv2.calcHist([imgGray], [0], None, [256], [0, 256]) # histogram
print("Histogram shape: ", imgHist.shape)
plt.hist(imgGray.ravel(), 256)
plt.show()

# 彩色图 histogram
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0,256])
```
##### 创建mask掩码
``` python
import numpy as np
import cv2
import matplotlib.pyplot as plt
print("Package Imported")

##############################
imagePath = "Data/whdTest03.jpg"
imgBGR = cv2.imread(imagePath)
imgGray = cv2.imread(imagePath, 0) # 直接读成灰度图
##############################

mask = np.zeros(imgBGR.shape[:2], np.uint8)
mask[100:350, 100:400] = 255
imgMasked = cv2.bitwise_and(imgGray, imgGray, mask=mask) # 与操作

histFull = cv2.calcHist([imgGray], [0], None, [256], [0,256]) # 无掩码
histMask = cv2.calcHist([imgGray], [0], mask, [256], [0,256]) # 有掩码

# 画出结果
plt.subplot(221), plt.imshow(imgGray, 'gray')
plt.subplot(222), plt.imshow(mask, 'gray')
plt.subplot(223), plt.imshow(imgMasked, 'gray')
plt.subplot(224), plt.plot(histFull), plt.plot(histMask)
plt.xlim([0, 256])
plt.show()
```
##### 直方图均衡化
``` python
import numpy as np
import cv2
import matplotlib.pyplot as plt
print("Package Imported")

##############################
imagePath = "Data/whdTest02.jpg"
imgBGR = cv2.imread(imagePath)
imgGray = cv2.imread(imagePath, 0) # 直接读成灰度图
##############################
# 灰度图 histogram
imgHist = cv2.calcHist([imgGray], [0], None, [256], [0, 256]) # histogram
print("Histogram shape: ", imgHist.shape)
plt.hist(imgGray.ravel(), 256)
plt.show()
# 直方图均衡化
histEqu = cv2.equalizeHist(imgGray)
plt.hist(histEqu.ravel(), 256)
plt.show()

# 对比展示
result = np.hstack((imgGray, histEqu))
showImage(result)
```
##### 直方图均衡化
``` python
import numpy as np
import cv2
import matplotlib.pyplot as plt
print("Package Imported")

##############################
imagePath = "Data/whdTest03.jpg"
imgBGR = cv2.imread(imagePath)
imgGray = cv2.imread(imagePath, 0) # 直接读成灰度图
##############################

# 灰度图 histogram
imgHist = cv2.calcHist([imgGray], [0], None, [256], [0, 256]) # histogram
print("Histogram shape: ", imgHist.shape)
# plt.hist(imgGray.ravel(), 256)
# plt.show()
# 直方图均衡化
histEqu = cv2.equalizeHist(imgGray)
# plt.hist(histEqu.ravel(), 256)
# plt.show()
# 自适应直方图均衡化
histClahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
imgClahe = histClahe.apply(imgGray)
showImage(imgClahe)

# 对比展示
result = np.hstack((imgGray, histEqu, imgClahe))
showImage(result)
```
#### 傅里叶变化
## 傅里叶变化
1. 作用
    - 高频：变化剧烈的灰度分离，例如边界
    - 低频：变化缓慢的灰度分离，例如一片大海
2. 滤波
    - 低通滤波器：只保留低频，使图像模糊
    - 高通滤波器：只保留高频，图像细节增强
3. 注
    - opencv中cv2.dft()和cv2.idft(),输入图像需要先转为np.float32格式
    - 频率为0部分在左上角，需要通过shift转到中间位置
    - dft结果为双通道(实部，虚部), 需要再转为图片格式
``` python
import numpy as np
import cv2
import matplotlib.pyplot as plt
print("Package Imported")

##############################
imagePath = "Data/whdTest03.jpg"
imgBGR = cv2.imread(imagePath)
imgGray = cv2.imread(imagePath, 0) # 直接读成灰度图
imgFloat32 = np.float32(imgGray)
##############################

imgDft = cv2.dft(imgFloat32, flags=cv2.DFT_COMPLEX_OUTPUT)
dftShift = np.fft.fftshift(imgDft)
result = 20 * np.log(cv2.magnitude(dftShift[:,:,0], dftShift[:,:,1]))

plt.subplot(121), plt.imshow(imgGray, 'gray')
plt.title("Input Image"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(result, 'gray')
plt.title("Magnitude Spectrum"), plt.xticks([]), plt.yticks([])
plt.show()
```
##### 低通滤波
``` python
# 低通滤波
import numpy as np
import cv2
import matplotlib.pyplot as plt
print("Package Imported")

##############################
imagePath = "Data/whdTest02.jpg"
imgBGR = cv2.imread(imagePath)
imgGray = cv2.imread(imagePath, 0) # 直接读成灰度图
imgFloat32 = np.float32(imgGray)
##############################

imgDft = cv2.dft(imgFloat32, flags=cv2.DFT_COMPLEX_OUTPUT)
dftShift = np.fft.fftshift(imgDft)

rows, cols = imgGray.shape
crow, ccol = int(rows/2), int(cols/2) # 中心位置

# 低通滤波
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

# IDFT
fshift = dftShift * mask
f_ishift = np.fft.ifftshift(fshift)
imgBack = cv2.idft(f_ishift)
imgBack = cv2.magnitude(imgBack[:,:,0], imgBack[:,:,1])

plt.subplot(121), plt.imshow(imgGray, "gray")
plt.title("Input Image"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(imgBack, "gray")
plt.title("Result"), plt.xticks([]), plt.yticks([])
plt.show()
```
##### 高通滤波
``` python
# 高通滤波
import numpy as np
import cv2
import matplotlib.pyplot as plt
print("Package Imported")

##############################
imagePath = "Data/whdTest02.jpg"
imgBGR = cv2.imread(imagePath)
imgGray = cv2.imread(imagePath, 0) # 直接读成灰度图
imgFloat32 = np.float32(imgGray)
##############################

imgDft = cv2.dft(imgFloat32, flags=cv2.DFT_COMPLEX_OUTPUT)
dftShift = np.fft.fftshift(imgDft)

rows, cols = imgGray.shape
crow, ccol = int(rows/2), int(cols/2) # 中心位置

# 高通滤波
mask = np.ones((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 0

# IDFT
fshift = dftShift * mask
f_ishift = np.fft.ifftshift(fshift)
imgBack = cv2.idft(f_ishift)
imgBack = cv2.magnitude(imgBack[:,:,0], imgBack[:,:,1])

plt.subplot(121), plt.imshow(imgGray, "gray")
plt.title("Input Image"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(imgBack, "gray")
plt.title("Result"), plt.xticks([]), plt.yticks([])
plt.show()
```
#### 角点检测
``` python
import numpy as np
import cv2
import matplotlib.pyplot as plt
print("Package Imported")

##############################
imagePath = "Data/whdTest02.jpg"
imgBGR = cv2.imread(imagePath)
imgGray = cv2.imread(imagePath, 0) # 直接读成灰度图
# imgFloat32 = np.float32(imgGray)
##############################

imgDst = cv2.cornerHarris(imgGray, 2, 3, 0.04)
print("dst shape", imgDst.shape)
imgBGR[imgDst>0.01*imgDst.max()] = [0,0,255]
showImage(imgBGR)
```
## 背景建模
### 帧差法
由于场景中的目标在运动，目标的影像在不同图像帧中的位置不同。该类算法对时间上连续的两帧图像进行差分运算，不同帧对应的像素点相减，判断灰度差的绝对值，当绝对值超过一定阈值时，即可判断为运动目标，从而实现目标的检测功能。简单但会引起空洞
### 混合高斯模型
在进行前景检测前，先对背景进行训练，对图像中每个背景采用一个混合高斯模型进行模拟，每个背景的混合高斯的个数可以自适应。然后在测试阶段，对新来的像素进行GMM匹配，如果该像素值能够匹配其中一个高斯，则认为是背景，否则认为是前景。由于整个过程GMM模型在不断更新学习中，所以对动态背景有一定的鲁棒性。最后通过对一个有树枝摇摆的动态背景进行前景检测，取得了较好的效果。在视频中对于像素点的变化情况应当是符合高斯分布。背景的实际分布应当是多个高斯分布混合在一起，每个高斯模型也可以带有权重
#### 学习方法
- 1.首先初始化每个高斯模型矩阵参数。
- 2.取视频中T帧数据图像用来训练高斯混合模型。来了第一个像素之后用它来当做第一个高斯分布。
- 3.当后面来的像素值时，与前面已有的高斯的均值比较，如果该像素点的值与其模型均值差在3倍的方差内，则属于该分布，并对其进行参数更新。
- 4.如果下一次来的像素不满足当前高斯分布，用它来创建一个新的高斯分布。
#### 测试方法
在测试阶段，对新来像素点的值与混合高斯模型中的每一个均值进行比较，如果其差值在2倍的方差之间的话，则认为是背景，否则认为是前景。将前景赋值为255，背景赋值为0。这样就形成了一副前景二值图。
``` python
import numpy as np
import cv2
print("Package Imported")

dataDri = "Data/test.avi"
cap = cv2.VideoCapture(dataDri) # 经典测试视频
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) # 形态学操作者使用
fgbg = cv2.createBackgroundSubtractorMOG2() # 创建混合高斯模型用于背景建模

while(True):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel) #形态学开运算去噪点
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 寻找视频中的轮廓
    
    for c in contours:
        perimeter = cv2.arcLength(c, True) # 轮廓周长
        if perimeter > 188 and perimeter < 500:
            x,y,w,h = cv2.boundingRect(c) # 找到一个直矩形（不会旋转）
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2) # 画出这个矩形
    cv2.imshow("frame", frame)
    cv2.imshow("fgmask", fgmask)
    k = cv2.waitKey(150) & 0xff
    if k == 27:
        break
        
cap.release()
cv2.destroyAllWindows()
```
#### 光流法
光流是空间运动物体在观测成像平面上的像素运动的“瞬时速度”，根据各个像素点的速度矢量特征，可以对图像进行动态分析，例如目标跟踪。
- 亮度恒定：同一点随着时间的变化，其亮度不会发生改变。
- 小运动：随着时间的变化不会引起位置的剧烈变化，只有小运动情况下才能用前后帧之间单位位置变化引起的灰度变化去近似灰度对位置的偏导数。
- 空间一致：一个场景上邻近的点投影到图像上也是邻近点，且邻近点速度一致。因为光流法基本方程约束只有一个，而要求x，y方向的速度，有两个未知变量。所以需要连立n多个方程求解。
**cv2.calcOpticalFlowPyrLK()**参数：
- prevImage 前一帧图像
- nextImage 当前帧图像
- prevPts 待跟踪的特征点向量
- winSize 搜索窗口的大小
- maxLevel 最大的金字塔层数
返回：
- nextPts 输出跟踪特征点向量
- status 特征点是否找到，找到的状态为1，未找到的状态为0
``` python
import numpy as np
import cv2
print("Package Imported")

dataDri = "Data/test.avi"
cap = cv2.VideoCapture(dataDri)

featureParams = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 7) # 角点检测所需参数
lkParams = dict(winSize=(15,15), maxLevel=2) # lucas kanade参数
color = np.random.randint(0,255,(100,3)) # 随机颜色条

ret, oldFrame = cap.read() # 拿到第一帧图像
oldGray = cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY) # 灰度化

# 返回所有检测特征点，需要输入图像，角点最大数量（效率），品质因子（特征值越大的越好，来筛选）
# 距离相当于这区间有比这个角点强的，就不要这个弱的了
p0 = cv2.goodFeaturesToTrack(oldGray, mask=None, **featureParams)
mask = np.zeros_like(oldFrame) # 创建一个mask

while(True):
    ret,frame = cap.read()
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 需要传入前一帧和当前图像以及前一帧检测到的角点
    p1, st, err = cv2.calcOpticalFlowPyrLK(oldGray, frameGray, p0, None, **lkParams)

    # st=1表示
    good_new = p1[st==1]
    good_old = p0[st==1]

    # 绘制轨迹
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (int(a),int(b)), (int(c),int(d)), color[i].tolist(), 2)
        frame = cv2.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1)
    img = cv2.add(frame, mask)

    cv2.imshow('frame', img)
    k = cv2.waitKey(150) & 0xff
    if k == 27:
        break

    # 更新
    oldGray = frameGray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()
```
### OpenCV DNN模块
是OpenCV专门用来实现深度神经网络相关的模块
