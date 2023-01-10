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
## OpenCV with python
Github Link: https://github.com/murtazahassan/Learn-OpenCV-in-3-hours
B站：【3h精通Opencv-Python】https://www.bilibili.com/video/BV16K411W7x9?vd_source=cf518f0e157700ce8a169afae9bf19ea
### 1-读取数据
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
### 2-图像基本处理
灰度图像、模糊图像、图像边缘、图像膨胀、图像腐蚀functions
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
### 3-图像大小及裁剪
重置图像大小、裁剪图片
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
### 4-画图
画出矩形、线条、圆形、文本
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
### 5-透视
warp prespective
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