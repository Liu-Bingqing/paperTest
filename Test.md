# 实验
## 相关实验操作function
### 灰度化图像
``` python
# grayImageFunc
def grayImageFunc (image_dir, changeStyle=cv2.COLOR_BGR2GRAY):
    image = cv2.imread(image_dir)
    grayImage = cv2.cvtColor(image, changeStyle)
    return grayImage

# 帧差图像
background = gray003
diffImage = cv2.absdiff(background, gray004) # 帧差函数
diffImagergb = cv2.cvtColor(diffImage, cv2.COLOR_GRAY2RGB) #差分灰度rgb图
plt.imshow(diffImagergb)

# 二值化图像：grayImage -> 差值灰度图像
# binaryImageFunc
def binaryImageFunc (greyImage, initialThreshold, maxval):
    binaryDiffImage = cv2.threshold(greyImage, initialThreshold, maxval, cv2.THRESH_BINARY)[1]
    return binaryDiffImage

# 膨胀
# expandImageFunc
def expandImageFunc (image, size=5, pos=2):
    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, pos)) 
    kernel = np.ones((5,5),np.uint8)
    expandImage = cv2.dilate(image, es, iterations=2) #Morphological dilation 形态学膨胀
    return expandImage

# 开运算
# openImageFunc 
def openImageFunc (image, size=5, pos=3):
    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, pos)) 
    openImage = cv2.morphologyEx(image, cv2.MORPH_OPEN, es, iterations=2)
    return openImage

# 闭运算
# closeImageFuc
def closeImageFunc (image, size=5, pos=3):
    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, pos)) 
    closeImage = cv2.morphologyEx(image, cv2.MORPH_CLOSE, es, iterations=2)
    return closeImage

# 保存图像
cv2.imwrite('CNN003.jpg', closeImage)
# GMM
# 导入包
import copy
import numpy as np
import cv2
import matplotlib.image as mping
import matplotlib.pyplot as plt #用于画图
# 视频地址
cmvideodir = "copyMachine01.mp4"
cap01 = cv2.VideoCapture(cmvideodir)
backSub_mog2 = cv2.createBackgroundSubtractorMOG2() # MG2 创建背景分离对象 
# # 主函数，循环视频帧，构建背景模型
while True:
    ret, frame01mog2 = cap01.read()
    if frame01mog2 is None:
        break
    #更新背景模型
    fgmask01mog2 = backSub_mog2.apply(frame01mog2)
    #获取帧号并将其写入当前帧
    cv2.rectangle(frame01mog2, (10, 2), (100,20), (255,255,255), -1)
    cv2.putText(frame01mog2, str(cap01.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
#     cv2.imshow('Frame', frame)
#     cv2.imshow('FG Mask', fgmask)
#     plt.imshow(frame)
#     plt.imshow(fgmask)
grayrgb01mog2 = cv2.cvtColor(fgmask01mog2, cv2.COLOR_GRAY2BGR)
plt.imshow(grayrgb01mog2)
cap01.release
# 图片文件合成视频code
# 导入包
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt #用于画图
# 每张图像大小
size = (480,720)
print("每张图片大小{}{}".format(size[0],size[1]))
# 设置源路径与保存路径
srcPath = r"C:\\Users\\10959\\Desktop\\fix paper\\Paper Image set\\CDnet2014\\copyMachine\\input\\" # 源路径
savePath = r"C:\Users\10959\Desktop\fix paper\Paper Image set\CDnet2014\copyMachine\copyMachine.avi" #保存视频
# 获取图片总个数
allFiles = os.listdir(srcPath)
index = len(allFiles)
print("图片总数为：" + str(index) + "张")
# 设置视频写入器
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# 完成写入对象的创建
# 第一个参数是合成之后视频名称
# 第二个参数是可以使用的编码器
# 第三个参数是帧率即每秒展示多少张图片
# 第四个参数是图片大小
videowrite = cv2.VideoWriter(savePath,fourcc,25,size,True)
imgArray = []
# 读取所有jgp格式图片
for filename in [srcPath + '0' + r' ({0}).jpg'.format(i) for i in range(1,index)]:
    img = cv2.imread(filename)
    if img is None:
        print(filename + " is error!")
        continue
    imgArray.append(img)
# 合成视频
for i in range(0, index-1):
    imgArray[i] = cv2.resize(imgArray[i],(480,720))
    videowrite.write(imgArray[i])
    print("第{}张图片合成成功".format(i))
print("----done!!!----")
videowrite.release
```
## 图4实验 在UR Fall Detection数据上，不同算法提取结果
算法：灰度图 - LOBSTER [3] – GMM[19] – Alphapose[4] – CNN[5] – GAN[6] – HFID
灰度图 ok
LOBSTER ok
GMM ok
Alphapose ok 
CNN ok
GAN ok
HFID ok
003帧、052帧、090帧、103帧
## 图5实验  不同数据集上不同算法提取结果
算法：灰度图 - LOBSTER[3] - GMM - Alphapose - CNN - GAN - HFID
灰度图 ok
LOBSTER ok
GMM ok
Alphapose ok
CNN ok
GAN ok
HFID ok
copyMachine、office、set、fall


