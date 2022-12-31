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
算法：
