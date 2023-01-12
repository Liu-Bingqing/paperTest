# Body Detecion with python-OpenCV
## body detection in Image
```python
# 相关方法
# 等比缩放图片func resizeImage
def resize_keep_aspectratio(img, dst_size):
    srcH, srcW = img.shape[:2]
    dstH, dstW = dst_size
    # 判断应该按照哪个边做等比缩放
    h = int(dstW * (float(srcH) / srcW)) # 按照w做等比缩放
    w = int(dstH * (float(srcW) / srcH)) # 按照h做等比缩放
#     h = int(h)
#     w = int(w)
    if h < dstH:
        imgResult = cv.resize(img, (dstW, int(h)))
    else:
        imgResult = cv.resize(img, (int(w), dstH))
    
    h_, w_ = imgResult.shape[:2]
    print("Completed!!!")
    
    return imgResult

# 判断框中框is_inside func
def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih

# 筛选识别出的人矩形数据sreenFound func
def screenFound(found):
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and is_inside(r, q):
                break
                
            else:
                foundFiltered.append(r)
# 框出人 出入图片 矩形参数 BGR drawPerson func
def drawPerson(img, person, bgr):
    x, y, w, h = person
    cv.rectangle(img, (x, y), (x + w, y + h), bgr, 2)
    
# cascade图片人体识别和绘制边框
#参数 需要识别的图片 输出绘制的图片 xml路径 bgr颜色 目标的最小尺寸 目标的最大尺寸   
def cascadeImgPersonDetectionDraw(srcImg, dstImg, xmlPath, bgr, minSize, maxSize):
    # image表示的是要检测的输入图像
    # objects表示检测到的人脸目标序列
    # scaleFactor表示每次图像尺寸减小的比例
    # minNeighbors表示每一个目标至少要被检测到3次才算是真的目标(因为周围的像素和不同的窗口大小都可以检测到人脸)
    # flag 对于旧级联具有与函数cvHaarDetectObjects相同的含义。它不用于新的级联。
    # minSize为目标的最小尺寸
    # maxSize为目标的最大尺寸
    # found = detector.detectMultiScale(src_img)
    xmlPath = r"C:\\Users\10959\anaconda3\Lib\site-packages\cv2\data" + xmlPath
    detecor = cv.CascadeClassifier(xmlPath)
    found = detecor.detectMultiScale(srcImg, 1.1, 3, cv.CASCADE_SCALE_IMAGE, (0,0), (500,500))
    # 筛选识别出矩形数据
    screenFound(found)
    for person in foundFiltered:
        drawPerson(dstImg, person, bgr)
    foundFiltered.clear()
    print(xmlPath + " 识别出:" + str(len(found)) + "个结果。")
```
```python
# Test 调用方法
imgDri = "Data/Pedestrian.png"

imgOri = cv.imread(imgDri)
imgResize = resize_keep_aspectratio(imgOri, [500,500]) # 等比缩放
imgGray = cv.cvtColor(imgResize, cv.COLOR_BGR2GRAY)

# 存储所有识别出的坐标集合
foundFiltered = []

hog = cv.HOGDescriptor()
# 加载SVM模型 行人识别
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
# 对图像进行多尺度目标检测 返回检测区域坐标
found, w = hog.detectMultiScale(imgGray)
# print(found)

# 筛选识别出的人矩形数据
screenFound(found)

# 在原图上绘制出识别出的所有矩形 蓝色
for person in foundFiltered:
    drawPerson(imgResize, person, (255,0,0))
    
# 清空列表
foundFiltered.clear()
print("HOGDescriptor_getDefaultPeopleDetector 识别出:" + str(len(found)) + "个结果。")

# 正脸识别 绿色
cascadeImgPersonDetectionDraw(imgGray, imgResize, "\haarcascade_frontalface_default.xml", (0, 255, 0), (0, 0), (500, 500))
# 侧脸识别 红色
cascadeImgPersonDetectionDraw(imgGray, imgResize, "\haarcascade_profileface.xml", (0, 0, 255), (0, 0), (500, 500))
# 全身识别
cascadeImgPersonDetectionDraw(imgGray, imgResize, "\haarcascade_fullbody.xml", (255, 255, 0), (0, 0), (500, 500))
# 上半身识别 洋红
cascadeImgPersonDetectionDraw(imgGray, imgResize, "\haarcascade_upperbody.xml", (255, 0, 255), (0, 0), (500, 500))
# 下半身识别 黄色
cascadeImgPersonDetectionDraw(imgGray, imgResize, "\haarcascade_lowerbody.xml", (0, 255, 255), (0, 0), (500, 500))

# cv.imshow("Original Image", imgOri)
cv.imshow("Detecion Result", imgResize)


cv.waitKey(0)
```
## body detection in video
```python 
# func
import cv2 as cv
print("Package Imported")

# cascade视频人体识别和绘制边框 参数 需要识别的视频路径 xml路径 bgr颜色 目标的最小尺寸 目标的最大尺寸
# 按 Q 键退出函数
def cascadeVideoPersonDetecionDraw(videoPath, xmlPath, bgr, minSize, maxSize):
    cap = cv.VideoCapture(videoPath)
    
    # 识别分类器
    xmlPath = r"C:\\Users\10959\anaconda3\Lib\site-packages\cv2\data" + xmlPath
    classifier = cv.CascadeClassifier(xmlPath)
    
    while cap.isOpened():
        # 读取一帧数据
        ret, frame = cap.read()
        # 抓取不到视频帧,则退出循环
        if not ret:
            break
        # 显示方向
        frame = cv.flip(frame, 1)
        
        # 将当前帧图像转为灰度图像
        imgGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
         # 检测结果
        # 第一个参数是灰度图像
        # 第而个参数scaleFactor表示每次图像尺寸减小的比例
        # 第三个参数是人脸检测次数，设置越高，误检率越低，但是对于迷糊图片，我们设置越高，越不易检测出来
        # minSize为目标的最小尺寸
        # maxSize为目标的最大尺寸
        found = classifier.detectMultiScale(imgGray, scaleFactor=1.1, minNeighbors=3, minSize=minSize, maxSize=maxSize)
        
        # 框出结果
        if len(found) > 0:
            for foundRect in found:
                x, y, w, h = foundRect
                cv.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), bgr, 2)
        # 显示图像
        cv.imshow("Detection Result", frame)
        # 键盘Q键结束
        if cv.waitKey(10) & 0xFF == ord("q"):
            # 释放摄像头并关闭所有窗口
            cap.release()
            cv.destroyAllWindows()
            break
    # 释放摄像头并关闭所有窗口
    cap.release()
    cv.destroyAllWindows()
    
    print("Completed!!!")

# 调用 func
import cv2 as cv
print("Package Imported")

cascadeVideoPersonDetecionDraw("Data/objectDetectionTest.mp4", "\haarcascade_upperbody.xml", (0, 255, 0), (50, 50), (1000, 3000))
```