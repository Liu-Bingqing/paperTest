# 修改 - 上海理工大学学报
## 引言部分
引言：相关工作中提到的三个方向 - 在实验中需要进行对比实验
背景差分方法 改进ViBe - 背景建模 - LBSP
特征提取方法 - 基于骨骼点提取方法 - 与
深度学习方法 - 卷积神经网络CNN - SiamFc全卷积孪生神经网络/YOLO/STARS时空卷积模型 换个

本文方法优势：
本文方法融合特征提取和背景差分原理，比单一使用特征提取法和背景差分法更具稳定性，提取准确性更高。相较于深度学习方法更轻量级，算法整体耗时和算力消耗更小，更具应用性。
本文方法在对比三个前景基于背景提取方向的方法对背景动态干扰较为敏感，算法精准性较差


fixed:
第一个方向是基于背景建模原理，文献[]引入LBSP算子提出一种融合多特征的 ＶｉＢｅ背景建模改进算法有效分割视频动态目标。第二个方向基于提取特征信息训练算法，文献[]输入大量跌倒姿态数据训练YOLOv4初步框选目标后使用自顶而下的Alphapose骨骼姿态识别框架提取关节点信息，作为人体目标行为分析的动态特征变化数据。第三个方向基于深度学习算法，文献[]使用YOLOv3检测目标后建立时空三维卷积神经网络根据视频帧识别视频动态前景目标。文献[]提出基于生成对抗网络（GAN）结构模型，基于视频帧序列捕捉运动中时空变化属性来识别提取视频运动目标信息特征实现动态目标识别。


全域视频特征信息作为算法输入数据极大降低算法识别的效率和准确性，室内监控视频中正常动态行为运动速率缓慢和目标遮挡是动态前景目标识别的难点之一。针对以上问题本文提出特征定位与改进帧差法融合算法减少背景静态区域冗余特征和干扰源对提取视频前景目标特征的影响。本文主要贡献有：1) 融合像素级降噪方法和消除阴影方法进行预处理，解决背景差值建模方法识别结果出现鬼影问题、抖动噪声干扰问题和数据预处理耗时问题。2) 引入SIFT算法特征点匹配和帧序列视频图像特征点定位，从时序特征点动态变化角度生成差分图像，解决基于骨骼点姿态识别框架由于遮挡目标前景目标提取精确性低问题。3) 使用自适应阈值原理改进帧差法，依据视频前景动态目标运动变化使用最大类间方差法(OTSU)算法改进传统帧差法差分图像阈值，解决基于深度学习的动态目标前景识别因训练数据量少导致过拟合在测试数据中识别运动缓慢目标运动误判为背景问题。4) 使用连续多帧视频图像作为算法输入图像，增加图像逻辑运算和图像形态学变换改进帧差算法，解决视频动态前景目标识别完整度低问题。本文主要针对室内单摄像机监控视频下动态前景目标识别提取进行分析研究，对室外监控视频的人体动态行为识别情况不予考虑。


## 本文模型
特征提取是目标提取识别的重要步骤之一，是影响目标提取准确性的关键操作
许多研究使用骨架节点特征提取方法完成视频特征信息点提取，骨架特征提取分为[介绍]，优势+劣势。本文受特征提取方法提取动态目标方法原理启发，使用轻量级SIFT方法提取帧间特征信息，减少算法耗时并且减少硬件消耗，更适合家用摄像机设备硬件算法承载力。
本文融合特征提取和背景提取方法的原理，

特征信息定位是动态前景提取算法的重要步骤之一，是影响动态前景目标提取准确性的关键操作。近几年研究将骨骼关节点定位框架与深度学习算法结合实现视频动态前景目标提取，骨骼关节点特征定位成为视频动态前景目标特征信息提取主流方法，但室内视频中易出现障碍物遮挡导致关节点定位准确性和完整性下降，并且特征定位框架需要大量人体姿态数据训练提高特征信息表达精准度，因此基于骨骼关节点姿态定位框架设计的动态前景目标提取算法过于理想化难以应用于复杂监控视频环境。本文使用轻量级SIFT方法提取帧间特征信息，减少算法整体耗时及硬件算力消耗，更适合家用摄像机设备硬件算法承载力。本文引入SIFT特征点匹配算法提取连续单帧图像像素级特征值信息并进行特征点匹配，依据特征点匹配结果对帧序列连续单帧图像校正对齐。特征匹配差异在目标运动区域明显，在背景区域特征信息匹配差值小，基于时间一致性分析全域特征信息匹配对齐结果后消除背景区域像素信息可以减少视频数据大面积静态背景区域，精简目标特征点信息能降低冗余特征信息干扰本文。

## 实验
本文使用UR Fall Detection数据集[14]的视频序列作为HFID算法有效性实验数据集，每一帧图像分辨率为640像素×480像素，使用通用数据集CDnet 2014[15]作为算法多场景实用性实验数据集，视频每帧图像分辨率为320像素×240像素。实验的基本硬件配置：CPU是 Intel(R) Core(TM) i5-1135G7 2.42 GHz；内存是16G、Windows 64位的操作系统；实验使用Python库。
### 算法耗时分析 表1
本文算法HFID - 改进ViBe（背景提取算法) - 基于骨骼点（特征提取方法）Alphapose - CNN

改进ViBe[3]、骨骼点相关Alphapose[4]、深度学习方法（卷积神经网络）CNN[5]、HFID[本文算法]
本文选取动态目标前景提取算法三个方向算法与HFID算法比较在室内监控视频场景下动态目标提取总耗时，对比改进ViBe算法[3]、Alphapose框架[4]、CNN算法[5]和HFID算法的总运行时间如表1所示

由表1看出图像分辨率越大算法耗时越长。HFID算法与背景差值建模改进ViBe算法总体耗时均低于1ms,相较于深度学习方法算力要求较低。骨骼关节点框架Alphapose框架和CNN算法需要对输入数据进行预标注和正常非正常数据分类，提取动态前景目标运行总耗时更久。基于深度学习的视频动态前景目标提取算法的准确率达到75%以上需要更多算力和时间进行调参训练，因此HFID比基于深度学习的视频动态前景目标提取算法应用性更强，时间资源更小。

### 算法性能对比 表2
本文算法 - 经典帧差法 - 改进ViBe（背景提取) - GMM（背景建模） -Alphapose 基于骨骼点（特征点提取方法） - CNN - GAN
本文选取基于背景差值算法的动态前景提取目标算法经典帧差法[14]、改进ViBe算法[3]、基于背景建模的高斯混合模型(GMM)算法[19]、基于Alphapose关节点姿态识别框架的动态目标提取算法[4]、CNN动态前景目标算法[5]和基于无监督学习(GAN)的动态前景目标提取算法[6]作为对照算法，与HFID在UR Fall Detection室内监控视频跌倒行为场景下进行性能对比实验，算法性能结果如表2。

从表2看出

### 基于UR Fall Detection数据的算法结果对比 图
灰度图 - 改进ViBe - GMM - Alphapose - CNN - GAN - HFID
本文算法 - LBSP（经典帧差法（背景提取）） - GMM（背景提取模型）- 基于骨骼点的YOLO(特征点提取方法) - 基于STARS（时空卷积神经网络） - 基于无监督学习

### 不同场景下提取结果  图
灰度图 - 改进ViBe - GMM - Alphapose - CNN - GAN - HFID

深度学习算法的适应性较强，但大量数据准备工作耗时较多，为了增加算法的精确性需要扩充数据数量并进行特征点标记，并且一般家用监控器的算力无法满足深度学习算法的算力要求，算法应用性较低。

## 参考文献
###  1 引言 [1-6]
[1] Ghatak, Subhankar, Rup, et al.GAN based efficient foreground extraction and HGWOSA based optimization for video synopsis generation[J].Digital Signal Processing,2021,111,102988.
[2] Tan Jia-wai, Ding Qi-chuan, Bai Zhong-yu.Optimal Estimation Method of 3-Dimensional Human Pose Based on Video Frame Coherent Information[J]. Robot,2021,43(01): 9-16.**中文**
[2] 谭嘉崴,丁其川,白忠玉.基于视频帧连贯信息的3维人体姿势优化估计方法[J].机器人,2021,43(01):9-16.
[3] 杨春德,孟琦.一种融合ViBe与多特征提取的微动目标检测算法[J].计算机科学,2017,44(02):309-312+316.**中文** -> 改进ViBe
[4] Xiaodong Zhao,Fanxing Hou, Jingfang Su, et al. An Alphapose-Based Pedestrian Fall Detection Algorithm[J]. Lecture Notes in Computer Science,2022, https://doi.org/10.1007/978-3-031-06794-5_52. —> Alphapose
[5] Chuanwang Chang, Chuanyu Chang,Youying Lin. A hybrid CNN and LSTM-based deep learning model for abnormal behavior detection[J]. MULTIMEDIA TOOLS AND APPLICATIONS,2022,81: 11825–11843. -> CNN
[6] Weiwen Hsu, JingMing Guo, Chienyu Chen,et al. Fall Detection with the Spatial-Temporal Correlation Encoded by a Sequence-to-Sequence Denoised GAN[J].Sensors,2022,22(11):4194. -> GAN
### 2 整体设计 [7-12]
[7][原5] Sheng Mengxue, Hou Wanwan, Jiang Juchao. Implementation of Accelerating Video Preprocessing based on ZYNQ Platform Resource Management[J].Journal of Physics Conference Series,2020,1544(1):012112. -> 运动阴影
[8][原6] Di Wu, Chuanjiong Zhang, Li Ji, et al. Forest fire recognition based on feature extraction from multiview images[J].Traitement du Signal,2021,38:775-783. -> HSV颜色空间
[9][原7] Pei Song-wen, Fan Jing, Shen Tian-ma, et al.Research on Denoising Low Dose CT Image Using an Advanced Generative Adversarial Network with Multiple Generators[J].Journal of Chinese Computer Systems,2020,41(12):2582-2587.**中文** -> TriGan
[9] 裴颂文,樊静,沈天马,顾春华.面向低剂量CT图像的多生成器对抗网络降噪模型的研究[J].小型微型计算机系统,2020,41(12):2582-2587.
[10][原8] Pu Han, Tianqiang Huang, Bin Weng, et al. Overcome the Brightness and Jitter Noises in Video Inter-Frame Tampering Detection[J].Sensors,2021,21(12):3953. -> 抖动噪声
[11][原9]  Arnal, Josep, Luis Súcar.Hybrid Filter Based on Fuzzy Techniques for Mixed Noise Reduction in Color Images[J]. Applied Sciences,2020,10(1):243. -> 高斯函数
[12][原10] Weixing Wang, Limin Li, Fei Zhang.Crack image recognition on fracture mechanics cross valley edge detection by fractional differential with multi-scale analysis[J].SIViP,2022,DOI:10.1007/s11760-022-02202-6. -> 灰度化模型
### 3 本文算法 [13-15]
[13][原12] Liang Zheng, Yi Yang, Qi Tian. SIFT Meets CNN: A Decade Survey of Instance Retrieval[J].IEEE Transactions on Pattern Analysis and Machine Intelligence,2018,40(5):1224-1244.-> SIFT
[14][原3] Chen Quan, Huang Jun, Xu Fang. Research on improved visual background extraction algorithm in indoor[J]. Journal of Chinese Computer Systems,2021,42(06):1250-1255.**中文** -> 经典帧差法
[14]陈权,黄俊,徐访.改进视觉背景提取算法在室内的研究[J].小型微型计算机系统,2021,42(06):1250-1255. 
[15][原13]Xing Jiangwa, Yang Pei, Qingge Letu.Automatic thresholding using a modified valley emphasis[J]. IET image processing,2020,14(3):536-544. -> OTSU
### 4 实验 [15]

