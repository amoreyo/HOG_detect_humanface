# 人脸分割日志

## 模拟过程： 

1. 输入图像(512×512)
2. 先把图片resize成同一规格
3. 把人脸灰度化（暂时来说，颜色对检测没有什么用）
4. 计算图像梯度
5. 写高斯滤波和平均滤波
6. 计算HOG特征向量
   1. 先得到每个cell的直方矩阵
    2. 2*2cell为一个block，经过block归一化（消除光照的影响）
    3. 将滑动得到的block堆叠，得到HOG特征向量
7. 得到HOG平均特征向量
    1. 最后决定裁剪的面部大小是128×128
8. 遗传算法
    1. 先切分图片为grid（M×N）
    2. 每个图片里面随机放置预先设定大小的anchor
    3. anchor算出HOG特征图，在所有grid中排序
    4. 筛选算法
        1. NMS，IOU小于阈值
    5. 以筛选出的anchor的中心为grid的中心，循环迭代step.2~5



## 需要改进的地方

定义函数的时候，建议都是同一输入np.array和输出np.array、
最基础的读取图片直接封装一个输入path的函数就ok了




## 碰到的error
1. RuntimeWarning: overflow encountered in ubyte_scalars像素加减运算溢出异常             
    用python处理图像时，可能会涉及两幅图像像素值之间的加减运算，这里需要注意的是图像像素值是ubyte类型，ubyte类型数据范围为0~255，若做运算出现负值或超出255，则会抛出异常


2. **前景提要**：
    我在写fliter，先写了mean的fliter，最后写出来，发现展示的图片好黑好黑，这显然是因为像素值太小所导致的，然后我去仔细看程序，奇妙的事情发生了  
    61 61 61 61 61 61 61 61 61  
    37
    37是61×9的和？？？想到像素的最高点是255，所以61×9-256-256 = 37  
    真相浮出睡眠
    结果：只要是 img[i, j, chan] 相加，程序会自动把超过255的减去256使之小于255，原因不详。  
    **解决方法**：就傻乎乎给每个像素值先除后加咯


3. requests.exceptions.TooManyRedirects: Exceeded 30 redirects.
    简单意思就是重定向太多了，造成的原因可能是我没接受他的cookies然后导致他一直重定向到他自己就形成了一个loop    
   **解决方案**：跳过呗，这么多图片还差你一张不成

    好像这样的图片还不止一张，那有相同的报错就直接跳过吧
   
4. cv2.error: OpenCV(4.5.4-dev) D:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\resize.cpp:4051: error: (-215:Assertion failed) !ssize.empty() in function 'cv::resize'
   应该是数据的问题，存在错误数据  
   **解决方案**: 跳过出错的图片并且把名字给保存一下
   
5. OSError: image file is truncated (1 bytes not processed)  
    这个问题其实超级常见啦，网上的很多教程只告诉你需要怎么做，但没有告诉你为何这么做。  
    这个truncated image是因为图片太大导致的，如果你去读ImageFile.py这个文件，会发现在文件顶部，有固定一个数值：  
    MAXBLOCK = xxxxxx(一个还蛮大的数字）  65536 64MB  
    **很奇怪的是我检查了自己的数据，最大就4.6MB**  
    这一行代码解释了为什么会报Image file is truncated这个错误——你传入的图片已经超过了MAXBLOCK限制的大小，PIL处理不了，必须要把这个图片删除一部分，所以raise了这个error。  
    如果你按照网上的办法，在你自己要跑的python脚本里设置：
    from PIL import ImageFile  
    ImageFile.LOAD_TRUNCATED_IMAGES = True  
    把LOAD_TRUNCATED_IMAGES设为true，实际上会导致你加载的图片少掉一部分，虽然在大数据中，偶尔一两张被裁掉没什么大问题，但是还是应当注意，不要暴力用才好。


6. IndexError: list index out of range  
   具体情景：  
   for i in range(len(list)):
        if():
            list.remove(list[i])  
   报错原因，for循环中的range(len(list))是动态变化的  
   因为list.remove(list[i])导致len(list)的变化

## 12.1
人脸的数据还再下载，速度及其缓慢
上午写了小脚本剔除无关数据
但实际上发现，直接将文件按大小排序手动删除效率也蛮高的
目前所有的数据是3559张图片

## 12.3
要完成对HOG数据的处理，先求平均值，再求一下方差，在把每个数据的方差可视化  
然后确定，对图片的分割，每个grid里面放几个anchor，anchor大小随机分布，然后每个anchor计算HOG，再算与标准HOG的差值，排序，对每个grid使用NMS

## 12.4
基础框架已经完全实现，但现在有个状况，就是epoch越多，anchor也越多  
我先尝试调大var  
没用，因为var不变，但是var一直在减小  
救命啊，运行的好慢好慢，我感觉8100的特征向量太大了  
而且效果好差，anchor都过小  （这个的原因应该是针对图片而言的，我的算法只是单一维度，所以anchor的大小有上下限，而yolo是多维度 ）  
var太主观了，应该要自适应  
测了相当多的图片，发现一个共同的问题 ，后来anchor都跑到了右下角落。  
现在先debug找出为什么都去了右下角  
再实现其他anchor的分布来优化  
1. 把所有anchors 类似k-mean分块， 分出的大块就是下一次的grid

### 12.14

在看卡方分布的时候，突然想到，数据的var分布有很多0，很不对劲啊，var是取平方了的值的和。猜想原因是特征向量的每个值都太小了，或者特征向量太大了。TODO： 先调试第二个方法