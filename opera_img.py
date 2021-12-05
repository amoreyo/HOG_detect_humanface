import numpy as np
import cv2
from PIL import Image
from skimage import io, color
import os

# 只是封装一下函数而已
# cv2.resize是双线性插值
def resize_img(img_array, size):
    return cv2.resize(img_array, (size, size))


# 返回np.array的灰度
def grey_img(array=None):
    # 包子的人脸是6*6 grid 比较合适
    # img = Image.open("baozi.jpg")
    # cv2.imread 直接返回np array
    img = array
    # rgb = io.imread("baozi.jpg") # io.imread 也是直接返回np array
    # print(type(rgb))

    # 从RGB转成lab，lab也是三通道，L，a，b
    # lab = color.rgb2lab(img)
    # print(lab[0][0]) # [69.19619304 20.24194333 57.47090844]
    # Lab颜色空间中的
    # L分量用于表示像素的亮度，取值范围是[0,100],表示从纯黑到纯白；
    # a表示从红色到绿色的范围，取值范围是[127,-128]；
    # b表示从黄色到蓝色的范围，取值范围是[127,-128]
    # print(lab.shape)
    # lab = Image.fromarray(lab)
    # lab.show()
    # 竟然报错了！！原因也是非 0~225


    # print(type(img))
    # img = np.array(img)
    img = img/np.max(img)*255
    # print(img.shape)
    img = img.astype('uint8')
    # print(img.shape)
    # print(img.shape)  # (248, 203, 3)
    # img = np.resize(img, (248, 203))
    # print(img.shape)
    img = Image.fromarray(img)
    img = img.convert('L')
    img = np.array(img)
    # print("the size of the grey img: ",img.shape)
    # TypeError: Cannot handle this data type: (1, 1, 3), <f8
    # 原因是要0~255的整数，而我传入了0~1的小数
    #
    # img = Image.fromarray(img)
    # img.show()
    return img

# 我的想法是，输入图片的名字(图片的前提是包含一个matrix)，所以是一个str类对吧
# 例如: Abhishek_Bachan_11_265,92,375,202.jpg
# 然后返回剪切好的np array
def crop_img(str):
    # imgs = os.listdir(path="F:\\desktop\\human_face_seg\\dataset\\dev")
    # print(imgs[1])
    imgs = Image.open(os.path.join("F:\\desktop\\human_face_seg\\dataset\\dev", str))
    # imgs.show()
    imgs = np.array(imgs)
    # print(imgs.shape)
    # print(imgs)
    a= str[:-4]
    b = a.split("_")
    c = b[-1]
    c = c.split(",")
    # print(c)# c是右，下的点，原点是左上角
    k = []
    for i in c:
        k.append(int(i))
    # print(k)
    # crop_array = np.zeros(shape=((k[3]-k[1]),(k[2]-k[0]),3),dtype=int)
    print("裁剪之后的大小： [h,w,3] " ,(k[3]-k[1]),(k[2]-k[0]),3)
    # 直接对imgs切片不就好了，sb
    # for j in range(crop_array.shape[0]):
    #     for l in range(crop_array.shape[1]):
    crop_imgs = imgs[k[1]:k[3], k[0]:k[2]]  # RGB格式
    # img = Image.fromarray(crop_imgs, 'RGB')
    # img.show()
    # return imgs[]
    return crop_imgs
