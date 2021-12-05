import numba as nb
import numpy as np
from opera_img import grey_img, resize_img
from PIL import Image
import math
from fliter import mean_fliter
import cv2
# 传入的是np.array
# 输出的是每个算子计算出的梯度
# padding = 1， stride = 1
# padding的方法： np.insert(a,1,[1,1,1,1],0)  | numpy.insert(arr,obj,value,axis=None)
# arr:为目标向量
# obj:为目标位置
# value:为想要插入的数值
# axis:为插入的维度
# padding = False
stride = 1
kernel_size = 3
path = "dataset/baozi.jpg"

Sx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
Sy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
# 听说是更好的算子
Sx_up = np.array([[-3,0,3],[-10,0,10],[-3,0,3]])
Sy_up = np.array([[3,10,3],[0,0,0],[-3,-10,-3]])

# 这里采用的padding是复制最近的row和col
def pad(grey_array, kernel_size):
    print(grey_array.shape)
    padding_size = int((kernel_size-1)/2)
    # print(grey_array[0:1])
    # print(np.squeeze(grey_array[-1:]))
    grey_array = np.insert(grey_array, 0, np.squeeze(grey_array[:padding_size], 0), 0)
    grey_array = np.insert(grey_array, -1, np.squeeze(grey_array[-padding_size:], 0), 0)
    print("padding row: ")
    print(grey_array.shape)
    # print(grey_array[::, [-4,-3,-1]]) # 这个操作属实有点神奇
    # 重点是，这种操作他自动append_dim(axis=1) , 反而变烦了
    if(padding_size == 1):
        grey_array = np.append(grey_array, np.expand_dims(grey_array[::, -1], axis=1), 1)
        grey_array = np.append(np.expand_dims(grey_array[::, 0], axis=1), grey_array, 1)
        print("padding col: ")
        print(grey_array.shape)
        return grey_array
    else:
        pado = []
        for i in range(padding_size):
            pado.append((i+1)*(-1))
        grey_array = np.append(grey_array, grey_array[::, pado], 1)
        pado = []
        for i in range(padding_size):
            pado.append(i)
        grey_array = np.append(grey_array[::, pado], grey_array, 1)
        print("padding col: ")
        print(grey_array.shape)
        return grey_array

@nb.jit()
def conv(grey_array, pad_array, Sx, stride, padding, kernel_size):
    # 从每一行这样开始遍历走stride吧
    conv_array = np.zeros((grey_array.shape[0], grey_array.shape[1]),dtype=float)
    # padding 肯定会padding的呀
    # if(padding):
    #     padding_size = int((kernel_size-1)/2)
    # else:
    #     padding_size = 0
    for i in range(grey_array.shape[0]):
        for j in range(grey_array.shape[1]):
            conv_array[i,j] += pad_array[i,j]*Sx[0,0] + pad_array[i,j+1]*Sx[0,1] + pad_array[i,j+2]*Sx[0,2]
            conv_array[i,j] += pad_array[i+1,j]*Sx[1,0] + pad_array[i+1,j+1]*Sx[1,1] + pad_array[i+1,j+2]*Sx[1,2]
            conv_array[i,j] += pad_array[i + 2, j] * Sx[2, 0] + pad_array[i + 2, j + 1] * Sx[2, 1] + pad_array[i + 2, j + 2] * Sx[2, 2]

    return conv_array


# np.arctan(1) = pi/4 = 0.78539
# np.arctan 返回的是弧度值，取值范围应该在 (-pi/2 ~ pi/2)
# 之后考量一下np.arctan2,它取的范围是-180~180
# 明确一下梯度函数究竟要返回什么
# 应该是返回角度 准确来说是弧度值
@nb.jit()
def grad(conv_array_x, conv_array_y):
    grad_array = np.zeros((conv_array_x.shape[0], conv_array_x.shape[1]), dtype=float)
    for i in range(conv_array_x.shape[0]):
        for j in range(conv_array_x.shape[1]):
            if(conv_array_y[i, j] == 0):
                grad_array[i, j] = math.pi/2
            else:
                grad_array[i, j] = np.arctan(conv_array_x[i, j] / conv_array_y[i, j])
    return grad_array


# sobel_xy 算子主要计算出梯度的方向和幅值
@nb.jit()
def sobel_xy(grey_array, padding = True):
    if(padding):
        # pad_array = pad(grey_array, kernel_size)
        # 没错我写的padding太费时间拉，倒不如直接cv2.resize嘿嘿
        pad_array = cv2.resize(grey_array, (130,130))
    else:
        pad_array = grey_array
    # steps = grey_array.h * w
    # 返回一个np数组，存x方向上的梯度
    conv_array_x = conv(grey_array, pad_array, Sx, stride, padding, kernel_size)
    conv_array_y = conv(grey_array, pad_array, Sy, stride, padding, kernel_size)

    grad_array = grad(conv_array_x, conv_array_y)
    amp_array = (conv_array_x**2 + conv_array_y**2)**0.5
    return grad_array, amp_array
    # amp_array 存储了梯度的幅值
    # grad_array 存储了梯度的方向
    # img = Image.fromarray(grad_array)
    # img.show()



# def main():
#     img_array = np.array(Image.open(path)) # RGB
#     img_array = resize_img(img_array)
#     # print(img_array.shape)
#     # img_array = cv2.imread(path) # BGR
#     grad_array , amp_array = sobel_xy(grey_img(mean_fliter(img_array)))
#     # print(grad_array.shape)
#     # img = Image.fromarray(amp_array)
#     # img.show()
