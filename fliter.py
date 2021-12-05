# 高斯滤波器作用原理和均值滤波器类似，都是去滤波器窗口中的像素的均值作为输出。
# 只不过均值滤波器的系数都是1，而高斯滤波器的模板系数会随着模板中心的增大而系数减小
# 滤波器算是对图片的预处理了吧，也不用padding
path = "dataset/baozi.jpg"
import numpy as np
import cv2
from PIL import Image
import math

# σ = 0.8 (标准差)应该就是方差吧‘
# 3 * 3 大小
def Gaussian_fliter(img):
    # pi = math.pi
    # std = 0.8
    # std = math.pow(std, 2)
    #
    # gaussian_array = np.zeros((3, 3), dtype=float)
    # for i in range(3):
    #     for j in range(3):
    #         gaussian_array[i, j] = math.exp((-1)*(math.pow((i-1), 2) + math.pow((j-1), 2))/2/std)
    # gaussian_array = gaussian_array/2/pi/std
    # print(gaussian_array)
    # print(np.sum(gaussian_array))
    # gaussian_array = gaussian_array/np.sum(gaussian_array)
    # print(gaussian_array)
    # print(np.sum(gaussian_array))
    gaussian_fliter = np.array( [[0.05711826, 0.12475775, 0.05711826],
                                [0.12475775, 0.27249597, 0.12475775],
                                [0.05711826, 0.12475775, 0.05711826]] )

    gaussian_array = np.zeros((img.shape[0],img.shape[1], img.shape[2]), dtype=float)
    for chan in range(3):
        for i in range(img.shape[0]-2):
            for j in range(img.shape[1]-2):
                jk = (img[i, j, chan]*gaussian_fliter[0,0] + img[i,j+1,chan]*gaussian_fliter[0,1] +img[i,j+2,chan]*gaussian_fliter[0,2] + img[i+1, j, chan]*gaussian_fliter[1,0] + img[i+1, j + 1, chan]*gaussian_fliter[1,1] + img[i+1, j + 2, chan]*gaussian_fliter[1,2] + img[i+2, j, chan]*gaussian_fliter[2,0] + img[i+2,j+1,chan]*gaussian_fliter[2,1] +img[i+2,j+2,chan]*gaussian_fliter[2,2])
                # 目前这种写法太冗余了，只是缓兵之计，等要求滤波器更多样在修改成范式吧
                gaussian_array[i+1,j+1,chan] = int(jk)
    # print(gaussian_array.shape)
    # img = Image.fromarray(gaussian_array.astype(np.uint8))
    # img.show()
    return gaussian_array

# 输入的是图片，输出的还是图片(还是数组吧)
# 目前的kernel是3*3的
def mean_fliter(img):
    # print(img.shape[0],img.shape[1], img.shape[2])
    mean_array = np.zeros((img.shape[0],img.shape[1], img.shape[2]), dtype=float)
    for chan in range(3):
        for i in range(img.shape[0]-2):
            for j in range(img.shape[1]-2):
                # print((img[i, j, chan] + img[i,j+1,chan] +img[i,j+2,chan] + img[i+1, j, chan] + img[i+1, j + 1, chan] + img[i+1, j + 2, chan] + img[i+2, j, chan] + img[i+2,j+1,chan] +img[i+2,j+2,chan])/9)
                # print(img[i, j, chan] , img[i,j+1,chan] ,img[i,j+2,chan] , img[i+1, j, chan] , img[i+1, j + 1, chan] , img[i+1, j + 2, chan] , img[i+2, j, chan] , img[i+2,j+1,chan],img[i+2,j+2,chan])
                jk = (img[i, j, chan]/9 + img[i,j+1,chan]/9 +img[i,j+2,chan]/9 + img[i+1, j, chan]/9 + img[i+1, j + 1, chan]/9 + img[i+1, j + 2, chan]/9 + img[i+2, j, chan]/9 + img[i+2,j+1,chan]/9 +img[i+2,j+2,chan]/9)
                # print(jk)
                mean_array[i+1,j+1,chan] = int(jk)
                # print(mean_array[i+1,j+1,chan])
                # mean_array[i + 1, j + 1, chan] += (img[i+1, j, chan] + img[i+1, j + 1, chan] + img[i+1, j + 2, chan])/3
                # mean_array[i+1,j+1,chan] += (img[i+2, j, chan] + img[i+2,j+1,chan] +img[i+2,j+2,chan])/3
    # print(mean_array)
    # print(mean_array.shape)
    # img = Image.fromarray(mean_array.astype(np.uint8))
    # img.show()
    return mean_array
