import numba as nb
import os
import requests
from PIL import Image
from PIL import ImageChops
from io import BytesIO
import cv2
import numpy as np
from decimal import Decimal
import pandas as pd
import matplotlib.pyplot as plt

# 剔除下载数据中None和显示错误的图片
def dele_Noneimg():
    path = "F:\\desktop\\human_face_seg\\dataset\\dev"

    imgs = os.listdir(path)
    # line = Image.open(os.path.join(path, "Julia_Stiles_37_228,378,606,756.jpg"))

    for img in imgs:
        img_path = os.path.join(path, img)
        judg = cv2.imread(img_path)
        # judge = Image.open(img_path)
        # differ = cv2.subtract(line, judg)
        # result = not np.any(differ)
        if judg is None:
            os.remove(img_path)
            continue
        # try:
        #     diff = ImageChops.difference(line, judge)
        #     if diff.getbbox() is None:
        #         os.remove(img_path)
        #         continue
        # except ValueError as e:
        #     print(e)
        #     continue
        # except OSError as o:
        #     print(o)
        #     os.remove(img_path)

def request_download():
    # os.makedirs('./dataset/image/', exist_ok=True)
    # 如果exist_ok是False（默认），当目标目录（即要创建的目录）已经存在，会抛出一个OSError
    dev_path = "./dataset/dev_urls.txt"
    eval_path = "./dataset/eval_urls.txt"
    proxies = {
        'https': 'https://10.253.87.117:2802',
        'http': 'http://10.253.87.117:2802'
    }
    count = 0
    with open(dev_path, "r") as da:
        data  = da.readlines()
        for line in data:
            count += 1
            if(count<=12945):
                continue # 前2行的数据不要
            line = line.strip('\n')
            contain = line.split()
            # url = contain[3]
            # trct = contain[4]
            print(contain[3])
            try:
                r = requests.get(contain[3], proxies = proxies)
            except: # (requests.exceptions.TooManyRedirects, requests.exceptions.SSLError, requests.exceptions.ProxyError) as e:
                # print(type(e),"::",e)
                # 直接接受所有报错然后跳过
                continue
            file_name = contain[0] + "_" + contain[1] + "_" + contain[2] + "_" + contain[4]
            with open(f'./dataset/dev/{file_name}.jpg', 'wb') as f:
                # print(type(r.content))
                # img = Image.open(BytesIO(r.content))
                # print(img.size)
                # img.show()
                # 想尝试直接裁剪requests得到的图片但是失败了
                # print(r.content)
                f.write(r.content)
                # break

def mean_HOG():
    count = 0
    path = "F:\desktop\human_face_seg\dataset\HOG_all.txt"
    mean_hog = np.zeros(shape=(8100), dtype=float)
    with open(path, 'r') as file:
        while(True):
            count += 1
            line = file.readline()
            if not line:
                break
            line_split = line.split(",")
            line_split = line_split[4:-1]
            # print(len(line_split[-1]))
            # 最后一个是空格
            for index, num in enumerate(line_split):
                num_float = num[:17]
                try:
                    ri_num = float(num_float)
                except ValueError:
                    ri_num = float(num)
                mean_hog[index] += ri_num
    mean_hog = mean_hog/count
    print("the final mean_hog is : ", mean_hog)
    return mean_hog
                # 午觉睡醒一想，没必要双精度这么打嘛，计算还慢
                # 而且，python的double的精度默认是64bit相当于17位数

# 计算方差，方便我查看数据的分布
def Var():
    path = "F:\desktop\human_face_seg\dataset\HOG_all.txt"
    mean_path = "F:\desktop\human_face_seg\dataset\HOG_mean.txt"
    mean_hog = np.zeros(shape=(8100), dtype=float)
    mean_array = np.zeros(shape=(8100),dtype=float)
    with open(mean_path, 'r') as mean_file:
        mean_data = mean_file.readlines()
        i=0
        for line in mean_data:
            mean_array[i] = float(line)
            i+=1

    var_array = np.zeros(shape=(3592),dtype=float)
    with open(path, 'r') as file:
        i=0
        # print(len(line)) # 3592行数据在file里面
        while(True):
            # count += 1
            line = file.readline()
            if not line:
                break
            line_split = line.split(",")
            line_split = line_split[4:-1]
            var_all = 0
            for index, num in enumerate(line_split):
                num_float = num[:17]
                try:
                    ri_num = float(num_float)
                except ValueError:
                    ri_num = float(num)
                var_value = pow(ri_num-mean_array[index],2)
                var_all += var_value
            var_all = pow(var_all, 1/2)
            var_array[i] = var_all
            i += 1
    return var_array

def main_Var(block_array):
    var_all = 0
    mean_path = "F:\desktop\human_face_seg\dataset\HOG_mean.txt"
    # mean_hog = np.zeros(shape=(8100), dtype=float)
    mean_array = np.zeros(shape=(8100),dtype=float)
    with open(mean_path, 'r') as mean_file:
        mean_data = mean_file.readlines()
        i=0
        for line in mean_data:
            mean_array[i] = float(line)
            i+=1
    for j in range(8100):
        var_value = pow(block_array[j] - mean_array[j], 2)
        var_all += var_value
    var_all = pow(var_all, 1 / 2)
    return var_all



# 数据方差的可视化
def var_plot():
    path = "F:\desktop\human_face_seg\dataset\HOG_var.txt"
    var_array = np.zeros(shape=(3592), dtype=float)
    with open(path, 'r') as var_file:
        var_data = var_file.readlines()
        i = 0
        for line in var_data:
            var_array[i] = float(line)
            i += 1
    print(var_array.shape)

    plt.figure("lena")  # 定义了画板
    arr = var_array.flatten()  # 若上面的array不是一维数组，flatten()将其变为一维数组，是numpy中的函数
    # hist函数可以直接绘制直方图
    # 参数有四个，第一个必选
    # arr: 需要计算直方图的一维数组
    # bins: 直方图的柱数，可选项，默认为10
    # normed: 是否将得到的直方图向量归一化。默认为0
    # facecolor: 直方图颜色
    # alpha: 透明度
    # 返回值为n: 直方图向量，是否归一化由参数设定；bins: 返回各个bin的区间范围；patches: 返回每个bin里面包含的数据，是一个list
    n, bins, patches = plt.hist(arr, bins=256, facecolor='green', alpha=0.75)
    plt.show()

# request_download()
# crop_img("Kevin_Bacon_17_1168,636,1706,1174.jpg")
# mean_HOG()
# var_array  = Var()
# np.savetxt("F:\desktop\human_face_seg\dataset\HOG_var.txt", var_array, fmt='%f', delimiter=',')
#
# var_plot()