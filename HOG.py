import numba as nb
import numpy as np
from opera_img import grey_img, resize_img
from PIL import Image
import math
from fliter import mean_fliter
import cv2

# 这步操作，相当于凝练特征，以减小噪声的干扰
# 得到每个cell的直方图
# 标准图像大小的默认是256*256
# 每个cell为8*8像素
# 所以得到的cell是32*32
# 因为每个cell由九个数值组成，所以cell的大小为 32*32*9
# 9个数值对应着角度，分别是 0、 20、 40、 60....160
# 因为grad_array 的取值范围为[-pi/2, pi/2]
def Historode(grad_array, amp_array):
    pi = math.pi
    # print(amp_array.shape)
    cell_array = np.zeros((int(grad_array.shape[0]/8), int(grad_array.shape[1]/8), 9), dtype=float)
    # angle_array = grad_array + pi/2
    # # angle_array [0, pi]
    # angle_array = angle_array/pi*180
    # 感觉这么做，没必要，我直接分割弧度不就好了吗

    # 这里我要做的是，将弧度值映射到数组的index中去
    # 比较然后寻找，虽然只有9个比较的例程，但是分而治之的算法会剩下很多时间，最多比较3次
    # 实际上划分成9等分也是超参数，在精确的前提上进行泛化
    # split_angle = [-pi/2, -7*pi/9, -5*pi/9, -pi/3, -pi/9, pi/9, pi/3, 5*pi/9, 7*pi/9]
    split_angle = [-pi/2 + i*(pi/9) for i in range(9)]
    # print(split_angle)
    # 比较的话也不是前面说的3等分了哟，但是，先实例成4次吧
    # 没办法啊，暂时不用递归实现
    for l in range(cell_array.shape[0]):
        for b in range(cell_array.shape[1]):
            for i in range(8):
                for j in range(8):
                    i_ = i + 8*l
                    j_ = j + 8*b
                    if(grad_array[i_, j_] > split_angle[4]):
                        if(grad_array[i_, j_] > split_angle[6]):
                            if(grad_array[i_, j_] > split_angle[7]):
                                if(grad_array[i_, j_] >= split_angle[8]): # [8, 9]
                                    cell_array[l, b, 8] += amp_array[i_, j_]
                                else: # (7, 8)
                                    jk = amp_array[i_, j_]*((grad_array[i_, j_] - split_angle[7])/(pi/9))
                                    cell_array[l, b, 8] += jk
                                    cell_array[l, b, 7] += amp_array[i_, j_] - jk
                            else: # (6, 7]
                                jk = amp_array[i_, j_]*((grad_array[i_, j_] - split_angle[6])/(pi/9))
                                cell_array[l, b, 7] += jk
                                cell_array[l, b, 6] += amp_array[i_, j_] - jk
                        else: # (4, 6]
                            if(grad_array[i_, j_] >= split_angle[5]):
                                jk = amp_array[i_, j_]*((grad_array[i_, j_] - split_angle[5])/(pi/9))
                                cell_array[l, b, 6] += jk
                                cell_array[l, b, 5] += amp_array[i_, j_] - jk
                            else:
                                jk = amp_array[i_, j_]*((grad_array[i_, j_] - split_angle[4])/(pi/9))
                                cell_array[l, b, 5] += jk
                                cell_array[l, b, 4] += amp_array[i_, j_] - jk
                    else: # [0, 4]
                        if(grad_array[i_, j_] >= split_angle[2]):
                            if(grad_array[i_, j_] >= split_angle[3]):
                                jk = amp_array[i_, j_] * ((grad_array[i_, j_] - split_angle[3]) / (pi / 9))
                                cell_array[l, b, 4] += jk
                                cell_array[l, b, 3] += amp_array[i_, j_] - jk
                            else: # [2, 3)
                                jk = amp_array[i_, j_] * ((grad_array[i_, j_] - split_angle[2]) / (pi / 9))
                                cell_array[l, b, 3] += jk
                                cell_array[l, b, 2] += amp_array[i_, j_] - jk
                        else:
                            if(grad_array[i_, j_] >= split_angle[1]):
                                jk = amp_array[i_, j_] * ((grad_array[i_, j_] - split_angle[1]) / (pi / 9))
                                cell_array[l, b, 2] += jk
                                cell_array[l, b, 1] += amp_array[i_, j_] - jk
                            else: # [0, 1]
                                cell_array[l, b, 8] += amp_array[i_, j_]

        # 写着是有点多，但是计算量相对少
        # 写着好冗余啊，想个递归吧(臭笨逼，下次写代码先写伪代码)
    return cell_array


# block 归一化
# 图像的梯队对整体光照非常敏感，所以需要将直方图 归一化
# 滑动-归一化-拼接
@nb.jit()
def Block(cell_array):
    block_array = np.array([])
    block_feature = np.array([])
    for i in range(cell_array.shape[0]-1):
        for j in range(cell_array.shape[1]-1):
            block_array = np.array([])
            for l in range(2):
                for b in range(2):
                    block_array = np.append(block_array, cell_array[i+l, j+b])
                    # print(block_array.shape)
            jk = np.sqrt(np.sum(np.power(block_array, 2)))
            if(jk==0):
                # print("a?")
                block_array = block_array*0
            else:
                block_array = block_array/np.sqrt(np.sum(np.power(block_array, 2)))
            block_feature = np.append(block_feature, block_array)
    return block_feature
