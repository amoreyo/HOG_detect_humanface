import numba as nb
import numpy as np
import os
import PIL
import math
import random
from Sobel import sobel_xy
from opera_img import grey_img, resize_img, crop_img
from fliter import mean_fliter, Gaussian_fliter
from HOG import Historode, Block
from PIL import Image, ImageDraw
import time
import cv2
from opera_data import main_Var
from operator import itemgetter, attrgetter

#----------------这是一个流程，img_array是切割好的anchor----------------#
# img_array 是单通道，大小为 128*128, np_array
def HOG_process(img_array):

    grad_array, amp_array = sobel_xy(img_array)
    cell_array = Historode(grad_array, amp_array)
    block_array = Block(cell_array)
    return block_array
#-------------------------------------------------------------------#

@nb.jit()
def read_anchor(anchor_path):
    # 打开anchor文件并读取
    f = open(anchor_path, "r")
    anchor_info = f.readline()
    anchor_infos = anchor_info.split(";")
    percent = int(anchor_infos[0])
    anchors = []
    one = True
    for i in anchor_infos:
        if(one):
            one = False
            continue

        ratio = i.split(":")
        ratio[0] = int(ratio[0])
        ratio[1] = int(ratio[1])
        anchors.append(ratio)
    anchors = np.array(anchors)
    return percent, anchors

# true_anchors_size是对不同的grid来说的
@nb.jit()
def anchor_size(grid_size, percent, anchors):
    small_size = grid_size*percent/100
    true_anchors_size = []
    for anchor in anchors:
        divide = max(anchor)
        true_anchor = [small_size/divide*anchor[0], small_size/divide*anchor[1]]
        true_anchors_size.append(true_anchor)
    # true_anchors_size = np.array(true_anchors_size)
    return true_anchors_size

def grid_and_anchors(all_grids_anchors, img_array):
    # origin_anchors存放var值和对应的anchor信息
    all_anchors = []
    # for grid in grids:
        # 需要得出每个anchor在图上的具体位置，右上角坐标和大小
        # 先确定anchor的右上角的坐标
    # all_grid_anchors 的组成是：
    # [[[0, 0], 256], [[204.8, 204.8], [136.53333333333333, 204.8],
    # [204.8, 136.53333333333333], [153.60000000000002, 204.8],
    # [204.8, 153.60000000000002], [146.2857142857143, 204.8],
    # [204.8, 146.2857142857143]]]
    all_anchors_split_grids = []
    for all_grid_anchors in all_grids_anchors:
        # origin_anchor = []
        for anchor_size in all_grid_anchors[1]:
            anchor_size[0] = int(anchor_size[0])
            anchor_size[1] = int(anchor_size[1])
            grid_wh = all_grid_anchors[0][1]
            h = grid_wh * random.random()
            w = grid_wh * random.random()
            #----------这样写while来防止越界会导致anchor分配的不均匀------------------------#
            # 解决方案：不重新random直接剪切(错，我们先理清楚，是什么均匀分布在grid中，应该是anchor的中心)
            # anchor_cor = all_grid_anchors[0][0][0] + 平均随机（grid的大小）
            # if (all_grid_anchors[0][0][0] + h - all_grid_anchors[0][1]/2> 512):
            #     h = h - all_grid_anchors[0][1]/2 # all_grid_anchors[0][1] * random.random()
            # if (all_grid_anchors[0][0][0] + w - all_grid_anchors[0][1]/2> 512):
            #     w = w - all_grid_anchors[0][1]/2 # all_grid_anchors[0][1] * random.random()
            #——----------------------------------------------------------------------#
            # H 和 W 是anchor_cor-1/2(anchor_size)
            H = int(all_grid_anchors[0][0][0] + h - 0.5 - anchor_size[0]/2)
            W = int(all_grid_anchors[0][0][1] + w - 0.5 - anchor_size[1]/2)
            # anchor_cor 不能小于gird的右上角(肯定的)，不能大于左下角(也是肯定的)，要是超过就裁剪
            # all_grid_anchors[0][0][0] + 平均随机（grid的大小）>
            if(h - anchor_size[0] / 2 < 0):
                H = int(all_grid_anchors[0][0][0])
            if(w - anchor_size[1] / 2 < 0):
                W = int(all_grid_anchors[0][0][1])
            if(h + anchor_size[0] / 2 > grid_wh):
                H = int(all_grid_anchors[0][0][0] + grid_wh - anchor_size[0] / 2) # 这时候anchor_size也要修改了
                anchor_size[0] = int(anchor_size[0] / 2-0.5)# int((grid_wh + all_grid_anchors[0][0][0] - H) * 2)
            if(w + anchor_size[1] / 2 > grid_wh):
                W = int(all_grid_anchors[0][0][1] + grid_wh - anchor_size[1] / 2)
                anchor_size[1] = int(anchor_size[1] / 2-0.5)# int((grid_wh + all_grid_anchors[0][0][1] - W) * 2)

            origin_anchor = [H, W]
            # print("origin_anchor : ", origin_anchor)
            # print("anchor_size : ", anchor_size)
            # origin_anchors.append(origin_anchor)
            img_array_anchor = img_array[origin_anchor[0]:(origin_anchor[0]+anchor_size[0]), origin_anchor[1]:(origin_anchor[1]+anchor_size[1])]
            # np的resize操作是直接复制黏贴啥的，我好不放心啊，淦，那直接cv2.resize不就好了
            # print(img_array_anchor)
            # try:
            img_array_anchor = cv2.resize(img_array_anchor, dsize=(128, 128),interpolation=cv2.INTER_NEAREST)
            # except:
            #     print(img_array_anchor)
            #     print(origin_anchor)
            block_array = HOG_process(img_array_anchor) # padding太费时了
            # 计算该Block_array与mean_arry的平方差
            var = main_Var(block_array)
            # 方差，右上位置，高宽
            an_anchor = [var, origin_anchor, anchor_size]
            all_anchors.append(an_anchor)
        all_anchors_split_grids.append([all_anchors, all_grid_anchors[0]])
    return all_anchors_split_grids
        # img_array_anchor = img_array()

# NMS问题很大啊
#------我改了多层次之后就回复成每个grid的NMS，取消全局NMS---------#
# all_anchors: [[all_anchors_split],[all_anchors_split]]
# all_anchors_split: [[anchors],[grid]]
# anchors: [[var, origin_anchor, anchor_size],[]]
@nb.jit()
def NMS(all_anchors):
    grid_anchors = []
    for all_anchors_split in all_anchors:
        anchors_spilt = all_anchors_split[0]
        grid_spilt = all_anchors_split[1]
        anchors_spilt.sort(key=itemgetter(0))  # 这竟然是可行的，真好啊python
        # 排序完之后是升序
        # 直接删掉高于阈值（13）的
        # ----我有理由相信我见鬼了------ #
        length = len(anchors_spilt)
        copy = anchors_spilt
        var_stand = anchors_spilt[int(length/6-0.1)][0]
        #----------用IOU的阈值一棒子打死太过头了，本身我这个算法是对HOG进行计算，而且视野还在不断下降-----------#
        for i in range(length):
            # print(all_anchors[i][0])
            # File "F:/desktop/human_face_seg/main.py", line 103, in NMS
            #     print(all_anchors[i][0])
            # IndexError: list index out of range
            if(copy[length-1-i][0] >= var_stand):
                anchors_spilt.remove(anchors_spilt[length-1-i])
        print(var_stand)
        print(anchors_spilt)
        # print(all_anchors)
        # print(all_anchors)
        # IOU: >75%, <0%
        # print(all_anchors)
        # print(len(all_anchors)) # 4×7=28
        while(len(anchors_spilt)>0):
            grid_anchors.append(anchors_spilt[0])
            anchors_spilt.remove(anchors_spilt[0])
            # 计算IOU
            del_anchors = []
            index = 0
            for anchor in anchors_spilt:
                anchor_H = anchor[1][0]+anchor[2][0]
                anchor_W = anchor[1][1]+anchor[2][1]
                grid_anchor_H = grid_anchors[-1][1][0] + grid_anchors[-1][2][0]
                grid_anchor_W = grid_anchors[-1][1][1] + grid_anchors[-1][2][1]

                base_area  = grid_anchors[-1][2][0] * grid_anchors[-1][2][0]
                count = 0
                for i in range(grid_anchors[-1][1][0], grid_anchor_H):
                    for j in range(grid_anchors[-1][1][1], grid_anchor_W):
                        if(i>=anchor[1][0] and i<=anchor_H and j >=anchor[1][1] and j<=anchor_W):
                            count += 1
                        else:
                            continue
                IOU = count/base_area*100
                # 只要合并一次就够了
                # 不够
                if(IOU >= 70):
                    max_H = max(grid_anchor_H, anchor_H)
                    max_W = max(grid_anchor_W, anchor_W)
                    grid_anchors[-1][1][0] = min(grid_anchors[-1][1][0], anchor[1][0])
                    grid_anchors[-1][1][1] = min(grid_anchors[-1][1][1], anchor[1][1])
                    grid_anchors[-1][2][0] = max_H - grid_anchors[-1][1][0]
                    grid_anchors[-1][2][1] = max_W - grid_anchors[-1][1][1]
                    del_anchors.append(index)
                index += 1
            x = len(del_anchors)
            for j in range(x):
                anchors_spilt.remove(anchors_spilt[del_anchors[x-1-j]])
    return grid_anchors
# 虽然报错了，但我们假装自己是正确的，接着写下去吧

@nb.jit()
def anchor2grid(grid_anchors):
    grids = []
    for grid_anchor in grid_anchors:
        or_point_h = grid_anchor[1][0] + grid_anchor[2][0] / 2
        or_point_w = grid_anchor[1][1] + grid_anchor[2][1] / 2
        bian = min(grid_anchor[2][0],grid_anchor[2][1]) # 如果是min的话就不用约束 是否超过边界了
        grid = [[or_point_h-bian/2, or_point_w-bian/2], bian]
        grids.append(grid)
    return grids

@nb.jit()
def img_draw(path, all_anchors, size):
    draw_size = 512
    img = Image.open(path)
    img = img.resize((draw_size,draw_size))
    draw = ImageDraw.Draw(img)
    print("the final anchors :", all_anchors)
    bei = draw_size / size
    for anchor_grid in all_anchors:
        for anchor in anchor_grid[0]:
            draw.rectangle(((anchor[1][0]* bei, anchor[1][1]* bei), (anchor[1][0]*bei + anchor[2][0]* bei, anchor[1][1]* bei + anchor[2][1]* bei)),
                           fill=None, outline="red", width=3)
    img.show()


def grids_size(size):
    a = int(size/128)
    grids = []
    for i in range(a):
        for j in range(a):
            grid = [i*128,j*128]
            grids.append([grid,128])
    for i in range(a-1):
        for j in range(a-1):
            grid = [64+i*128,64+j*128]
            grids.append([grid,128])
    return grids

epoch  = 2
if __name__ == "__main__":
    path = "F:\desktop\human_face_seg\dataset\dev\Mark_Ruffalo_44_74,81,222,229.jpg"
    anchor_path = "F:\desktop\human_face_seg\dataset\\anchor.txt"
    start_time = time.time()


    img_array = Image.open(path)
    img_array = np.array(img_array)
    sizes = [128,256,384,512]
    #------开始多层------#
    for size in sizes:
        img_array = resize_img(img_array, size)
        end_time = time.time()
        print("the time before gaussian : ", end_time - start_time)
        # img_array = Gaussian_fliter(img_array)   # 我自己写的高斯模糊好慢好慢
        img_array = grey_img(img_array)
        # 先不要高斯模糊了
        # img_array 数组坐标第一维是纵向向下，第二维横向向右


        # 先分成2×2的grid
        # 通过每个grid，计算对应的anchor大小，anchor中心随机
        # 实际上不用先后之分把第一次grid和之后得到的grid分开了，本质结构的是一样的
        # 只是第一次的grid需要手动设置而已
        #---------------先得到最初的anchor-------------#
        percent, anchors = read_anchor(anchor_path)
        # percent 对于当前grid大小的变化比例
        # grid_size = img_array.shape[0]/2
        # grids 包括了每个grid的左上角坐标和右下角的坐标
        # 初始化grids
        # grid应该都是正方形，如果不是，那就按最大边比例增大（以中心增大）
        # grids = [[[0,0],[256,256]], [[0,256],[256,512]], [[256,0],[512,256]], [[256,256],[512,512]]]
        # 用这样写grids也太蠢了，定下左上角的坐标和宽高应该会更舒服一点，更何况宽高是一样的
        first = True
        grids = []
        for i in range(epoch):
            if(first):
                grids = grids_size(size)
                print("the grids :", grids)
                # grids = [[[0,0],256] , [[0,256],256], [[256,0],256], [[256,256],256], [[128,128],256]]# #[[[0,256],256], [[256,0],256]]
                first = False
            # 这里的grid_size应该改成girds的遍历，取gird[1],取到对应grid的anchors
            all_grids_anchors = []
            # for循环得到每个grid的anchors
            for grid in grids:
                anchors_size = anchor_size(grid[1], percent, anchors)
                all_grid_anchors = [grid ,anchors_size]
                all_grids_anchors.append(all_grid_anchors)
            all_anchors = grid_and_anchors(all_grids_anchors, img_array)
            # all_anchors = [[11.197367357118399, [219, 191], [136, 204]], [11.661618290978199, [260, 268], [204, 136]], [11.747328814293182, [282, 263], [146, 204]], [11.869714661751386, [325, 278], [153, 204]], [11.897485478722277, [325, 272], [136, 204]], [11.910908485551548, [268, 266], [146, 204]], [11.974309153694149, [338, 269], [153, 204]], [12.042762768819351, [221, 125], [153, 204]], [12.07948219335866, [248, 111], [146, 204]], [12.28002264470513, [148, 169], [204, 146]], [12.28741813731569, [285, 295], [204, 153]], [12.32837390962966, [280, 282], [204, 204]], [12.46547809147931, [260, 296], [204, 146]], [12.48988538296292, [18, 170], [153, 204]], [12.5573637295543, [164, 228], [204, 204]], [12.70901008528637, [44, 237], [204, 153]], [12.73218322015342, [315, 302], [136, 204]], [12.78169967744870, [293, 339], [204, 146]], [12.80895346901199, [149, 83], [204, 204]], [12.83596242040577, [298, 319], [204, 136]], [12.91020312677948, [158, 99], [136, 204]], [13.2931086613100, [208, 65], [204, 136]], [13.30993614149073, [28, 22], [204, 136]], [13.33886082484020, [274, 344], [204, 153]], [13.44380605640075, [259, 301], [204, 204]], [13.49213276094855, [127, 3], [204, 153]], [13.6440867461386, [80, 22], [146, 204]], [14.35870322700052, [42, 100], [204, 146]]]
            # print(len(all_anchors))
            # for i in range(len(all_anchors)):
            #     if(all_anchors[i][0] > 12):
            #         all_anchors.remove(all_anchors[i])
            # for i in range(len(all_anchors)):
            #     print(all_anchors[i])
            img_draw(path, all_anchors, size)
            print("all_anchors : 没有经过筛选合并",all_anchors)
            grid_anchors = NMS(all_anchors)
            print("grid_anchors : 经过筛选合并", grid_anchors)
            # img_draw(path, grid_anchors, size)
            grids = anchor2grid(grid_anchors)
            print("grids : ",grids)
            # if(i == epoch-1):
            #     img_draw(path, grid_anchors)
        # 把经过固定epoch的all_anchors 输出
        # img_draw(path, grid_anchors)

        #--------------------------------------------#
        # （把anchor定义为一个结构体），存放，在原图（已经resize）上的中心位置，左上角坐标，右下角坐标
        # 把anchor放入grid计算HOG

    end_time = time.time()
    print("the running time: ", end_time - start_time)
