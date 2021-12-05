# import numpy as np
# import os
# import PIL
# import math
# from Sobel import sobel_xy
# from opera_img import grey_img, resize_img, crop_img
# from fliter import mean_fliter, Gaussian_fliter
# from HOG import Historode, Block
# from PIL import Image
#
#
#
# path = "F:\desktop\human_face_seg\dataset\dev"
# imgs = os.listdir(path)
# for img in imgs:
#     print(img,"is come! ")
#     # img_array = np.array(PIL.Image.open(path))
#     try:
#         img_array = crop_img(img) # 这里默认了我数据都放在"F:\desktop\human_face_seg\dataset\dev"
#     except OSError:
#         print(img, "the picture is too big!!")
#         continue
#
#     try:
#         img_array = resize_img(img_array)
#     except: # 先指定所有错误，之后再看图片具体报错的原因
#         f = open("F:\desktop\human_face_seg\dataset\wrong_img.txt", "a")
#         f.write(img)
#         f.write("\n")
#         continue
#     # print(img_array.shape)
#     # img_array = cv2.imread(path) # BGR
#     grad_array, amp_array = sobel_xy(grey_img(Gaussian_fliter(img_array)))
#     # show_array = Image.fromarray(amp_array)
#     # show_array.show()
#     cell_array = Historode(grad_array, amp_array)
#     # print(cell_array[10,10])
#     block_array = Block(cell_array)
#     print("the block_array's shape is :")
#     print(block_array.shape)
#     print("the block_array looks like :")
#     print(block_array)
#
#     # block_array = block_array.T
#     # print(block_array.shape)
#     # dets = np.array([[1, 2], [3, 4]])
#     # np.savetxt("F:\desktop\human_face_seg\dataset\HOG_mean.txt", block_array, fmt='%f', delimiter=',')
#     f = open("F:\desktop\human_face_seg\dataset\HOG_all.txt","a")
#     f.write(img)
#     f.write(",")
#     for i in block_array:
#         f.write(str(i))
#         f.write(",")
#     f.write('\n')
#     f.close()
#
