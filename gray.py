# import 需要的package
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import time
from scipy.optimize import curve_fit
import re
import time 
from tqdm import tqdm

# 定义拟合函数
def function(x, y_0, a, k):
    return y_0 + a * np.exp(-k * x)

# 定义print函数的颜色
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
RESET = '\033[0m'

# 开始时间
start_time = time.time()

# 定义图片文件夹的路径
image_folder_path = "D:/image process/3x3-00/31-569-3x3-00/"        # ************
# 展开该文件夹里的所有文件名
image_folder = os.listdir(image_folder_path)
# 得到图片文件夹中含有多少张图片（一组图片大概50~90）
length = len(image_folder)  

# 定义一个空list，用来存储每张图片的灰度数据，每个图片的大小是（宽*高）
image_folder_data = []

# 此for循环读取每个灰度图片的数据，将之从（宽*高）的2维数组压缩至1维数组，然后存到image_folder_data中
for i in range(length):
    # 得到每个图片的路径
    image_single_path = image_folder_path + image_folder[i]
    # 打开每个图片，得到该图片的数据
    image_single_data = Image.open(image_single_path)
    # 将该图片的数据转化成numpy array
    imarray = np.array(image_single_data)
    # 得到图片的宽和高
    width, height = imarray.shape
    # 将每个图片的数组压缩至1维
    imarray = imarray.flatten()
    # 将1维数据存到image_folder_data，形成（length, width*height)的shape
    image_folder_data.append(imarray)
    
# 将image_folder_data转化成数组
image_folder_data = np.array(image_folder_data)
# 将该数组转置
image_folder_data = image_folder_data.T
# 找到image_folder_data中所有灰度值小于20的点的索引值（index_pixel)
mask = image_folder_data < 20               # ************
index = np.where(mask == True)
index_pixel = index[0]
index_pixel = np.unique(index_pixel)
# 定义每两张图片的时间间隔
delta_t = 28e-3                             # ************
# 得到图片对应的时间序列，比如（0s, 0.28s, 0.56s ...)
time_list = delta_t * np.arange(length)
# 定义存储k和ln(k)的空list
k_list = []
lnk_list = []
# 此for循环用来拟合灰度大于20的值
for i in tqdm(range(width * height)):
    # 此if语句排除灰度值小于20的点
    if i not in index_pixel:
        # 拟合函数：scipy.optimize.curve_fit。得到拟合函数的参数（y_0, a, k）
        params = curve_fit(function, time_list, image_folder_data[i], p0 = [100, -50, 5.5], maxfev=5000)[0]     # p0:起始猜测值 maxfev：猜测次数************
        # 得到k参数
        k = params[2]
        # 得到ln(k)
        lnk = np.log(k)
        # 将k，ln(k)存到list
        k_list.append((lnk, i))  # zip the lnk and pixel index (index after flatten)
        lnk_list.append(lnk)

# 求得ln(k)的均值、标准差
lnk_list = np.array(lnk_list)
lnk_mean = np.mean(lnk_list)
lnk_std = np.std(lnk_list)
end_time = time.time()
# 输出脚本运行时间、ln(k)的最大值、最小值、均值、标准差
print(GREEN + "Time cost:  |" + RESET, CYAN + str(end_time - start_time) + RESET)
print(GREEN + "Max ln(k):  |" + RESET, CYAN + str(sorted(k_list, reverse = True)[0][0]) + RESET)
print(GREEN + "Min ln(k):  |" + RESET, CYAN + str(sorted(k_list)[0][0]) + RESET)
print(GREEN + "Mean ln(k): |" + RESET, CYAN + str(lnk_mean) + RESET)
print(GREEN + "Std ln(k):  |" + RESET, CYAN + str(lnk_std) + RESET)

# 将压缩的1维数组还原成（width,height)形状
res = np.zeros(width * height)
for i in tqdm(k_list):
    res[i[1]] = i[0]
res = res.reshape(width, height)

# 将灰度值矩阵（已删除小于20的点）存储成csv表格
df = pd.DataFrame(res)
df.to_csv("C:/Users/fishi/Desktop/test.csv", index=False)       # ************

# 定义colorbar的极值
vmin = 1    # ************
vmax = 3    # ************

# 显示伪彩图片并存储
plt.imshow(res, cmap = 'jet', vmin = vmin, vmax = vmax)     # cmap定义伪彩种类************
plt.colorbar()
plt.savefig("C:/Users/fishi/Desktop/test.png")                  # ************
plt.show()

