#coding:UTF-8
import numpy as np
import sys,os,cv2,re,time
import MakeNewFolder as mnf
import Fol_confiramation as FC
import datetime
from skimage.feature import hog


#ポケモン名を推定するメソッド。引数のdataはポケモン名とそのHOGを組にしたものです。
def estimate_poke_index(image, data):
    hog = calculateHOG(image)
    distance_list = calculate_manhattan_distance(hog, data)
    return np.argmin(np.array(distance_list))

# 画像を読み込み、HOG特徴量を計算して返すメソッド
def calculateHOG(image, orient=9, cell_size=5, block_size=6):
    # 画像を読み込む
    number_color_channels = np.shape(image)[2]
    if number_color_channels > 3:
        mask = image[:, :, 3]
        image = image[:, :, :3]
        for i in range(len(mask)):
            for j in range(len(mask[0])):
                if mask[i][j] == 0:
                    image[i, j, :] = 255
    # 画像を30x30にリサイズする
    resized_image = cv2.resize(image, (30, 30))
    images = cv2.split(resized_image)
    fd = []
    for monocolor_image in images:
        blur_image = cv2.GaussianBlur(monocolor_image, (3,3),0)
        fd.extend(hog(blur_image, orientations=orient,
                      pixels_per_cell=(cell_size, cell_size), cells_per_block=(block_size, block_size)))
    return fd

# HOG特徴量間の距離計算メソッド
def calculate_distance_HOG(target, data):
    distance_list = []
    rows, columns = np.shape(data)
    for i in range(rows):
        distance_list.append(np.linalg.norm(data[i, :] - target))
    return distance_list

# マンハッタン距離の計算メソッド
def calculate_manhattan_distance(target, data):
    distance_list = []
    rows, columns = np.shape(data)
    for i in range(rows):
        distance_list.append(np.sum(np.abs(data[i, :] - target)))
    return distance_list

if __name__ == '__main__':
    src = cv2.imread('/Users/okumurashunsuke/Desktop/Datas_2018_1027/test_images_after_trim/2017_10_02_38_1080_1440_Original.png')
    print len(calculateHOG(src))
