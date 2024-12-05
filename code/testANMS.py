import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from pylab import *
from main_rechauffement import appliqueTransformation
from tqdm import tqdm
import os
from skimage import color
from harris import fspecial_gaussian
from scipy.signal import convolve2d as conv2
from scipy.ndimage.filters import generic_filter as gf
from skimage import draw
from main_manuel import estimH
import random
from scipy.spatial import distance

def ANMS(corners, k, c_robust):
    # I, vI, dI = [], [], []
    # nb_corners = len(corners[0])
    # mat_corners = np.array([corners[1], corners[2]])
    # mat_v = np.array([corners[0]])

    # i_max = np.argmax(corners[0])
    # mat_max = np.array([[corners[1][i_max]], [corners[2][i_max]]])
    # I.append(i_max)
    # vI.append(corners[0][i_max])
    # distances = np.sqrt(np.sum((mat_max-mat_corners)**2, axis=0))
    # distances[I] = -1
    # dI.append(distances)

    # while len(I) != k:
    #     for ic, c in enumerate(I):
    #         if not np.all(dI[ic]==-1):
    #             i_max = np.argmax(dI[ic])
    #             mat_max = np.array([[corners[1][i_max]], [corners[2][i_max]]])
    #             I.append(i_max)
    #             vI.append(corners[0][i_max])

    #             dI[ic][i_max] = -1
    #             idx2modif =  np.where(mat_v < c_robust*corners[0][i_max])[1]
    #             # for iv, v in enumerate(corners[0]):
    #             #     if v < c_robust*corners[0][i_max]:
    #             #         idx2modif.append(iv)
                
    #             distances = np.sqrt(np.sum((mat_max-mat_corners)**2, axis=0))
    #             distances[[i for i in range(distances.shape[0]) if i not in idx2modif]] = -1
    #             distances[I] = -1
    #             dI.append(distances)
    #             for ind_dI in range(len(dI)-1):
    #                 dI[ind_dI][idx2modif] = -1

    #             if len(I) == k: 
    #                 break

    # selected_corners = [vI]
    # selected_x = [corners[1][x] for x in I]
    # selected_y = [corners[2][y] for y in I]
    # selected_corners.append(selected_x), selected_corners.append(selected_y)


    selected_corners = []
    mat_c = np.array([corners[1], corners[2]])
    mat_v = np.array([corners[0]])
    r_list = []

    for idx in range(len(corners[0])):
        curr_point = np.array([[corners[1][idx]], [corners[2][idx]]])
        dist = np.sqrt(np.sum((mat_c-curr_point)**2, axis=0)).T

        mask = np.where(mat_v*c_robust > corners[0][idx], 1, 0)
        dist = dist*mask
        if np.all(dist==0):
            r = inf
        else:
            r = np.min(dist[dist!=0])
        r_list.append(r)

    I = []
    while len(I) < k:
        idx = np.argmax(r_list)
        if idx not in I:
            I.append(idx)

        r_list[idx] = -1
    
    if len(set(I)) != k:
        print('Erreur')
    
    selected_x = [corners[1][x] for x in I]
    selected_y = [corners[2][y] for y in I]
    selected_v = [corners[0][v] for v in I]
    selected_corners.append([selected_v, selected_x, selected_y])

    return selected_corners

    # selec = []
    # nb_coins = 5  # !!! changer à 500 !!!
    # v, x, y = corners
    # list_harris = [[v[i], x[i], y[i]] for i in range(len(v))]
    # # ANMS -> calculer les 500 ri et choper les 500 max juste
    # rayons_min = []
    # for i in tqdm(range(len(list_harris))):
    #     ri_min = np.inf
    #     xi = list_harris[i]
    #     for j in range(len(list_harris)):
    #         xj = list_harris[j]
    #         if (xi != xj) & (xi[0] < 0.9 * xj[0]):
    #             ri = distance.euclidean(xi[1:], xj[1:])
    #             if ri < ri_min:
    #                 ri_min = ri

    #     rayons_min.append([xi[1:],ri_min])

    # def sortSecond(val):
    #     return val[1]
    # rayons_min.sort(key=sortSecond,reverse=True)

    # coins = rayons_min[:(nb_coins-1)]
    # selec.append([c[0] for c in coins])

    # return selec

if __name__ == "__main__":
    corners = [[100, 85, 75, 25, 30], [2, 13, 7, 4, 13], [13, 14, 9, 4, 2]]
    k = 4
    c_robust = 0.9
    print(ANMS(corners, k, c_robust))