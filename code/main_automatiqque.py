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
from main_manuel import estimH, norm_pts, Cursor, getmosaique
import random
import skimage as sk
import skimage.io as skio

def load_Serie(img_name, numb_serie):
    path = f'../images/2-PartieAutomatique/Serie{numb_serie}'
    im_list = []
    for name in img_name:
        imgpath = os.path.join(path, name)
        im = io.imread(imgpath)
        # if len(im.shape) == 3:
        #     im_list.append(color.rgb2gray(im))
        # else:
        im_list.append(im)
    
    return im_list

def harris(im_list):
    corners = []
    threshold = 1e-3
    for im in tqdm(im_list, desc='harris detection'):
        if len(im.shape) == 3:
            im = color.rgb2gray(im)
        
        g1 = fspecial_gaussian([9, 9], 1)  # Gaussian with sigma_d
        g2 = fspecial_gaussian([11, 11], 1.5)  # Gaussian with sigma_i

        img1 = conv2(im, g1, 'same')  # blur image with sigma_d
        Ix = conv2(img1, np.array([[-1, 0, 1]]), 'same')  # take x derivative
        Iy = conv2(img1, np.transpose(np.array([[-1, 0, 1]])), 'same')  # take y derivative

        # Compute elements of the Harris matrix H
        # we can use blur instead of the summing window
        Ix2 = conv2(np.multiply(Ix, Ix), g2, 'same')
        Iy2 = conv2(np.multiply(Iy, Iy), g2, 'same')
        IxIy = conv2(np.multiply(Ix, Iy), g2, 'same')
        eps = 2.2204e-16
        R = np.divide(np.multiply(Ix2, Iy2) - np.multiply(IxIy, IxIy),(Ix2 + Iy2 + eps))

        # don't want corners close to image border
        R[0:15] = 0  # all columns from the first 15 lines
        R[-16:] = 0  # all columns from the last 15 lines
        R[:, 0:15] = 0  # all lines from the first 15 columns
        R[:, -16:] = 0  # all lines from the last 15 columns

        # non-maxima suppression within 3x3 windows
        Rmax = gf(R, np.max, footprint=np.ones((3, 3)))
        Rmax[Rmax != R] = 0  # suppress non-max
        Rmax[Rmax < threshold] = 0
        v = Rmax[Rmax != 0]
        y, x = np.nonzero(Rmax)

        corners.append([v, x, y])

    return corners

def show_harris_corners(im_list, corners):
    n_row = len(im_list) // 3
    fig, axes = plt.subplots(n_row, 3, figsize=(12, 8))
    for ax, img, pts in zip(axes.flatten(), im_list, corners):
        _,x,y = pts
        for xp, yp in zip(x, y):
            rr, cc = draw.circle_perimeter(yp, xp, radius=6, shape=img.shape)
            img[rr, cc] = 1
        ax.imshow(img)

    plt.show()

def ANMS(corners_list, k, c_robust):
    selected_corners = []
    for corners in tqdm(corners_list, desc='Corners selection'):
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

def getDescriptors (img_list, corners_list):
    descriptors = []
    for idx, corners in enumerate(corners_list):
        img_desc = []
        _, x, y = corners
        for id_corner, (xc, yc) in enumerate(zip(x, y)):

            window_size = 40
            begin_x = max(xc - window_size // 2, 0)
            begin_y = max(yc - window_size // 2, 0)
            end_x = min(begin_x + window_size, img_list[idx].shape[1])
            end_y = min(begin_y + window_size, img_list[idx].shape[0])
            if len(img_list[idx]) == 3:
                cropped_img = img_list[idx][begin_y:end_y, begin_x:end_x, :]
            else:
                cropped_img = img_list[idx][begin_y:end_y, begin_x:end_x]

            if cropped_img.shape[0] == 40 and cropped_img.shape[1] == 40:
                # sub_img = cropped_img[::5, ::5]
                if len(img_list[idx].shape) == 3:
                    lum_cropped = np.sum(cropped_img, axis=2) / 3
                    sub_img = sk.transform.rescale(lum_cropped, 0.2, anti_aliasing=True)
                else:
                    sub_img = sk.transform.rescale(cropped_img, 0.2, anti_aliasing=True)

                desc = np.zeros(sub_img.shape)
                moy = np.mean(sub_img)
                std = np.std(sub_img-moy)
                desc = (sub_img-moy)/std
                
                img_desc.append([desc, id_corner])
        descriptors.append(img_desc)

    return descriptors

def matching_descriptors(descriptors_list, seuil, nb_im_ref):
    '''
    Retourne la liste des index des descriptors qui match
    '''
    matching_descriptors = []
    descriptors = [[desc[0] for desc in descriptors_list[0]], [desc[0] for desc in descriptors_list[1]]]
    id_cornerofDesc = [[desc[1] for desc in descriptors_list[0]], [desc[1] for desc in descriptors_list[1]]]

    desc_im1 = descriptors[0]
    desc_im2 = descriptors[1]

    for idx_ref, desc_ref in enumerate(desc_im1):
        SDC = [np.mean(np.square(desc_ref-desc)) for desc in desc_im2]
        SDC_NN1, id_NN1 = np.min(SDC), np.argmin(SDC)
        SDC_NN2 = inf
        for id_sdc in range(len(SDC)):
            if SDC[id_sdc] < SDC_NN2 and id_sdc != id_NN1:
                SDC_NN2 = SDC[id_sdc]
    
        if SDC_NN1/SDC_NN2 < seuil:
            matching_descriptors.append([id_cornerofDesc[0][idx_ref], id_cornerofDesc[1][id_NN1]])

    return matching_descriptors

def Ransac(matching_desc,  corners, iteration, seuil, im_list, i):
    '''
    matching_desc: liste des index des match [[[idxim1, idxim2], [[idxim1, idxim2]]]]
    corners : les coins sélectionnés [[[v], [x], [y]]]
    iteration: nb iterations
    seuil : seuil
    im_list: liste des images [im1, im2]

    RETURN
    H_list: la liste des H entre deux images
    '''
    matching_corners = getCorners_matching(matching_desc, corners, i)
    all_p1, all_p2 = [row[0] for row in matching_corners], [row[1] for row in matching_corners]
    all_p1, all_p2 = np.array(all_p1), np.array(all_p2)
    all_p1_homo, all_p2_homo = np.column_stack((all_p1, np.ones(all_p1.shape[0]))).T, np.column_stack((all_p2, np.ones(all_p2.shape[0]))).T

    nb_max_pt = 0
    best_combinaison = []
    sum_SDC = inf
    for it in range(iteration):
        samples = random.sample(matching_corners, k=4)
        pts1, pts2 = [row[0] for row in samples], [row[1] for row in samples]
        pts1_norm, T1 = norm_pts(np.array(pts1))
        pts2_norm, T2 = norm_pts(np.array(pts2))

        # fig, axs = plt.subplots(1, 2, figsize=(10, 3))
        # axs[0].imshow(im_list[0])
        # axs[0].scatter([pts1[i][0] for i in range(4)], [pts1[i][1] for i in range(4)], color='red', marker='+')
        # axs[1].imshow(im_list[1])
        # axs[1].scatter([pts2[i][0] for i in range(4)], [pts2[i][1] for i in range(4)], color='red', marker='+')
        # plt.title('pts piger')
        # plt.show()

        Hnorm = estimH(pts1_norm, pts2_norm)
        H = inv(T2) @ Hnorm @ T1

        preds = H @ all_p1_homo
        preds = preds / preds[-1, :]

        SDC = np.mean(np.square(preds[:-1, :]-all_p2_homo[:-1, :]), axis=0)
        idx_consistant = np.where(SDC < seuil)[0]
        new_sum_sdc = np.sum([SDC[i] for i in idx_consistant])

        if len(idx_consistant) > nb_max_pt:
            nb_max_pt = len(idx_consistant)
            best_combinaison = idx_consistant
            sum_SDC = new_sum_sdc
        elif len(idx_consistant) == nb_max_pt and new_sum_sdc < sum_SDC:
            nb_max_pt = len(idx_consistant)
            best_combinaison = idx_consistant
            sum_SDC = new_sum_sdc
        
    # print(best_combinaison)
    cons_p1, cons_p2 = all_p1[best_combinaison, :], all_p2[best_combinaison, :]

    cons_p1_norm, T1 = norm_pts(cons_p1)
    cons_p2_norm, T2 = norm_pts(cons_p2)

    Hnorm = estimH(cons_p1_norm, cons_p2_norm)
    H = inv(T2) @ Hnorm @ T1

    # cons_p1_homo = np.column_stack((cons_p1, np.ones(cons_p1.shape[0]))).T
    # preds = H @ cons_p1_homo
    # preds = preds / preds[-1, :]

    # fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    # axs[0].imshow(im_list[0])
    # axs[0].scatter(cons_p1[:,0], cons_p1[:,1], color='red', marker='+')
    # axs[1].imshow(im_list[1])
    # axs[1].scatter(cons_p2[:,0], cons_p2[:,1], color='red', marker='+')
    # axs[1].scatter(preds[0,:], preds[1,:], color='blue', marker='x')
    # plt.savefig()
    # plt.show()

    return [H, cons_p1, cons_p2]

def getCorners_matching(matching_desc, selected_corners, i):
    matching_corners = []
    for id_match, match_list in enumerate(matching_desc):
        idx_im1, idx_im2 = match_list
        corner1 = [selected_corners[i][1][idx_im1], selected_corners[i][2][idx_im1]]
        corner2 = [selected_corners[i+1][1][idx_im2], selected_corners[i+1][2][idx_im2]]
        matching_corners.append([corner1, corner2])

    return matching_corners

def getTransformationsSerie1(im_list, nb_im_ref):
    corners = harris(im_list)
    selected_corners = ANMS(corners, k=500, c_robust=0.9)
    descriptors = getDescriptors(im_list, selected_corners)
    #Garder comme ça

    seuil = 0.4
    idx_matching_descriptors = []
    for idx in tqdm(range(1, len(im_list)), desc='Matching desc'):
        desc = [descriptors[idx-1], descriptors[idx]]
        idx_matching = matching_descriptors(desc, seuil, nb_im_ref)
        idx_matching_descriptors.append(idx_matching)

        matching_pts = getCorners_matching(idx_matching, selected_corners, idx-1)
        c1x, c1y = [c[0][0] for c in matching_pts], [c[0][1] for c in matching_pts]
        c2x, c2y = [c[1][0] for c in matching_pts], [c[1][1] for c in matching_pts]

        # fig, axs = plt.subplots(1, 2, figsize=(10, 3))
        # axs[0].imshow(im_list[idx-1])
        # for idx_c, (x,y) in enumerate(zip(c1x, c1y)):
        #     axs[0].scatter(x, y, color='red', marker='+')
        #     axs[0].annotate(str(idx_c), (x, y))
        # axs[1].imshow(im_list[idx])
        # for idx_c, (x,y) in enumerate(zip(c2x, c2y)):
        #     axs[1].scatter(x, y, color='red', marker='+')
        #     axs[1].annotate(str(idx_c), (x, y))
        # plt.savefig(f'../resultats/Ensemble_des_matchs_pour_{idx-1}_et_{idx}_Serie1.png')
        # plt.show()

    #Refaire pour avoir une fonction que entre deux images
    H_list = []
    for i, idx_match in enumerate(idx_matching_descriptors):
        H, cons_p1, cons_p2 = Ransac(idx_match, selected_corners, 50, 0.4, im_list, i)
        H_list.append(H)

        fig, axs = plt.subplots(1, 2, figsize=(10, 3))
        axs[0].imshow(im_list[i])
        axs[0].scatter(cons_p1[:,0], cons_p1[:,1], color='red', marker='+')
        axs[1].imshow(im_list[i+1])
        axs[1].scatter(cons_p2[:,0], cons_p2[:,1], color='red', marker='+')
        plt.savefig(f'../resultats/Ensemble_des_matchs_pour_{i}_et_{i+1}_Serie1.png')
        plt.show()
    #idem
    
    H02 = H_list[1] @H_list[0]
    H12 = H_list[1]
    H32 = inv(H_list[2])
    H42 = inv(H_list[2]) @ inv(H_list[3])
    H52 = H42 @ inv(H_list[4])

    
    im1Trans, new_origine1 = appliqueTransformation(im_list[0], H02)
    im2Trans, new_origine2 = appliqueTransformation(im_list[1], H12)
    im3 = (im_list[2]- im_list[2].min())/(im_list[2].max() - im_list[2].min())
    im4Trans, new_origine3 = appliqueTransformation(im_list[3], H32)
    im5Trans, new_origine4 = appliqueTransformation(im_list[4], H42)
    im6Trans, new_origine5 = appliqueTransformation(im_list[5], H52)

    fig, axs = plt.subplots(1, 6, figsize=(10, 3))
    axs[0].imshow(im1Trans)
    axs[1].imshow(im2Trans)
    axs[2].imshow(im3)
    axs[3].imshow(im4Trans)
    axs[4].imshow(im5Trans)
    axs[5].imshow(im6Trans)
    plt.show()

    im_trans = [im1Trans, im2Trans, im3, im4Trans, im5Trans, im6Trans]
    trans = [new_origine1, new_origine2, new_origine3, new_origine4, new_origine5]
    mosaique = getmosaique(im_trans, trans, 2)

    mosaique = sk.img_as_ubyte(mosaique)
    name = input("Nom du fichier (.png): ")
    fname = '../resultats/'+name
    skio.imsave(fname, mosaique)

    return 0

def getTransformationsSerie2(im_list, nb_im_ref):
    corners = harris(im_list)
    selected_corners = ANMS(corners, k=500, c_robust=0.9)
    descriptors = getDescriptors(im_list, selected_corners)
    #Garder comme ça

    seuil = 0.4
    idx_matching_descriptors = []
    for idx in tqdm(range(1, len(im_list)), desc='Matching desc'):
        desc = [descriptors[idx-1], descriptors[idx]]
        idx_matching = matching_descriptors(desc, seuil, nb_im_ref)
        idx_matching_descriptors.append(idx_matching)

        matching_pts = getCorners_matching(idx_matching, selected_corners, idx-1)
        c1x, c1y = [c[0][0] for c in matching_pts], [c[0][1] for c in matching_pts]
        c2x, c2y = [c[1][0] for c in matching_pts], [c[1][1] for c in matching_pts]

        # fig, axs = plt.subplots(1, 2, figsize=(10, 3))
        # axs[0].imshow(im_list[idx-1])
        # for idx_c, (x,y) in enumerate(zip(c1x, c1y)):
        #     axs[0].scatter(x, y, color='red', marker='+')
        #     axs[0].annotate(str(idx_c), (x, y))
        # axs[1].imshow(im_list[idx])
        # for idx_c, (x,y) in enumerate(zip(c2x, c2y)):
        #     axs[1].scatter(x, y, color='red', marker='+')
        #     axs[1].annotate(str(idx_c), (x, y))
        # plt.savefig(f'../resultats/Ensemble_des_matchs_pour_{idx-1}_et_{idx}_Serie2.png')
        # plt.show()

    H_list = []
    for i, idx_match in enumerate(idx_matching_descriptors):
        H, cons_p1, cons_p2 = Ransac(idx_match, selected_corners, 50, 0.4, im_list, i)
        H_list.append(H)

        fig, axs = plt.subplots(1, 2, figsize=(10, 3))
        axs[0].imshow(im_list[i])
        axs[0].scatter(cons_p1[:,0], cons_p1[:,1], color='red', marker='+')
        axs[1].imshow(im_list[i+1])
        axs[1].scatter(cons_p2[:,0], cons_p2[:,1], color='red', marker='+')
        plt.savefig(f'../resultats/Ensemble_des_matchs_pour_{i}_et_{i+1}_Serie2.png')
        plt.show()

    H01 = H_list[0]
    H21 = inv(H_list[1])
    H31 = inv(H_list[2] @ H_list[1])#H21 @ inv(H_list[2])


    print('transformation I1...\n')
    im1Trans, new_origine1 = appliqueTransformation(im_list[0], H01)
    print('done\n')
    im2 = (im_list[1]- im_list[1].min())/(im_list[1].max() - im_list[1].min())
    print('transformation I3...\n')
    im3Trans, new_origine2 = appliqueTransformation(im_list[2], H21)
    print('done\n')
    print('transformation I4...\n')
    im4Trans, new_origine3 = appliqueTransformation(im_list[3], H31)
    print('done\n')

    fig, axs = plt.subplots(1, 4, figsize=(10, 3))
    axs[0].imshow(im1Trans)
    axs[1].imshow(im2)
    axs[2].imshow(im3Trans)
    axs[3].imshow(im4Trans)
    plt.show()

    im_trans = [im1Trans, im2, im3Trans, im4Trans]
    trans = [new_origine1, new_origine2, new_origine3]
    mosaique = getmosaique(im_trans, trans, 1)

    mosaique = sk.img_as_ubyte(mosaique)
    name = input("Nom du fichier (.png): ")
    fname = '../resultats/'+name
    skio.imsave(fname, mosaique)

    return 0

def getTransformationsSerie3(im_list, nb_im_ref):
    im_list1,im_list2 = [im_list[i] for i in range(0, 3)], [im_list[i] for i in range(3, len(im_list))]

    corners = harris(im_list1)
    selected_corners = ANMS(corners, k=500, c_robust=0.9)
    descriptors = getDescriptors(im_list1, selected_corners)
    seuil = 0.4
    idx_matching_descriptors1,  idx_matching_descriptors2= [], []

    for idx in tqdm(range(1, len(im_list1)), desc='Matching desc'):
        desc = [descriptors[idx-1], descriptors[idx]]
        idx_matching = matching_descriptors(desc, seuil, 1)
        idx_matching_descriptors1.append(idx_matching)

    H_list1 = []
    for i, idx_match in enumerate(idx_matching_descriptors1):
        H, cons_p1, cons_p2 = Ransac(idx_match, selected_corners, 50, 0.4, im_list1, i)
        H_list1.append(H)

        fig, axs = plt.subplots(1, 2, figsize=(10, 3))
        axs[0].imshow(im_list1[i])
        axs[0].scatter(cons_p1[:,0], cons_p1[:,1], color='red', marker='+')
        axs[1].imshow(im_list1[i+1])
        axs[1].scatter(cons_p2[:,0], cons_p2[:,1], color='red', marker='+')
        plt.savefig(f'../resultats/Ensemble_des_matchs_pour_{i}_et_{i+1}_Serie31.png')
        plt.show()


    corners = harris(im_list2)
    selected_corners = ANMS(corners, k=500, c_robust=0.9)
    descriptors = getDescriptors(im_list2, selected_corners)
    for idx in tqdm(range(1, len(im_list2)), desc='Matching desc'):
        desc = [descriptors[idx-1], descriptors[idx]]
        idx_matching = matching_descriptors(desc, seuil, 1)
        idx_matching_descriptors2.append(idx_matching)

    H_list2 = []
    for i, idx_match in enumerate(idx_matching_descriptors2):
        H, cons_p1, cons_p2 = Ransac(idx_match, selected_corners, 50, 0.4, im_list2, i)
        H_list2.append(H)

        fig, axs = plt.subplots(1, 2, figsize=(10, 3))
        axs[0].imshow(im_list2[i])
        axs[0].scatter(cons_p1[:,0], cons_p1[:,1], color='red', marker='+')
        axs[1].imshow(im_list2[i+1])
        axs[1].scatter(cons_p2[:,0], cons_p2[:,1], color='red', marker='+')
        plt.savefig(f'../resultats/Ensemble_des_matchs_pour_{i}_et_{i+1}_Serie32.png')
        plt.show()

    H01_1 = H_list1[0]
    H21_1 = inv(H_list1[1])
    H01_2 = H_list2[0]
    H21_2 = inv(H_list2[1])


    
    im1Trans, new_origine1 = appliqueTransformation(im_list1[0], H01_1)
    im2 = (im_list1[1]- im_list1[1].min())/(im_list1[1].max() - im_list1[1].min())
    im3Trans, new_origine2 = appliqueTransformation(im_list1[2], H21_1)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    axs[0].imshow(im1Trans)
    axs[1].imshow(im2)
    axs[2].imshow(im3Trans)
    plt.show()

    im_trans = [im1Trans, im2, im3Trans]
    trans = [new_origine1, new_origine2]
    mosaique1 = getmosaique(im_trans, trans, 1)

    mosaique1 = sk.img_as_ubyte(mosaique1)
    name = input("Nom du fichier (.png): ")
    fname = '../resultats/'+name
    skio.imsave(fname, mosaique1)

    im1Trans, new_origine1 = appliqueTransformation(im_list2[0], H01_2)
    im2 = (im_list2[1]- im_list2[1].min())/(im_list2[1].max() - im_list2[1].min())
    im3Trans, new_origine2 = appliqueTransformation(im_list2[2], H21_2)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    axs[0].imshow(im1Trans)
    axs[1].imshow(im2)
    axs[2].imshow(im3Trans)
    plt.show()

    im_trans = [im1Trans, im2, im3Trans]
    trans = [new_origine1, new_origine2]
    mosaique2 = getmosaique(im_trans, trans, 1)

    mosaique2 = sk.img_as_ubyte(mosaique2)
    name = input("Nom du fichier (.png): ")
    fname = '../resultats/'+name
    skio.imsave(fname, mosaique2)


    mosaic_list = [mosaique1, mosaique2]

    corners = harris(mosaic_list)
    selected_corners = ANMS(corners, k=500, c_robust=0.9)
    descriptors = getDescriptors(mosaic_list, selected_corners)
    idx_matching_descriptors = []
    for idx in tqdm(range(1, len(mosaic_list)), desc='Matching desc'):
        desc = [descriptors[idx-1], descriptors[idx]]
        idx_matching = matching_descriptors(desc, seuil, 0)
        idx_matching_descriptors.append(idx_matching)

        # matching_pts = getCorners_matching(idx_matching, selected_corners, idx-1)
        # c1x, c1y = [c[0][0] for c in matching_pts], [c[0][1] for c in matching_pts]
        # c2x, c2y = [c[1][0] for c in matching_pts], [c[1][1] for c in matching_pts]

        # fig, axs = plt.subplots(1, 2, figsize=(10, 3))
        # axs[0].imshow(im_list2[idx-1])
        # for idx_c, (x,y) in enumerate(zip(c1x, c1y)):
        #     axs[0].scatter(x, y, color='red', marker='+')
        #     axs[0].annotate(str(idx_c), (x, y))
        # axs[1].imshow(im_list2[idx])
        # for idx_c, (x,y) in enumerate(zip(c2x, c2y)):
        #     axs[1].scatter(x, y, color='red', marker='+')
        #     axs[1].annotate(str(idx_c), (x, y))
        # plt.savefig(f'../resultats/Ensemble_des_matchs_pour_{idx-1}_et_{idx}_Serie3.png')
        # plt.show()

    H_list = []
    for i, idx_match in enumerate(idx_matching_descriptors):
        H, cons_p1, cons_p2 = Ransac(idx_match, selected_corners, 50, 0.4, mosaic_list, i)
        H_list.append(H)

        fig, axs = plt.subplots(1, 2, figsize=(10, 3))
        axs[0].imshow(mosaic_list[i])
        axs[0].scatter(cons_p1[:,0], cons_p1[:,1], color='red', marker='+')
        axs[1].imshow(mosaic_list[i+1])
        axs[1].scatter(cons_p2[:,0], cons_p2[:,1], color='red', marker='+')
        plt.savefig(f'../resultats/Ensemble_des_matchs_pour_{i}_et_{i+1}_Serie3.png')
        plt.show()

    im1 = (mosaic_list[0]-mosaic_list[0].min())/(mosaic_list[0].max() - mosaic_list[0].min())
    im2Trans, new_origine2 = appliqueTransformation(mosaic_list[1], inv(H_list[0]))

    fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    axs[0].imshow(im1)
    axs[1].imshow(im2Trans)
    plt.show()
    
    im_trans = [im1, im2Trans]
    trans = [new_origine2]
    mosaique = getmosaique(im_trans, trans, 0, vertical=False)

    mosaique = sk.img_as_ubyte(mosaique)
    name = input("Nom du fichier (.png): ")
    fname = '../resultats/'+name
    skio.imsave(fname, mosaique)


    return 0

if __name__ == "__main__":
    name_Serie1 = ['goldengate-00.png', 'goldengate-01.png', 'goldengate-02.png', 'goldengate-03.png', 'goldengate-04.png', 'goldengate-05.png']
    name_Serie2 = ['IMG_2415.JPG', 'IMG_2416.JPG', 'IMG_2417.JPG', 'IMG_2418.JPG']
    name_Serie3 = ['IMG_2436.JPG', 'IMG_2435.JPG', 'IMG_2434.JPG', 'IMG_2468.JPG', 'IMG_2467.JPG', 'IMG_2466.JPG']
    name_test = ['goldengate-02.png', 'goldengate-03.png']

    im_list = load_Serie(name_Serie1, 1)
    getTransformationsSerie1(im_list, 2)

    im_list = load_Serie(name_Serie2, 2)
    getTransformationsSerie2(im_list, 1)

    im_list = load_Serie(name_Serie3, 3)
    getTransformationsSerie3(im_list, 2)

