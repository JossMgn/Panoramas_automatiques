import numpy as np
from numpy.linalg import inv, svd
import skimage as sk
import skimage.io as skio
from imageio.v2 import imread
import matplotlib.pyplot as plt
from pylab import *
from main_rechauffement import appliqueTransformation
from tqdm import tqdm

def ptsread(fname):
    points = []
    with open(fname, 'r') as f:
        for line in f:
            l = line.strip().split(',')
            l = [float(each) for each in l]
            points.append(l)
    
    return np.array(points)

class Cursor:
    def __init__(self, ax, s):
        self.ax = ax
        self.lx = ax.axhline(color='k')  # the horiz line
        self.ly = ax.axvline(color='k')  # the vert line

        self.points = []
        self.count = 1
        self.s = s

    def mouseclick(self, event):
        if not event.inaxes: return

        x, y = event.xdata, event.ydata

        self.points.append([x, y])
        self.ax.text(x+4, y-4, str(self.count), fontsize=14, color='r')
        self.ax.plot(x, y, '.r')
        self.count += 1
        draw()
    
    def convert_to_numpy_array(self):
        # Convertir la liste de points en une matrice NumPy
        return np.array(self.points)

def norm_pts(pts):
    mean = np.mean(pts, axis=0)
    var = np.var(pts, axis=0)

    # T = np.array([[1/var[0], 0, -mean[0]], [0, 1/var[1], -mean[1]], [0, 0, 1]])
    if var[0] != 0 or var[1] != 0:
        T = np.array([[1/var[0], 0, -mean[0]/var[0]], [0, 1/var[1], -mean[1]/var[1]], [0, 0, 1]])
        return [(pts-mean)/var, T]
    else:
        T = np.array([[1, 0, -mean[0]], [0, 1, -mean[1]], [0, 0, 1]])
        return [(pts-mean), T]
        
        
def estimH(pts_p, pts_p2):
    # h = np.zeros((9,1))
    # A = np.array([[-pts_p[0][0], -pts_p[0][1], -1, 0, 0, 0, pts_p[0][0]*pts_p2[0][0], pts_p[0][1]*pts_p2[0][0], pts_p2[0][0]],
    #                 [0, 0, 0, -pts_p[0][0], -pts_p[0][1], -1, pts_p[0][0]*pts_p2[0][1], pts_p[0][1]*pts_p2[0][1], pts_p2[0][1]],
    #                 [-pts_p[1][0], -pts_p[1][1], -1, 0, 0, 0, pts_p[1][0]*pts_p2[1][0], pts_p[1][1]*pts_p2[1][0], pts_p2[1][0]],
    #                 [0, 0, 0, -pts_p[1][0], -pts_p[1][1], -1, pts_p[1][0]*pts_p2[1][1], pts_p[1][1]*pts_p2[1][1], pts_p2[1][1]],
    #                 [-pts_p[2][0], -pts_p[2][1], -1, 0, 0, 0, pts_p[2][0]*pts_p2[2][0], pts_p[2][1]*pts_p2[2][0], pts_p2[2][0]],
    #                 [0, 0, 0, -pts_p[2][0], -pts_p[2][1], -1, pts_p[2][0]*pts_p2[2][1], pts_p[2][1]*pts_p2[2][1], pts_p2[2][1]],
    #                 [-pts_p[3][0], -pts_p[3][1], -1, 0, 0, 0, pts_p[3][0]*pts_p2[3][0], pts_p[3][1]*pts_p2[3][0], pts_p2[3][0]],
    #                 [0, 0, 0, -pts_p[3][0], -pts_p[3][1], -1, pts_p[3][0]*pts_p2[3][1], pts_p[3][1]*pts_p2[3][1], pts_p2[3][1]]])
    
    A = []
    for i in range(pts_p.shape[0]):
        A.append([-pts_p[i][0], -pts_p[i][1], -1, 0, 0, 0, pts_p[i][0]*pts_p2[i][0], pts_p[i][1]*pts_p2[i][0], pts_p2[i][0]])
        A.append([0, 0, 0, -pts_p[i][0], -pts_p[i][1], -1, pts_p[i][0]*pts_p2[i][1], pts_p[i][1]*pts_p2[i][1], pts_p2[i][1]])
    A = np.array(A)
    
    _, _, Vh = svd(A)
    sol = Vh[-1, :]
    
    H= np.array([[sol[0], sol[1], sol[2]],
                 [sol[3], sol[4], sol[5]],
                 [sol[6], sol[7], sol[8]]])

    return H

def test_estimH():
    np.random.seed(42)
    pts1 = np.random.randint(0, 100, size=(4, 2))
    pts2 = np.random.randint(0, 100, size=(4, 2))

    pts1_norm, T1 = norm_pts(pts1)
    pts2_norm , T2= norm_pts(pts2)

    Hnorm = estimH(pts1_norm, pts2_norm)
    H = inv(T2)@Hnorm@T1

    pts_estim = H @ np.vstack((pts1.T, np.ones(pts1.T.shape[1])))
    pts_estim = pts_estim / pts_estim[2]

    print('objectif:\n ', pts2)
    print('estimation:\n ', pts_estim.T)

def getmosaique(im_list, trans, indice_ref, vertical=True):
    nb_im = len(im_list)
    if vertical:
        if indice_ref != len(im_list)-1:
            width = int(abs(trans[0][0]) + abs(trans[-1][0])) + im_list[-1].shape[1]

            idx_max = np.argmax([im.shape[0] for im in im_list])
            height = im_list[idx_max].shape[0]
        else:
            width = int(abs(trans[0][0]) + im_list[-1].shape[1])

            idx_max = np.argmax([im.shape[0] for im in im_list])
            height = im_list[idx_max].shape[0]
    else:
        if len(im_list) == 2 and indice_ref == 0:
            idx_max = np.argmax([im.shape[1] for im in im_list])
            width = im_list[idx_max].shape[1]

            height = 2*int(abs(trans[0][1])) + im_list[-1].shape[0]

    if len(im_list[0].shape) == 3:
        mosaique = np.zeros((height, width, 3))
    else: 
        mosaique = np.zeros((height, width))

    if idx_max > indice_ref:
        origine_ref_x, origine_ref_y = int(abs(trans[0][0])), int(abs(trans[idx_max-1][1]))
    else:
        origine_ref_x, origine_ref_y = int(abs(trans[0][0])), int(abs(trans[idx_max][1]))

    for idx_im, im in tqdm(enumerate(im_list)):
        if idx_im == indice_ref:
            begin_x = origine_ref_x#abs(int(trans[0][0]))
            begin_y = origine_ref_y#abs(int(trans[0][1]))
            
        elif idx_im < indice_ref:
            begin_x = origine_ref_x + int(trans[idx_im][0])
            begin_y = origine_ref_y + int(trans[idx_im][1])
            
        else:
            begin_x = origine_ref_x + int(trans[idx_im-1][0])
            begin_y = origine_ref_y + int(trans[idx_im-1][1])

        for col in range(im.shape[1]):
            for row in range(im.shape[0]):
                count_mosaique = 0
                count_im = 0
                if len(im.shape) == 3:
                    for ch in range(im.shape[2]):
                        if mosaique[row+begin_y, col+begin_x, ch] < 0.08:
                            count_mosaique+=1
                        if im[row, col, ch] < 0.08:
                            count_im+=1
                    if count_mosaique >=2 :
                        mosaique[row+begin_y, col+begin_x] = im[row, col]
                    elif count_mosaique <=1 and count_im <=1: 
                        mosaique[row+begin_y, col+begin_x] = im[row, col]*0.2 + mosaique[row+begin_y, col+begin_x]*0.8
                else:
                    # if mosaique[row+begin_y, col+begin_x] < 0.08:
                    #     mosaique[row+begin_y, col+begin_x] = im[row, col]
                    # else:
                    #     mosaique[row+begin_y, col+begin_x] = im[row, col]*0.2 + mosaique[row+begin_y, col+begin_x]*0.8
                    if im[row, col] > 0.1:
                        if idx_im == indice_ref:
                            mosaique[row+begin_y, col+begin_x] = im[row, col]
                        elif nb_im > 3 and (idx_im < indice_ref -1 or idx_im > indice_ref +1):
                            mosaique[row+begin_y, col+begin_x] = im[row, col]
                        else:
                            mosaique[row+begin_y, col+begin_x] = im[row, col]*0.2 + mosaique[row+begin_y, col+begin_x]*0.8 

    
    plt.imshow((mosaique - mosaique.min())/(mosaique.max() - mosaique.min()))
    plt.show()

    return (mosaique - mosaique.min())/(mosaique.max() - mosaique.min())


if __name__ == '__main__':

    # fname1 = '../images/1-PartieManuelle/Serie1/IMG_2415.JPG'
    # fname2 = '../images/1-PartieManuelle/Serie1/IMG_2416.JPG'
    # fname3 = '../images/1-PartieManuelle/Serie1/IMG_2417.JPG'
    # pts1_12 = ptsread('../images/1-PartieManuelle/Serie1/pts_serie1/pts1_12.txt')
    # pts2_12 = ptsread('../images/1-PartieManuelle/Serie1/pts_serie1/pts2_12.txt')
    # pts2_32 = ptsread('../images/1-PartieManuelle/Serie1/pts_serie1/pts2_32.txt')
    # pts3_32 = ptsread('../images/1-PartieManuelle/Serie1/pts_serie1/pts3_32.txt')
    # dict_pts = {'pts1_12':pts1_12[:4], 'pts2_12':pts2_12[:4], 'pts2_32':pts2_32[5:-1], 'pts3_32':pts3_32[5:-1]}


    im1 = imread(fname1)
    im2 = imread(fname2)
    im3 = imread(fname3)
    im_list = [im1, im2, im2, im3]
    keys = list(dict_pts.keys())

    pts1_12norm, T112 = norm_pts(dict_pts['pts1_12'])
    pts2_12norm, T212 = norm_pts(dict_pts['pts2_12'])
    pts2_32norm, T232 = norm_pts(dict_pts['pts2_32'])
    pts3_32norm, T332 = norm_pts(dict_pts['pts3_32'])

    test_estimH()

    H12norm = estimH(pts1_12norm, pts2_12norm)
    H12 = inv(T212)@H12norm@T112

    H32norm = estimH(pts3_32norm, pts2_32norm)
    H32 = inv(T232)@H32norm@T332

    im1Trans, new_origine1 = appliqueTransformation(im1, H12)
    im3Trans, new_origine2 = appliqueTransformation(im3, H32)
    im_ref = (im2- im2.min())/(im2.max() - im2.min())

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    axs[0].imshow(sk.img_as_ubyte(im1Trans))
    axs[1].imshow(sk.img_as_ubyte(im_ref))
    axs[2].imshow(sk.img_as_ubyte(im3Trans))
    plt.savefig("../resultats/figSerie1.png", bbox_inches='tight')
    plt.show()

    im_list = [im1Trans, im_ref, im3Trans]
    trans = [new_origine1, new_origine2]
    mosaique = getmosaique(im_list, trans, 1)

    mosaique = sk.img_as_ubyte(mosaique)
    name = input("Nom du fichier (.png): ")
    fname = '../resultats/'+name
    skio.imsave(fname, mosaique)

    ##  Serie 3

    fname1 = '../images/1-PartieManuelle/Serie3/IMG_2409.JPG'
    fname2 = '../images/1-PartieManuelle/Serie3/IMG_2410.JPG'
    fname3 = '../images/1-PartieManuelle/Serie3/IMG_2411.JPG'
    im1 = imread(fname1)
    im2 = imread(fname2)
    im3 = imread(fname3)
    dict_pts = {'pts1_12':None, 'pts2_12':None, 'pts2_32':None, 'pts3_32':None}

    im_list = [im1, im2, im2, im3]
    keys = list(dict_pts.keys())

    for idx, im in enumerate(im_list):
        fig, ax = subplots()
        ax.imshow(im)
        ax.set_title(keys[idx])
        cursor = Cursor(ax, im.shape)
        connect('button_press_event', cursor.mouseclick)
        show()
        dict_pts[keys[idx]] = cursor.convert_to_numpy_array()

    fig, axs = subplots(1, 3)
    axs[0].imshow(im1)
    axs[0].scatter(dict_pts['pts1_12'][:, 0],dict_pts['pts1_12'][:, 1], color='red', marker='+')
    axs[1].imshow(im2)
    axs[1].scatter(dict_pts['pts2_12'][:, 0], dict_pts['pts2_12'][:, 1], color='red', marker='+')
    axs[1].scatter(dict_pts['pts2_32'][:,0], dict_pts['pts2_32'][:,1], color='blue', marker='+')
    axs[2].imshow(im3)
    axs[2].scatter(dict_pts['pts3_32'][:, 0], dict_pts['pts3_32'][:, 1], color='blue', marker='+')
    plt.savefig("../resultats/figSerie3.png", bbox_inches='tight')
    plt.show()
    

    pts1_12norm, T112 = norm_pts(dict_pts['pts1_12'])
    pts2_12norm, T212 = norm_pts(dict_pts['pts2_12'])
    pts2_32norm, T232 = norm_pts(dict_pts['pts2_32'])
    pts3_32norm, T332 = norm_pts(dict_pts['pts3_32'])

    H12norm = estimH(pts1_12norm, pts2_12norm)
    H12 = inv(T212)@H12norm@T112
    H32norm = estimH(pts3_32norm, pts2_32norm)
    H32 = inv(T232)@H32norm@T332

    im1Trans, new_origine1 = appliqueTransformation(im1, H12)
    im3Trans, new_origine2 = appliqueTransformation(im3, H32)
    im_ref = (im2- im2.min())/(im2.max() - im2.min())

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    axs[0].imshow(im1Trans)
    axs[1].imshow(im_ref)
    axs[2].imshow(im3Trans)
    plt.savefig("../resultats/figSerie3_2.png", bbox_inches='tight')
    plt.show()
    


    im_trans = [im1Trans, im_ref, im3Trans]
    trans = [new_origine1, new_origine2]
    mosaique = getmosaique(im_trans, trans, 1)
 
    mosaique = sk.img_as_ubyte(mosaique)
    name = input("Nom du fichier (.png): ")
    fname = '../resultats/'+name
    skio.imsave(fname, mosaique)


    ## Serie 2

    # fname1 = '../images/1-PartieManuelle/Serie2/IMG_2426.JPG'
    # fname2 = '../images/1-PartieManuelle/Serie2/IMG_2425.JPG'
    # im1 = imread(fname1)
    # im2 = imread(fname2)

    # dict_pts = {'pts1_12':None, 'pts2_12':None}
    # im_list = [im1, im2]
    # keys = list(dict_pts.keys())

    # for idx, im in enumerate(im_list):
    #     fig, ax = subplots()
    #     ax.imshow(im)
    #     ax.set_title(keys[idx])
    #     cursor = Cursor(ax, im.shape)
    #     connect('button_press_event', cursor.mouseclick)
    #     show()
    #     dict_pts[keys[idx]] = cursor.convert_to_numpy_array()
    
    # fig, axs = subplots(1, 2)
    # axs[0].imshow(im1)
    # axs[0].scatter(dict_pts['pts1_12'][:, 0],dict_pts['pts1_12'][:, 1], color='red', marker='+')
    # axs[1].imshow(im2)
    # axs[1].scatter(dict_pts['pts2_12'][:, 0], dict_pts['pts2_12'][:, 1], color='red', marker='+')
    # plt.savefig("../resultats/figSerie2.png", bbox_inches='tight')
    # plt.show()
    
    # pts1_12norm, T112 = norm_pts(dict_pts['pts1_12'])
    # pts2_12norm, T212 = norm_pts(dict_pts['pts2_12'])

    # H12norm = estimH(pts1_12norm, pts2_12norm)
    # H12 = inv(T212)@H12norm@T112

    # im1Trans, new_origine1 = appliqueTransformation(im1, H12)
    # im_ref = (im2- im2.min())/(im2.max() - im2.min())

    # fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    # axs[0].imshow(im1Trans)
    # axs[1].imshow(im_ref)
    # plt.savefig("../resultats/figSerie2_2.png", bbox_inches='tight')
    # plt.show()
    

    # im_trans = [im1Trans, im_ref]
    # trans = [new_origine1]
    # mosaique = getmosaique(im_trans, trans, 1)
 
    # mosaique = sk.img_as_ubyte(mosaique)
    # name = input("Nom du fichier (.png): ")
    # fname = '../resultats/'+name
    # skio.imsave(fname, mosaique)