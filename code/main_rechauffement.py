import numpy as np
from numpy.linalg import inv
import skimage as sk
import skimage.io as skio
from imageio.v2 import imread
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

def appliqueTransformation(im, H):
    if (len(im.shape)) == 3:
        row, col, ch = im.shape
    else:
         row, col = im.shape
    corners = np.array([[0, col, 0, col], [0, 0, row, row], [1, 1, 1, 1]])
    new_corners = H @ corners
    new_corners = new_corners/new_corners[2]
    max_x, min_x = np.amax(new_corners[0]), np.amin(new_corners[0])
    max_y, min_y = np.amax(new_corners[1]), np.amin(new_corners[1])
    new_origine = np.array([[int(min_x)], [int(min_y)], [0]])

    if (len(im.shape)) == 3:
        imgTrans = np.zeros((int(max_y-min_y), int(max_x-min_x), ch))
    else:
        imgTrans = np.zeros((int(max_y-min_y), int(max_x-min_x)))

    x, y = np.arange(imgTrans.shape[1]), np.arange(imgTrans.shape[0])
    x_coord, y_coord = np.meshgrid(x, y)
    coord = np.vstack((x_coord.flatten(), y_coord.flatten()))
    coord_homo = np.vstack((coord, np.ones(coord.shape[1])))

    og_coord = inv(H) @ (coord_homo+new_origine)
    og_coord = og_coord/og_coord[2]
    og_x, og_y = og_coord[0][:], og_coord[1][:]

    if (len(im.shape)) == 3:
        channels = [RectBivariateSpline(np.arange(row), np.arange(col), im[:,:,i]) for i in range(ch)]
        for idx, c in enumerate(channels):
            z = c.ev(og_y, og_x)
            for idx_z,_ in enumerate(coord[0]):
                j = coord[0][idx_z]
                i = coord[1][idx_z]
                if 0<=og_x[idx_z]<=col and 0<=og_y[idx_z]<=row:
                    imgTrans[i][j][idx] = z[idx_z]
                else:
                    imgTrans[i][j][idx] = 0

    else:
        channels = [RectBivariateSpline(np.arange(row), np.arange(col), im[:,:])]
        for idx, c in enumerate(channels):
            z = c.ev(og_y, og_x)
            for idx_z,_ in enumerate(coord[0]):
                j = coord[0][idx_z]
                i = coord[1][idx_z]
                if 0<=og_x[idx_z]<=col and 0<=og_y[idx_z]<=row:
                    imgTrans[i][j] = z[idx_z]
                else:
                    imgTrans[i][j] = 0

    
    return [(imgTrans - imgTrans.min())/(imgTrans.max() - imgTrans.min()), new_origine]


if __name__ == '__main__':
    fname1 = '../images/0-Rechauffement/pouliot'
    im = imread(fname1+'.jpg')

    H1 = np.array([[0.9752, 0.0013, -100.3164], [-0.4886, 1.7240, 24.8480], [-0.0016, 0.0004, 1.0000]])

    imgTrans,_ = appliqueTransformation(im, H1)
    fname = '../resultats/pouliot_h1.png'
    skio.imsave(fname, sk.img_as_ubyte(imgTrans))

    H2 = np.array([[0.1814, 0.7402, 34.3412], [1.0209, 0.1534, 60.3258], [0.0005, 0, 1.0000]])
    imgTrans,_ = appliqueTransformation(im, H2)
    fname = '../resultats/pouliot_h2.png'
    skio.imsave(fname, sk.img_as_ubyte(imgTrans))

    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(im)
    # plt.title('Image originale')
    # plt.subplot(1, 2, 2)
    # plt.imshow(imgTrans)
    # plt.title('Image transformée')
    # plt.show()