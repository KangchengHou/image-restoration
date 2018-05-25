from sklearn.neighbors import KDTree
import numpy as np


def knn_regression(img, n_neighbors=7, weights='distance'):
    res_img = img.copy()
    channels = img.shape[2]
    for c in range(channels):
        c_img = img[:, :, c]
        train_X = [[z[0], z[1]] for z in zip(*np.where(c_img != 0))]
        train_y = [c_img[i[0], i[1]] for i in train_X]
        test_X = [[z[0], z[1]] for z in zip(*np.where(c_img == 0))]
        train_X = np.array(train_X)
        train_y = np.array(train_y)
        test_X = np.array(test_X)
        kdtree = KDTree(train_X)
        dist, ind = kdtree.query(test_X, k=n_neighbors)
        # predict the result
        # weights is the inverse of the distance
        ww = 1 / dist
        if weights == 'distance':
            test_y = np.sum(train_y[ind] * ww, 1) / np.sum(ww, 1)
        else:
            test_y = np.sum(train_y[ind], 1) / n_neighbors 
        for i in range(len(test_X)):
            res_img[test_X[i][0], test_X[i][1], c] = test_y[i]
    res_img[res_img > 1.0] = 1.0
    res_img[res_img < 0.0] = 0.0
    return res_img