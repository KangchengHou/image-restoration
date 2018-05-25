import numpy as np

def normal_phi(x):
    def normal_pdf(x, mu, sigma):
        return np.exp(- np.power((x - mu), 2) / (2 * np.power(sigma, 2)))
    num = len(x)
    # number of basis functions
    basis_num = 50
    # the variance parameter in Gaussain regression
    sigma = 0.05
    phi_mu = np.linspace(0, 1, basis_num)
    # set the sigma to be the same
    
    phi_sigma = sigma * np.ones(basis_num)
    # compute parameters used for linear regression
    phi = np.hstack((np.ones([num, 1]), np.zeros([num, basis_num - 1])))
    for j in range(1, basis_num):
        phi[:, j] = normal_pdf(x, phi_mu[j-1], phi_sigma[j-1])
        
    return phi

def sigmoid_phi(x):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    num = len(x)
    # number of basis functions
    basis_num = 50
    # the variance parameter in Gaussain regression
    s = 0.05
    phi_mu = np.linspace(0, 1, basis_num - 1)
    # compute parameters used for linear regression
    phi = np.hstack((np.ones([num, 1]), np.zeros([num, basis_num - 1])))
    for j in range(1, basis_num):
        phi[:, j] = sigmoid((x - phi_mu[j-1]) / s)
    return phi



def regression_by_row(img, phi_func):

    rows, cols, channels = img.shape

    # the index as the variable
    x = np.array(range(cols))
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    res_img = img.copy()
    # for each channel
    for k in range(channels):
        # for each rows
        for i in range(rows):
            img_row = img[i, :, k]
            mis_idx = np.where(img_row == 0)[0]
            mis_num = len(mis_idx)
            # data index
            dd_idx = np.where(img_row != 0)[0]
            dd_num = len(dd_idx)
            
            phi = phi_func(x[dd_idx])                      
            w = np.dot(np.linalg.pinv(np.dot(phi.T, phi)), np.dot(phi.T, img[i,dd_idx, k]))

            # restore the missing values using the parameters
            phi1 = phi_func(x[mis_idx])
            res_img[i,mis_idx,k] = np.dot(phi1, w)
    res_img[res_img > 1.0] = 1.0
    res_img[res_img < 0.0] = 0.0
    return res_img
