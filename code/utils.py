from numpy.random import permutation as randperm

def corrupt_img(img, mode):
    if mode == 'A':
        ratio = 0.8
    elif mode == 'B':
        ratio = 0.4
    elif mode == 'C':
        ratio = 0.6
    else:
        print('not implemented')
    
    rows, cols, channels = img.shape
    corr_img = img.copy()
    #  for every rows, add some noise
    sub_noise_num = int(round(ratio * cols))
    # for every channels, randomly choose some rows to remove
    for k in range(channels):
        for i in range(rows):
            tmp = randperm(cols)
            noise_idx = tmp[1:sub_noise_num]
            corr_img[i, noise_idx, k] = 0
    return corr_img