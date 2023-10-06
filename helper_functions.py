import numpy as np


def distribution_sample(sinogram, N):
    max = np.max(sinogram)
    sinogram = sinogram/max
    i=0
    xs = []
    height, width = sinogram.shape
    while i<N:
        # Generate random indices for row and column
        random_row = np.random.randint(0, height)
        random_col = np.random.randint(0, width)
        p = sinogram[random_row,random_col]
        if p <0:
            p=0
        binom = np.random.binomial(1, p,1) 
        if binom == 1:
            xs.append([random_row, random_col])
            i = i+1
    xs = np.array(xs)
    return xs


def shift_array(x, n):
    if n == 0:
        return x
    elif n > 0:
        # Shift to the right
        return np.concatenate([np.zeros(n), x[:-n]])
    else:
        # Shift to the left
        n = abs(n)
        return np.concatenate([x[n:],np.zeros(n)])



def center(array):
    overall_sum = sum(array)
    center = sum([w * x for w, x in zip(array, range(len(array)))])
    return center/overall_sum