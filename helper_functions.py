import numpy as np


def highest_non_zero_index(arr):
  # Find the highest index in the array that is non-zero
    l = len(arr) - 1
    while l > 0:
        if arr[l] != 0:
            return l
        l -= 1
    return l

def lowest_non_zero_index(arr):
  # Find the highest index in the array that is non-zero
    k = 0
    while k < len(arr):
        if arr[k] != 0:
            return k
        k += 1
    return k


def wasserstein_distance(hist1, hist2):
  # Normalize histograms
  hist1 = hist1 / np.sum(hist1)
  hist2 = hist2 / np.sum(hist2)

  # Calculate CDFs
  cdf1 = np.cumsum(hist1)
  cdf2 = np.cumsum(hist2)

  # Calculate Wasserstein distance
  W = np.sum(np.abs(cdf1 - cdf2))

  return W


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

    
def find_cusps(data, h):
    # Compute the derivative of the data using the difference quotient
    derivative = [(data[i+1] - data[i]) / h for i in range(len(data) - 1)]
    # Find points where the derivative is close to zero
    cusps = []
    for i, value in enumerate(derivative):
        if abs(value) < 0.01: 
            cusps.append(i)
 
    return cusps


def find_middle_values(lst):
    # Split the list into two sublists at the first occurrence of a gap of more than one element
    sublists = []
    current_sublist = []
    for i, value in enumerate(lst):
        if i > 0 and lst[i] - lst[i-1] > 1:
            sublists.append(current_sublist)
            current_sublist = []
        current_sublist.append(value)
    sublists.append(current_sublist)
 
    # Find the middle value of each sublist
    middle_values = []
    for sublist in sublists:
        middle_index = len(sublist) // 2
        middle_values.append(sublist[middle_index])
 
    return middle_values


def split_array(array, split_points):
    # Add the end of the array as a split point if it is not already in the list
    if split_points[-1] != len(array):
        split_points.append(len(array))
 
    # Split the array at each split point
    split_arrays = []
    start = 0
    for end in split_points:
        end = int(round(end))
        #print(end)
        split_arrays.append(array[start:end])
        start = end
 
    return split_arrays
