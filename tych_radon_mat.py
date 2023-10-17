import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator
import scipy.sparse as sp
import scipy.sparse.linalg as spla



def get_regularized_solution(lamda, theta, image_shape, sinogram):
    sinogram2 = sinogram.ravel()
    R = get_radon_matrix(theta, sinogram.shape, image_shape)
    Rt = R.T
    C = lamda * np.eye(image_shape[0]*image_shape[1])
    D = Rt @ R
    A = D + C
    x, info = cg(A, Rt @ sinogram2, tol=1e-3)
    im = np.reshape(x,image_shape)
    plt.imshow(im, origin='lower')
    plt.axis('off')
    plt.title(f"Reconstruction for λ={lamda}")
    plt.colorbar()
    plt.show()
    return im



def get_radon_matrix(theta, sinogram_shape ,image_shape):
    def matvec(v):
        im = np.reshape(v,image_shape)
        sinogram = radon(im, theta=theta)
        return sinogram.flatten()
    R = LinearOperator((sinogram_shape[0]*sinogram_shape[1], image_shape[0]*image_shape[1]), matvec=matvec)

    # create the unit bases
    unit_bases = [np.zeros(image_shape) for i in range(image_shape[0] * image_shape[1])]
    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            index = i * image_shape[1] + j
            unit_bases[index][i, j] = 1
    # apply R to each unit basis and stack the resulting columns
    R_columns = [R.dot(unit_base.ravel()) for unit_base in unit_bases]
    dense_representation = np.column_stack(R_columns)
    return dense_representation
    

def lamda_max_function(lamda, theta, image_shape, sinogram):
    x = get_regularized_solution(lamda, theta, image_shape, sinogram)
    Ax = radon(x, theta=theta)
    plt.imshow(Ax)
    plt.colorbar()
    plt.show()
    first_term = np.linalg.norm(Ax.ravel()-sinogram.ravel())
    first_term = np.linalg.norm(Ax.ravel()-sinogram.ravel())
    first_term = first_term*first_term
    f = x.ravel()
    norm = np.linalg.norm(f)
    second_term = lamda * norm * norm
    H = get_radon_matrix(theta, sinogram.shape ,image_shape)
    sparse_H = sp.csr_matrix(H)
    matrix = (1/lamda) * sparse_H.T @ sparse_H + np.eye(image_shape[0] * image_shape[1])
    sparse_matrix = sp.csr_matrix(matrix)
    eigvals = spla.eigsh(sparse_matrix, k=200, which='LM', return_eigenvectors=False, tol=25)
    det = np.log(eigvals)
    third_term = np.sum(det)
    return first_term + second_term + third_term, x



'''
circle = CT_Object(np.zeros([32,32]))
circle.add_circ_im(0.2,0.2,0.25,1)

# Load image
image = circle.image

# Compute Radon transform
theta = np.linspace(0., 180., 90, endpoint=False)
sinogram = radon(image, theta=theta)

# Display original image and sinogram
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
ax1.imshow(image)
ax1.set_title('Originales Bild')
ax2.imshow(sinogram, extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')
ax2.set_title('Radon transform\n(Sinogram)')
plt.show()


def matvec(v):
    im = np.reshape(v,image.shape)
    sinogram = radon(im, theta=theta)
    return sinogram.flatten()



R = LinearOperator((sinogram.shape[0]*sinogram.shape[1], image.shape[0]*image.shape[1]), matvec=matvec)

sinogram2 = R.dot(image.ravel())
sinogram3 = np.reshape(sinogram2, sinogram.shape)


# create the unit bases
unit_bases = [np.zeros(image.shape) for i in range(image.shape[0] * image.shape[1])]
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        index = i * image.shape[1] + j
        unit_bases[index][i, j] = 1

# apply R to each unit basis and stack the resulting columns
R_columns = [R.dot(unit_base.ravel()) for unit_base in unit_bases]
dense_representation = np.column_stack(R_columns)
R = dense_representation

Rt = R.T

alpha = 0.05
C = alpha * np.eye(image.shape[0]*image.shape[1])
D = Rt @ R
A = D + C

x, info = cg(A, Rt @ sinogram2, tol=1e-6)

im = np.reshape(x,image.shape)






# Display original image and sinogram
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
ax1.imshow(im)
ax1.set_title('Neues Bild')
ax2.imshow(sinogram, extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')
ax2.set_title('Radon transform\n(Sinogram)')
plt.show()

'''