import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from scipy.signal import convolve2d
import os

folder_path = "/Users/rezajabbir/Documents/HEP/23A/MSSH35/project_sh35/images"
image1_path = os.path.join(folder_path, 'photo_1.jpg')
image2_path = os.path.join(folder_path, 'photo_2.jpg')
image3_path = os.path.join(folder_path, 'photo_3.jpg')
image4_path = os.path.join(folder_path, 'photo_4.jpg')
image5_path = os.path.join(folder_path, 'photo_5.jpg')
image6_path = os.path.join(folder_path, 'photo_6.jpg')
image7_path = os.path.join(folder_path, 'photo_7.jpg')

image = io.imread(image1_path)
image2 = io.imread(image2_path)
image3 = io.imread(image3_path)
image4 = io.imread(image4_path)
image5 = io.imread(image5_path)
image6 = io.imread(image6_path)
image7 = io.imread(image7_path)

images = [image, image2, image3, image4, image5, image6, image7]

# Define and apply the kernels

# Gaussian Blur Kernel : adoucit l'image, réduit le bruit et crée un effet "dreamy".
gaussian_kernel = np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]]) / 9
for elem in images:
    gaussian_blurred = np.zeros_like(elem)
    for channel in range(3):
        gaussian_blurred[:, :, channel] = convolve2d(elem[:, :, channel], gaussian_kernel, mode='same', boundary='wrap')


# Sobel Kernels for edge detection : Accentue les bords et les contours de l'image, lui donnant un aspect de croquis.
sobel_x = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])
edge_detection = np.zeros_like(image)
for channel in range(3):
    edge_detection[:, :, channel] = convolve2d(image[:, :, channel], sobel_x, mode='same', boundary='wrap')

# Emboss Kernel : # Ajoute un effet 3D à l'image en soulignant les bords et en créant des ombres.
emboss_kernel = np.array([[-2, -1, 0],
                          [-1, 1, 1],
                          [0, 1, 2]])
embossed = np.zeros_like(image)
for channel in range(3):
    embossed[:, :, channel] = convolve2d(image[:, :, channel], emboss_kernel, mode='same', boundary='wrap')

# Custom Brushstroke-Like Kernels : Design kernels that emphasize certain features or edges to mimic brushstrokes or artistic details.
brushstroke_kernel = np.array([[-1, -1, -1],
                               [-1, 8, -1],
                               [-1, -1, -1]])
brushstroke = np.zeros_like(image)
for channel in range(3):
    brushstroke[:, :, channel] = convolve2d(image[:, :, channel], brushstroke_kernel, mode='same', boundary='wrap')

# Oil Painting Effect : Simulate the look of an oil painting by reducing detail and enhancing color variations.
oil_painting_kernel = np.array([[1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]]) / 9
oil_painting = np.zeros_like(image)
for channel in range(3):
    oil_painting[:, :, channel] = convolve2d(image[:, :, channel], oil_painting_kernel, mode='same', boundary='wrap')

# Watercolor effect : Imitate the appearance of a watercolor painting by blending colors and adding texture.
watercolor_kernel = np.array([[1, 0, -1],
                               [0, 1, 0],
                               [-1, 0, 1]])
watercolor = np.zeros_like(image)
for channel in range(3):
    watercolor[:, :, channel] = convolve2d(image[:, :, channel], watercolor_kernel, mode='same', boundary='wrap')

# Texture Emphasis : Enhance the texture of the image to give it a more painterly feel.
texture_kernel = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])
texture = np.zeros_like(image)
for channel in range(3):
    texture[:, :, channel] = convolve2d(image[:, :, channel], texture_kernel, mode='same', boundary='wrap')

# Random filter
random_kernel = np.array([[-1, -1, -1],
                        [-1, 5, -1],
                        [-1, -1, -1]])
random = np.zeros_like(image)
for channel in range(3):
    random[:, :, channel] = convolve2d(image[:, :, channel], random_kernel, mode='same', boundary='wrap')

filters = [image, gaussian_blurred, edge_detection, embossed, brushstroke, oil_painting, watercolor, texture, random]
# Titres correspondant à chaque image
titles = ['Original', 'Gaussian Blur', 'Edge Detection', 'Embossed', 'Brushstroke', 'Oil Painting', 'Watercolor', 'Texture', "random"]

# Affichage des résultats sur un subplot
plt.figure(figsize=(13, 8))
for i in range(len(filters)):
    plt.subplot(3, 3, i + 1)
    plt.imshow(filters[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

