import skimage as ski
from scipy.signal import convolve2d
from matplotlib import pyplot as plt
import os
import numpy as np

# Flou Gaussien
# adoucit l'image, réduit le bruit et crée un effet de rêve.



# Edge detection (Sobel)
# Accentue les bords et les contours de l'image, lui donnant un aspect de croquis.
edge_detection = np.array([[[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1],
                            [1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1],
                            [1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]]])

# Emboss
# Ajoute un effet 3D à l'image en soulignant les bords et en créant des ombres.


# Oil Painting Filter
# Simule l'aspect d'une peinture à l'huile en réduisant les détails et en accentuant les coups de pinceau.

# Watercolor Filter
# Imite l'aspect d'une aquarelle en mélangeant les couleurs et en ajoutant de la texture.


folder_path = "/Users/rezajabbir/Documents/HEP/23A/MSSH35/project_sh35/images"
image1_path = os.path.join(folder_path, 'photo_1.jpg')
image2_path = os.path.join(folder_path, 'photo_2.jpg')
image3_path = os.path.join(folder_path, 'photo_3.jpg')
image4_path = os.path.join(folder_path, 'photo_4.jpg')
image5_path = os.path.join(folder_path, 'photo_5.jpg')
image6_path = os.path.join(folder_path, 'photo_6.jpg')
image7_path = os.path.join(folder_path, 'photo_7.jpg')

images_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.jpg')]

folder_name = "image1_filtered"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

image1 = ski.io.imread(image1_path)
filtered_image = np.zeros_like(image1)  # To store the filtered image
for channel in range(3):  # Loop à travers les canaux RGB
    filtered_image[:, :, channel] = convolve2d(image1[:, :, channel], edge_detection[:, :, channel], mode='same', boundary='symm')

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.title("Image 1 d'origine")
plt.imshow(image1)

plt.subplot(1, 2, 2)
plt.title('Image 1 filtrée')
plt.imshow(filtered_image)  # afficher l'image filtrée

plt.tight_layout()
plt.show()

image2 = ski.io.imread(image2_path)
filtered_image2 = np.zeros_like(image2)  # To store the filtered image
for channel in range(3):  # Loop à travers les canaux RGB
    filtered_image2[:, :, channel] = convolve2d(image2[:, :, channel], edge_detection[:, :, channel], mode='same', boundary='symm')

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.title("Image 2 d'origine")
plt.imshow(image2)

plt.subplot(1, 2, 2)
plt.title('Image 2 filtrée')
plt.imshow(filtered_image2)  # afficher l'image filtrée

plt.tight_layout()
plt.show()

image3 = ski.io.imread(image3_path)
filtered_image3 = np.zeros_like(image3)  # To store the filtered image
for channel in range(3):  # Loop à travers les canaux RGB
    filtered_image3[:, :, channel] = convolve2d(image3[:, :, channel], edge_detection[:, :, channel], mode='same', boundary='symm')

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.title("Image 3 d'origine")
plt.imshow(image3)

plt.subplot(1, 2, 2)
plt.title('Image 3 filtrée')
plt.imshow(filtered_image3)  # afficher l'image filtrée

plt.tight_layout()
plt.show()