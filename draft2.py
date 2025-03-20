import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy.signal import convolve2d
import os

folder_path = "/Users/rezajabbir/Documents/HEP/23A/MSSH35/project_sh35/images"
image_paths = [
    'photo_1.jpg', 'photo_2.jpg', 'photo_3.jpg',
    'photo_4.jpg', 'photo_5.jpg', 'photo_6.jpg', 'photo_7.jpg'
]

images = [io.imread(os.path.join(folder_path, img)) for img in image_paths]

# Define kernels
gaussian_kernel = np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]]) / 9

sobel_x = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])

emboss_kernel = np.array([[-2, -1, 0],
                          [-1, 1, 1],
                          [0, 1, 2]])

brushstroke_kernel = np.array([[-1, -1, -1],
                               [-1, 8, -1],
                               [-1, -1, -1]])

oil_painting_kernel = np.array([[1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]]) / 9

watercolor_kernel = np.array([[1, 0, -1],
                               [0, 1, 0],
                               [-1, 0, 1]])

texture_kernel = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])

random_kernel = np.array([[-1, -1, -1],
                          [-1, 5, -1],
                          [-1, -1, -1]])

# Create filtered images for each filter and each original image
titles = ['Original', 'Gaussian Blur', 'Edge Detection', 'Embossed', 'Brushstroke', 'Oil Painting', 'Watercolor', 'Texture', 'Random']

for index, img in enumerate(images):
    plt.figure(figsize=(15, 9))

    filtered_images = [img]
    filters = [
        gaussian_kernel, sobel_x, emboss_kernel,
        brushstroke_kernel, oil_painting_kernel,
        watercolor_kernel, texture_kernel, random_kernel
    ]

    for filt in filters:
        filtered = np.zeros_like(img)
        for channel in range(3):
            filtered[:, :, channel] = convolve2d(img[:, :, channel], filt, mode='same', boundary='wrap')

        filtered_images.append(filtered)

    for i, filtered_img in enumerate(filtered_images):
        plt.subplot(3, 3, i + 1)
        plt.imshow(filtered_img)
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.suptitle(f"Image {index+1}", fontsize=8)
    plt.show()
