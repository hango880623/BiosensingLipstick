import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave, imshow
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import threshold_otsu
from mpl_toolkits.mplot3d import Axes3D

from skimage import color
import os

import cv2

def imgSeg(file_path):
    # Load Original Image
    img_test1 = imread(file_path)
    img_test1_gs = rgb2gray(img_test1)

    # Using Otsu's Method
    th_otsu = threshold_otsu(img_test1_gs)
    img_test1_otsu = img_test1_gs < th_otsu

    # Mask the original image based on the binary threshold
    img_masked = np.copy(img_test1)
    img_masked[~img_test1_otsu] = 0  # Set non-lips pixels to black (or any other desired color)
    return img_masked
    # # Plot
    # fig, ax = plt.subplots(1, 4, figsize=(24, 6))
    # fig.subplots_adjust(wspace=0.1)
    # ax[0].set_title("Original Image")
    # ax[0].imshow(img_test1)
    # ax[0].set_axis_off()
    # ax[1].set_title(f"Threshold (Value = {th:.2f})")
    # ax[1].imshow(img_test1_bn, cmap='gray')
    # ax[1].set_axis_off()
    # ax[2].set_title(f"Masked Image")
    # ax[2].imshow(img_masked)
    # ax[2].set_axis_off()
    # ax[3].set_title(f"Otsu's Method (Threshold = {th_otsu:.2f})")
    # ax[3].imshow(img_test1_otsu, cmap='gray')
    # ax[3].set_axis_off()
    # plt.savefig('./plot/result_plot.png', bbox_inches='tight')

def toLAB_dots(ax, color_list, label):
    # Convert LAB color list to numpy array
    lab_array = np.array(color_list)

    # Extract LAB values
    L_values = lab_array[:, 0]
    A_values = lab_array[:, 1]
    B_values = lab_array[:, 2]

    # Plot LAB dots
    ax.scatter(xs=A_values, ys=B_values, zs=L_values, s=100, label=label, marker='o')

    # Set axis labels
    ax.set_xlabel('A')
    ax.set_ylabel('B')
    ax.set_zlabel('L')

def show_Ours(lips_color_dic, labels):
    # Plot LAB dots
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('LAB Color Space: Human Lips')

    for i, (key, color_list) in enumerate(lips_color_dic.items()):
        toLAB_dots(ax, color_list, labels[i])

    ax.legend()
    plt.savefig('./plot/lab_human.png', bbox_inches='tight')

def process_images(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Construct full file paths
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)

            # Apply segmentation and save the modified image
            img_masked = imgSeg(input_file_path)
            imsave(output_file_path, img_masked)

if __name__ == "__main__":
    # imgSeg('./content/#6Lips/lips/55/p_2_55_6500.jpg')
    lips_color_dic = {0: [[30, 27, 13],[24, 25, 13],[26, 24, 10],[32, 33, 15],[22, 17, 5]],     # Shuyi
                      1: [[32, 47, 42],[35, 49, 45],[36, 48, 44],[36, 47, 46],[35, 41, 42]],    # Katia
                      2: [[40, 48, 46],[33, 42, 40],[31, 41, 39],[32, 39, 39],[31, 38, 37]],    # Howard
                      3: [[26, 40, 34],[30, 46, 40],[27, 39, 35],[26, 42, 36],[18, 28, 24]]}    # Nicole
                    #   4: [[42, 48, 22],[30, 45, 22],[39, 46, 16],[48, 52, 23],[35, 35, 8]],     # Shuyi-1
                    #   5: [[10, 24, 9],[6, 19, 6],[6, 13, 3],[9, 21, 8],[8, 22, 7]],}
    label_human = ['Shuyi', 'Katia', 'Howard', 'Nicole', 'Shuyi-1']

    ph_color_dic = {0: [[30, 27, 13],[32, 47, 42],[40, 48, 46],[26, 40, 34]],    # ph 5.5
                    1: [[24, 25, 13],[35, 49, 45],[33, 42, 40],[30, 46, 40]],    # ph 6.0
                    2: [[26, 24, 10],[36, 48, 44],[31, 41, 39],[27, 39, 35]],    # ph 6.5
                    3: [[32, 33, 15],[36, 47, 46],[32, 39, 39],[26, 42, 36]],    # ph 7.0
                    4: [[22, 17, 5],[35, 41, 42],[31, 38, 37],[18, 28, 24]]}     # ph 8.0
    label_pH = ['ph 5.5', 'ph 6.0', 'ph 6.5', 'ph 7.0', 'ph 8.0']

    lips_color_dic = {0: ['[26, 25, 10]', '[29, 27, 11]', '[25, 22, 7]', '[22, 16, 3]', '[31, 25, 9]'],     # Shuyi
                      1: ['[24, 25, 8]', '[22, 24, 8]', '[24, 22, 7]', '[20, 18, 3]', '[25, 24, 9]'],    # Katia
                      2: ['[23, 16, 6]', '[22, 18, 7]', '[22, 15, 3]', '[23, 20, 8]', '[27, 24, 10]'],    # Howard
                      3: ['[19, 23, 9]', '[16, 18, 4]', '[13, 12, 0]', '[21, 24, 8]', '[19, 22, 8]']}    # Nicole
    
    ph_color_dic_pixel = {0: ['[19, 22, 8]', '[27, 24, 10]', '[25, 24, 9]', '[31, 25, 9]'],    # ph 5.5
                    1: ['[24, 25, 8]', '[26, 25, 10]', '[21, 24, 8]', '[23, 20, 8]'],    # ph 6.0
                    2: ['[22, 18, 7]', '[16, 18, 4]', '[25, 22, 7]', '[24, 22, 7]'],    # ph 6.5
                    3: ['[23, 16, 6]', '[19, 23, 9]', '[29, 27, 11]', '[22, 24, 8]'],    # ph 7.0
                    4: ['[22, 15, 3]', '[13, 12, 0]', '[22, 16, 3]', '[20, 18, 3]']}     # ph 8.0
    
    show_Ours(lips_color_dic,label_human)
    


    
