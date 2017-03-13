from matplotlib import pyplot as plt
import math
import matplotlib.patches as mpatches
from skimage import data, io, filters,feature, color
from skimage.measure import label, regionprops
import os
from skimage import novice
from skimage.morphology import disk, dilation, closing, square
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi

# plot two images for comparison
def plot_comparison(original, filtered, filter_name, name):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original ' +name)
    ax1.axis('off')
    ax1.set_adjustable('box-forced')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name + " " + name)
    ax2.axis('off')
    ax2.set_adjustable('box-forced')

# check the area of interest zone
def check_area(image, min_area, max_area):
    label_image = label(image)
    flag = False
    for region in regionprops(label_image):

        # take regions with large enough areas
        if region.area >= min_area and region.area <= max_area:
            flag = True
            break

    if(flag):
        return True
    else:
        return False

# search food
def search_food(image, r_min, r_max, g_min, g_max, b_min, b_max, name, min_area, max_area):

    print("Ищем элемент: " + name)

    # color channels
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    # mask for color channels
    mask_r = (r > r_min) & (r < r_max)
    mask_g = (g > g_min) & (g < g_max)
    mask_b = (b > b_min) & (b < b_max)

    # summary mask
    mask = mask_r & mask_g & mask_b
    print("rgb2gray")

    # convert from rgb to gray
    mask1 = color.rgb2gray(mask)

    # fill holes
    mask1 = ndi.binary_fill_holes(mask1)


    #io.imshow(mask1)
    #io.show()

    # kernel for morphology
    s = 5
    # selem = disk(s)
    selem = square(s)

    # dilation filter
    if not check_area(mask1, min_area, max_area):
        print("dilation "+str(s))
        mask2 = dilation(mask1, selem)
        mask2 = ndi.binary_fill_holes(mask2)
        #plot_comparison(mask1, mask2, 'dilation', name)
    else: mask2 = mask1

    # closing filter
    if not check_area(mask2, min_area, max_area):
        print("closing")
        # th = threshold_otsu(mask2)
        # mask3 = closing(mask2 > th, square(10))
        mask3 = closing(mask2, selem)
        mask3 = ndi.binary_fill_holes(mask3)
        #plot_comparison(mask2, mask3, 'closing', name)
    else: mask3 = mask2

    # segment an image with image labelling
    label_image = label(mask3)

    for region in regionprops(label_image):

        # take regions with large enough areas
        if region.area >= min_area and region.area <= max_area:

            # draw rectangle around segmented objects
            # minr, minc, maxr, maxc = region.bbox
            # rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='blue', linewidth=3)
            # ax.add_patch(rect)

            # find center points in segmented objects
            y0, x0 = region.centroid

            # draw the name of food and its area
            ax.text(x0, y0, name+"\n"+str(region.area), fontsize=12, color = "white")


# open the input image
image1 = io.imread(os.getcwd()+'/pool/Меню (3).jpg')

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image1)

# searching elements
search_food(image1, 150, 225, 50, 120, 20, 65, "Морковь", 300000, 600000)
search_food(image1, 90, 130, 60, 115, 35, 90, "Хлеб", 180000, 250000)
search_food(image1, 135, 205, 120, 185, 80, 135, "Картофель", 100000, 600000)
search_food(image1, 150, 175, 140, 165, 140, 150, "Рыба", 50000, 200000)
search_food(image1, 150, 190, 55, 85, 45, 70, "Помидор", 5000, 100000)
search_food(image1, 35, 60, 45, 70, 5, 25, "Огурец", 7000, 50000)
search_food(image1, 65, 125, 10, 25, 15, 30, "Кетчуп", 40000, 300000)

search_food(image1, 90, 110, 45, 80, 25, 50, "Котлета 1", 50000, 300000)
search_food(image1, 120, 140, 70, 110, 40, 60, "Котлета 2", 50000, 300000)

search_food(image1, 40, 55, 5, 15, 10, 20, "Компот 1", 40000, 200000)
search_food(image1, 70, 80, 5, 10, 15, 25, "Компот 2", 40000, 200000)
search_food(image1, 20, 30, 1, 10, 1, 10, "Компот 3", 40000, 200000)
search_food(image1, 135, 155, 65, 90, 10, 25, "Компот 4", 40000, 200000)

search_food(image1, 125, 160, 100, 145, 45, 85, "Суп 1", 100000, 800000)
search_food(image1, 60, 85, 50, 75, 25, 50, "Суп 2", 100000, 800000)

# set axis off
ax.set_axis_off()
plt.tight_layout()
plt.show()


# images = io.imread_collection(os.getcwd()+'/pool/*.JPG')
# print(len(images))
# for im in images:
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.imshow(im)
#     search_food(image1, 150, 225, 50, 120, 20, 65, "Морковь", 300000, 600000)
#     ax.set_axis_off()
#     plt.tight_layout()
#     plt.show()



