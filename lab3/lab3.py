from matplotlib import pyplot as plt
import math
import matplotlib.patches as mpatches
from skimage import io, color, feature, morphology, draw
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
from skimage.morphology import disk, dilation, closing, square, erosion
import os
import numpy as np

# Haar primitives
haar = {'haar1':
            np.array([[1,1,1,1,1,1,1,1,1,1],
                      [1,1,1,1,1,1,1,1,1,1],
                      [1,1,1,1,1,1,1,1,1,1],
                      [1,1,1,1,1,1,1,1,1,1],
                      [1,1,1,1,1,1,1,1,1,1],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0]]),
        'haar2':
            np.array([[1,1,1,1,1,1,1,1,1,1],
                      [1,1,1,1,1,1,1,1,1,1],
                      [1,1,1,1,1,1,1,1,1,1],
                      [0,0,1,1,1,1,1,1,1,1],
                      [0,0,0,0,1,1,1,1,1,1],
                      [0,0,0,0,0,0,1,1,1,1],
                      [0,0,0,0,0,0,0,0,1,1],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0]]),
        'haar3':
            np.array([[1,1,1,1,1,1,1,1,1,1],
                      [1,1,1,1,1,1,1,1,1,1],
                      [1,1,1,1,1,1,1,1,1,1],
                      [0,1,1,1,1,1,1,1,1,1],
                      [0,0,0,0,1,1,1,1,1,1],
                      [0,0,0,0,0,0,1,1,1,1],
                      [0,0,0,0,0,0,0,0,0,1],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0]]),

        'haar4':
            np.array([[1,1,1,1,1,0,0,0,0,0],
                      [1,1,1,1,1,1,0,0,0,0],
                      [1,1,1,1,1,1,1,0,0,0],
                      [1,1,1,1,1,1,1,1,0,0],
                      [1,1,1,1,1,1,1,1,1,0],
                      [0,1,1,1,1,1,1,1,1,1],
                      [0,0,1,1,1,1,1,1,1,1],
                      [0,0,0,1,1,1,1,1,1,1],
                      [0,0,0,0,1,1,1,1,1,1],
                      [0,0,0,0,0,1,1,1,1,1]]),
        'haar5':
            np.array([[1,0,0,0,0,0,0,0,0,0],
                      [1,1,0,0,0,0,0,0,0,0],
                      [1,1,1,0,0,0,0,0,0,0],
                      [1,1,1,1,0,0,0,0,0,0],
                      [1,1,1,1,1,0,0,0,0,0],
                      [1,1,1,1,1,1,0,0,0,0],
                      [1,1,1,1,1,1,1,0,0,0],
                      [1,1,1,1,1,1,1,1,0,0],
                      [1,1,1,1,1,1,1,1,1,0],
                      [1,1,1,1,1,1,1,1,1,1]]),
        'haar6':
            np.array([[0,0,0,0,0,0,0,0,1,1],
                      [0,0,0,0,0,0,0,0,1,1],
                      [0,0,0,0,0,0,0,1,1,1],
                      [0,0,0,0,0,0,0,1,1,1],
                      [0,0,0,0,0,0,1,1,1,1],
                      [0,0,0,0,0,0,1,1,1,1],
                      [0,0,0,0,0,1,1,1,1,1],
                      [0,0,0,0,0,1,1,1,1,1],
                      [0,0,0,0,1,1,1,1,1,1],
                      [0,0,0,0,1,1,1,1,1,1]]),
        'haar7':
            np.array([[1,1,1,1,1,1,1,1,1,1],
                      [1,1,1,1,1,1,1,1,1,1],
                      [1,1,1,1,1,1,1,1,1,1],
                      [0,0,0,1,1,1,1,1,1,1],
                      [0,0,0,0,0,0,1,1,1,1],
                      [0,0,0,0,0,0,0,0,0,1],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0]]),
        'haar8':
            np.array([[1,1,1,1,1,0,0,0,0,0],
                      [1,1,1,1,1,0,0,0,0,0],
                      [1,1,1,1,1,0,0,0,0,0],
                      [1,1,1,1,1,0,0,0,0,0],
                      [1,1,1,1,1,0,0,0,0,0],
                      [1,1,1,1,1,0,0,0,0,0],
                      [1,1,1,1,1,0,0,0,0,0],
                      [1,1,1,1,1,0,0,0,0,0],
                      [1,1,1,1,1,0,0,0,0,0],
                      [1,1,1,1,1,0,0,0,0,0]]),
        'haar9':
            np.array([[0,0,0,0,0,0,0,0,0,1],
                      [0,0,0,0,0,0,0,0,1,1],
                      [0,0,0,0,0,0,0,1,1,1],
                      [0,0,0,0,0,0,1,1,1,1],
                      [0,0,0,0,0,1,1,1,1,1],
                      [0,0,0,0,1,1,1,1,1,1],
                      [0,0,0,1,1,1,1,1,1,1],
                      [0,0,1,1,1,1,1,1,1,1],
                      [0,1,1,1,1,1,1,1,1,1],
                      [1,1,1,1,1,1,1,1,1,1]]),
        }

# show primitive
# io.imshow(haar['haar'])
# io.show()

# primitives searching
def primitive_search(img, haar_type, th, point, marker_size, draw_haar):

    # input image rows and cols calculating
    rows = img.shape[0]
    cols = img.shape[1]

    prim_amount = 0
    cur_haar = haar[haar_type] # init haar primitive


    print("%s - %s" % (haar_type, point))

    res = []

    # for each pixel in image with step of size window
    for row in range(0, rows, win_size):
        for col in range(0, cols, win_size):
            b = 0
            w = 0

            # for each pixel in window
            for px in range(0, win_size):
                for py in range(0, win_size):

                    # if pixel is under '1' in haar primitive
                    if cur_haar[py][px] == 1:
                        b += img[row + py][col + px]
                    # if pixel is under '0' in haar primitive
                    else:
                        w += img[row + py][col + px]

            if abs(b - w) > th:
                prim_amount += 1

                # add marker in window
                if not draw_haar:
                    # print(abs(b - w))

                    ax.plot(col + win_size/2, row + win_size/2, point, markersize=marker_size)
                    bus.append([col + win_size / 2, row + win_size / 2])

                # draw haar primitive
                else:
                    for px in range(0, win_size):
                        for py in range(0, win_size):
                            if cur_haar[py][px] == 1:
                                ax.plot(col + px, row + py, 'bs', markersize=1)

                            else:
                                ax.plot(col + px, row + py, 'ws', markersize=1)





            res.append(b - w)

    print("max: %s" % max(res))
    print("min: %s" % min(res))
    print("amount: %s" % prim_amount)

    # fig.savefig(haar_type + '.png')

def draw_pix(img, coord, r, g, b):

    rows = img.shape[0]
    cols = img.shape[1]

    for row in range(0, rows):
        for col in range(0, cols):
            if [col, row] in coord:
                if img[row, col, 0] != r and img[row, col, 1] != g and img[row, col, 2] != b:
                    # draw circle in interesting point
                    img[row, col, :] = (r, g, b)
                    rr, cc = draw.circle(row, col, 7, img.shape)
                    img[rr, cc, :] = (r, g, b)

    return img

# open the input image
image1 = io.imread(os.getcwd()+'/pool/007.jpg')

# convert image from rgb to gray
image2 = color.rgb2gray(image1)

win_size = 10


fig, ax = plt.subplots(figsize=(10, 10))

ax.imshow(image1)

# array for interesting points
bus = []

# searching haar primitives
# for test
# primitive_search(image2, 'haar1', 14, 'gD', 10, False)
# primitive_search(image2, 'haar2', 15, 'rs', 9, False)
# primitive_search(image2, 'haar3', 13, 'g^', 8, False)
# primitive_search(image2, 'haar4', 28, 'ys', 7, False)
# primitive_search(image2, 'haar5', 21, 'bs', 6, False)
# primitive_search(image2, 'haar6', 20, 'wD', 6, False)
# primitive_search(image2, 'haar7', 21, 'ws', 6, False)
# primitive_search(image2, 'haar8', 12, 'b^', 7, False)
# primitive_search(image2, 'haar9', 20, 'ws', 7, False)
# primitive_search(image2, 'haar7', 21, 'ro', 6, True)

primitive_search(image2, 'haar1', 14, 'ws', 7, False)
primitive_search(image2, 'haar2', 15, 'ws', 7, False)
primitive_search(image2, 'haar3', 13, 'ws', 7, False)
primitive_search(image2, 'haar4', 28, 'ws', 7, False)
primitive_search(image2, 'haar5', 21, 'ws', 7, False)
primitive_search(image2, 'haar6', 17, 'ws', 7, False)
primitive_search(image2, 'haar7', 21, 'ws', 7, False)
primitive_search(image2, 'haar8', 12, 'ws', 7, False)
primitive_search(image2, 'haar9', 20, 'ws', 7, False)

# draw interesting points
image3 = draw_pix(image1, bus, 255, 255, 255)

# help image for result
image4 = color.rgb2gray(image3)
io.imsave('rgb2gray.jpg', image4)
# image4 = ndi.binary_fill_holes(image4)

selem = disk(7)

# dilation morph
image5 = dilation(image4, selem)
io.imsave('dilation.jpg', image5)

# closing morph
image6 = closing(image5, selem)
io.imsave('closing.jpg', image6)

# threshold
# thresh = threshold_otsu(image2)
# binary = image2 > 0.5 * thresh

# segment an image with image labelling
label_image = label(image6)


# for each interesting region
for region in regionprops(label_image):

    # take regions with large enough areas
    if region.area > 2000 and region.area < 70000:

        # draw rectangle around segmented objects
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=1)

        # add rect to interesting area
        ax.add_patch(rect)
        y0, x0 = region.centroid
        ax.text(x0, y0, str(region.area), fontsize=10, color = "red")


# set axis off
ax.set_axis_off()
fig.savefig('result.png')
plt.show()



