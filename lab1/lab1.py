from matplotlib import pyplot as plt
import math
import matplotlib.patches as mpatches
from skimage import data, io, filters,feature, color, feature, morphology
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.filters import roberts, sobel, prewitt
from scipy import ndimage as ndi

def find_row(dict, name, th):

    # sort the input seq
    sorted_y = sorted(dict.items(), key=lambda x: x[1])
    #print("["+str(name)+"]:")
    input = []
    for k, v in sorted_y:
        #print(k, v)
        input.append(v)

    res = []
    for i in range(0, len(input)-1):
        #print(input[i], input[i + 1], abs(input[i] - input[i + 1]))

        # if two points coord difference is less than the threshold
        if abs(input[i] - input[i + 1]) < th:
            if input[i] not in res:
                res.append(input[i])
            res.append(input[i+1])
            if i == (len(input) - 2):
                if len(res) > 1:
                    print("Следующие точки по оси " + str(name) + " лежат примерно на одной прямой:")
                    for el in res:
                        for k in dict.keys():
                            if dict[k] == el:
                                print("[" + str(k) + "] ", el)
        else:
            if len(res)>1:
                print("Следующие точки по оси " + str(name) + " лежат примерно на одной прямой:")
                for el in res:
                    for k in dict.keys():
                        if dict[k] == el:
                            print("["+str(k)+"] ", el)
            res = []

# open the input image
image1 = io.imread("1.jpg")

# convert image from rgb to gray
image2 = color.rgb2gray(image1)

# save the gray copy of image into a file
io.imsave('rgb2gray.jpg', image2)

# find edges using Canny algorithm
edges1 = feature.canny(image2, sigma=4)

# filled edges using mathematical morphology
edges2 = ndi.binary_fill_holes(edges1)
io.imsave('canny.jpg', edges1)
#edges1 = filters.prewitt(image2)
#edges1 = filters.roberts(image2)
#edges1 = filters.sobel(image2)
io.imshow(edges2)
io.show()

# segment an image with image labelling
label_image = label(edges2)
#image_label_overlay = label2rgb(label_image, image=edges1)

# create plot for result
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(edges1)

coord_y = {}
coord_x = {}

i = 1

# find regions and center of each region after image labelling
for region in regionprops(label_image):

    # take regions with large enough areas
    if region.area >= 150:

        # draw rectangle around segmented objects
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='green', linewidth=1)

        ax.add_patch(rect)

        # find center points in segmented objects
        y0, x0 = region.centroid
        coord_y[i] = y0
        coord_x[i] = x0
        #x0 = minc + (maxc - minc) / 2
        #y0 = minr + (maxr - minr) / 2

        ax.plot(x0, y0, 'rs', markersize=10)
        ax.text(x0, y0, str(i), fontsize=12, color = "white")

        print("[%s] x0: %s y0: %s" % (i, x0, y0,))
        i+=1


# interface analysis
find_row(coord_y, "y", 5)
find_row(coord_x, "x", 5)

# set axis off
ax.set_axis_off()
plt.tight_layout()
plt.show()



