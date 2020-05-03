import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt


def display(image, flip=True, cmap=None, figsize=(6, 6), **kwargs):
    if flip:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # adjustment for histograms
    if kwargs.get("is_histogram", False):
        plt.figure(figsize=figsize)
        plt.plot(image)
    else:
        width = image.shape[1]
        height = image.shape[0]
        margin = 50
        # dots per inch
        dpi = 100.
        # inches
        figsize = ((width+2*margin)/dpi, (height+2*margin)/dpi)
        # axes ratio
        left = margin/dpi/figsize[0]
        bottom = margin/dpi/figsize[1]
        fig = plt.figure(figsize=figsize, dpi=int(dpi))
        fig.subplots_adjust(left=left, bottom=bottom, right=1.-left, top=1.-bottom)
        _ = plt.imshow(image, cmap=cmap)
        plt.axis("off")
    plt.title(kwargs.get("title", None))
    plt.xlabel(kwargs.get("xlabel", None))
    plt.ylabel(kwargs.get("ylabel", None))
    plt.xlim(kwargs.get("xlim", None))
    plt.show()
