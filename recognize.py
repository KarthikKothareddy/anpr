

# contains all the test runs
import cv2
import imutils
import random
from imutils import paths
import numpy as np
from utils.display_utils import display
from license.plate import LicensePlateDetector


def localize_and_draw(image_path):
    # load the image
    image = cv2.imread(image_path)
    print(image_path)

    # if the width is greater than 640 pixels, then resize the image
    image = imutils.resize(image, width=640) if image.shape[1] > 640 else image

    # initialize the license plate detector and detect the license plates and characters
    lpd = LicensePlateDetector(image, mode="bgr")
    plates = lpd.detect()

    # loop over the license plate regions and draw the bounding box surrounding the
    # license plate
    """
    for lp_box in plates:
        lp_box = np.array(lp_box).reshape((-1, 1, 2)).astype(np.int32)
        cv2.drawContours(image, [lp_box], -1, (0, 255, 0), 2)
    
    """
    print(type(plates))
    print(len(plates))

    # show
    #display(plates, flip=False, cmap="gray")


paths = random.sample(list(paths.list_images("./data")), 2)
for path in paths:
    # print(path)
    localize_and_draw(path)


