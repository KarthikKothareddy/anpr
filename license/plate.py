from collections import namedtuple
from skimage.filters import threshold_local
from skimage import segmentation
from skimage import measure
from imutils import perspective
import numpy as np
import imutils
import cv2


LicensePlate = namedtuple("LicensePlateRegion", ["success", "plate", "thresh", "candidates"])


class LicensePlateDetector:

    def __init__(self, image, min_plate_w=60, min_plate_h=20, mode="bgr", num_chars=7, min_char_w=40):
        # license plate region
        self.image = image
        self.mode = mode
        self._to_gray = cv2.COLOR_BGR2GRAY if self.mode == "bgr" else cv2.COLOR_RGB2GRAY
        self.min_plate_w = min_plate_w
        self.min_plate_h = min_plate_h
        self.num_chars = num_chars
        self.min_char_w = min_char_w

    def detect(self):
        # wrapper around detect_plates
        regions = self.detect_plates()
        # loop over the license plate regions
        for region in regions:
            # detect character candidates in the current license plate region
            lp = self.detect_character_candidates(region)
            """
            # only continue if characters were successfully detected
            if lp.success:
                # yield a tuple of the license plate object and bounding box
                yield lp, region
            """
            return lp

    def detect_character_candidates(self, region):

        # apply a 4-point transform to extract the license plate
        plate = perspective.four_point_transform(self.image, region)

        # extract the Value component from the HSV color space and apply adaptive thresholding
        # to reveal the characters on the license plate
        V = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]
        T = threshold_local(V, 29, offset=15, method="gaussian")
        thresh = (V > T).astype("uint8") * 255
        thresh = cv2.bitwise_not(thresh)

        # resize the license plate region to a canonical size
        plate = imutils.resize(plate, width=400)
        thresh = imutils.resize(thresh, width=400)

        # perform a connected components analysis and initialize the mask to store the locations
        # of the character candidates
        labels = measure.label(thresh, connectivity=2, background=0)
        char_candidates = np.zeros(thresh.shape, dtype="uint8")

        # loop over the unique components
        for label in np.unique(labels):
            # if this is the background label, ignore it
            if label == 0:
                continue
            # otherwise, construct the label mask to display only connected components for the
            # current label, then find contours in the label mask
            label_mask = np.zeros(thresh.shape, dtype="uint8")
            label_mask[labels == label] = 255
            contours = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            return contours

    def detect_plates(self):
        # init rectangle kernel
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        # init square kernel
        square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # to grayscale
        gray = cv2.cvtColor(self.image, self._to_gray)
        # apply blackhat to reveal darker regions on light background
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)

        # find regions in the image that are light
        # apply morphology CLOSE
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, square_kernel)
        # threshold the image between 50 and 255 because our target plates are mostly white in color
        light = cv2.threshold(light, 50, 255, cv2.THRESH_BINARY)[1]

        # compute the Scharr gradient representation of the blackhat image in the x-direction
        grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad_x = np.absolute(grad_x)
        min_val, max_val = (np.min(grad_x), np.max(grad_x))
        # min/max scaling to [0, 255]
        grad_x = (255 * ((grad_x - min_val) / (max_val - min_val))).astype("uint8")

        # blur the gradient representation,
        grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
        # apply a morphology CLOSE
        grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kernel)
        # threshold the image using Otsu's method
        thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # perform a series of erosions and dilations on the image
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # take the bitwise 'and' between the 'light' regions of the image
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        # another series of erosions and dilations
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)

        # find contours in the thresholded image
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        regions = []
        # loop over the contours
        for c in cnts:
            # grab the bounding box associated with the contour and compute the area and
            # aspect ratio
            (w, h) = cv2.boundingRect(c)[2:]
            aspect_ratio = w / float(h)

            # calculate *extent* for additional filtering
            shape_area = cv2.contourArea(c)
            bbox_area = w * h
            extent = shape_area / float(bbox_area)
            extent = int(extent * 100) / 100

            # compute the rotated bounding box of the region
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.boxPoints(rect))

            # ensure the aspect ratio, width, and height of the bounding box fall within
            # tolerable limits, then update the list of license plate regions
            if (3 < aspect_ratio < 6) and h > self.min_plate_h and w > self.min_plate_w and extent > 0.50:
                regions.append(box)

        # return the list of license plate regions
        return regions



