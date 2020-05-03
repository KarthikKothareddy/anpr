import numpy as np
import cv2
import imutils


class LicensePlateDetector:

    def __init__(self, image, min_plate_w=60, min_plate_h=20, mode="bgr"):
        # license plate region
        self.image = image
        self.mode = mode
        self._to_gray = cv2.COLOR_BGR2GRAY if self.mode == "bgr" else cv2.COLOR_RGB2GRAY
        self.min_plate_w = min_plate_w
        self.min_plate_h = min_plate_h

    def detect(self):
        # wrapper around detect_plates
        return self.detect_plates()

    def detect_plates(self):

        # init rectangle kernel
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        # init square kernel
        square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        regions = []
        # to grayscale
        gray = cv2.cvtColor(self.image, self._to_gray)
        # apply the blackhat operation
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)

        return blackhat



