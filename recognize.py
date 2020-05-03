

# contains all the test runs
import cv2
from utils.display_utils import display
from license.plate import LicensePlateDetector


# initialize a plate detector
detector = LicensePlateDetector(
    image=cv2.imread("sample.jpg")
)

modified = detector.detect()

# show modified
display(modified, flip=False)
