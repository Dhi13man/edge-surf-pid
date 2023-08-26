"""
Script to detect the position of a needle image in a haystack image.
"""

import cv2
from mss import mss
from mss.screenshot import ScreenShot
from numpy import ndarray

from src.needle_haystack_algorithm import NeedleHaystackAlgorithm
from utilities.opencv_utils import show_frame, mark_rectangles, array

NEEDLE_FILE_PATH: str = "./assets/images/needles/image.png"
MATCH_THRESHOLD: float = 0.025

SHOULD_SHOW_LIVE_FEED: bool = True
LIVE_FEED_WINDOW_NAME: str = "Live Feed" if SHOULD_SHOW_LIVE_FEED else None

if __name__ == "__main__":
    # Set up the screen capture
    sct = mss()
    default_monitor = sct.monitors[1]

    # Load the needle
    needle_image: cv2.Mat = cv2.imread(NEEDLE_FILE_PATH, cv2.IMREAD_UNCHANGED)
    needle_image = cv2.cvtColor(needle_image, cv2.COLOR_RGB2BGR)

    # Initialize the Needle Haystack Algorithm
    algorithm: NeedleHaystackAlgorithm = NeedleHaystackAlgorithm(
        needle_image, MATCH_THRESHOLD
    )

    # Get the needle width and height
    needle_w: float = needle_image.shape[1]
    needle_h: float = needle_image.shape[0]

    while True:
        # Take a screenshot
        img: ScreenShot = sct.grab(default_monitor)

        # Convert the screenshot to a numpy array and then to a cv2 image
        haystack_image: ndarray = cv2.cvtColor(array(img), cv2.COLOR_RGB2BGR)

        # Check if needle is in haystack
        positions: list[tuple[int, int]] = algorithm.get_needle_in_haystack(
            haystack_image
        )

        if SHOULD_SHOW_LIVE_FEED:
            if len(positions) > 0:
                marked_image = mark_rectangles(
                    needle_w, needle_h, positions, haystack_image
                )
            show_frame(haystack_image, LIVE_FEED_WINDOW_NAME)

        # Stop recording when we press 'q'
        if cv2.waitKey(5) == ord("q"):
            break

    # Destroy all windows
    cv2.destroyAllWindows()
