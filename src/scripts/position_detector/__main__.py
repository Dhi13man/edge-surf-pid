"""
Script to detect the position of a needle image in a haystack image.
"""

import cv2
from mss import mss
from mss.screenshot import ScreenShot
from numpy import ndarray, array

from needle_haystack_algorithm import NeedleHaystackAlgorithm

NEEDLE_FILE_PATH: str = "assets/images/needles/image.png"
MATCH_THRESHOLD: float = 0.025

SHOULD_SHOW_LIVE_FEED: bool = True
LIVE_FEED_WINDOW_NAME: str = "Live Feed" if SHOULD_SHOW_LIVE_FEED else None

DEFAULT_FEED_RESOLUTION: tuple[int, int] = (640, 480)


def show_frame(
    frame: cv2.Mat,
    frame_window_name: str = None,
    window_size: tuple[int, int] = DEFAULT_FEED_RESOLUTION,
) -> None:
    """
    Show the live feed in a window.
    """

    if not frame_window_name:
        return

    resized_frame: cv2.Mat = cv2.resize(frame, window_size)
    cv2.imshow(frame_window_name, resized_frame)


def mark_rectangles(
    width: float,
    height: float,
    location_list: list[tuple[int, int]],
    image: cv2.Mat,
    match_line_color: tuple[int, int, int] = (0, 0, 255),
    match_line_type: int = cv2.LINE_4,
) -> cv2.Mat:
    """
    Draw a circle around the needle in the haystack.
    """

    marked_image_cpy: cv2.Mat = image.copy()
    for loc in location_list:
        # Determine the box positions
        top_left: tuple[int, int] = loc
        bottom_right: tuple[int, int] = (top_left[0] + width, top_left[1] + height)

        # Draw the box
        marked_image_cpy: cv2.Mat = cv2.rectangle(
            marked_image_cpy, top_left, bottom_right, match_line_color, match_line_type
        )
    return marked_image_cpy

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
        print(positions)

        if SHOULD_SHOW_LIVE_FEED:
            if len(positions) > 0:
                haystack_image = mark_rectangles(
                    needle_w, needle_h, positions, haystack_image
                )
            show_frame(haystack_image, LIVE_FEED_WINDOW_NAME)

        # Stop recording when we press 'q'
        if cv2.waitKey(5) == ord("q"):
            break

    # Destroy all windows
    cv2.destroyAllWindows()
