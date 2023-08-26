"""
Trying out opencv and pyinput to make a AI/ML-less bot that can auto-play Edge surf.
"""

from time import sleep

import cv2
from mss import mss
from mss.screenshot import ScreenShot
from numpy import array, ndarray, arange, newaxis, sum as np_sum
from simple_pid import PID
from pynput.keyboard import Controller, Key

from src.utilities.opencv_utils import (
    color_range_mask_out,
    get_bottom_parts,
    show_frame,
)

SHOULD_SHOW_LIVE_FEED: bool = True
LIVE_FEED_WINDOW_NAME: str = "Live Feed" if SHOULD_SHOW_LIVE_FEED else None

# Important positions and rectangles
MAIN_CHAR_POS: tuple[int, int] = (1686, 966)
CHAR_VISION_DELTA: int = 50
RECT_WIDTH: int = 500
RECT_HEIGHT: int = 500
LEFT_BOTTOM_WINDOW_RECT: tuple[int, int, int, int] = (
    MAIN_CHAR_POS[0] - RECT_WIDTH,
    MAIN_CHAR_POS[1] + CHAR_VISION_DELTA,
    RECT_WIDTH,
    RECT_HEIGHT,
)
RIGHT_BOTTOM_WINDOW_RECT: tuple[int, int, int, int] = (
    MAIN_CHAR_POS[0],
    MAIN_CHAR_POS[1] + CHAR_VISION_DELTA,
    RECT_WIDTH,
    RECT_HEIGHT,
)

# Important colors
LIGHT_BLUE: tuple[int, int, int] = (0, 0, 200)
WHITE: tuple[int, int, int] = (255, 255, 255)


def weighted_sum(image: ndarray) -> float:
    """
    Get the weighted sum of the pixels in the image.
    """

    # Define the weight factor
    height, width = image.shape

    # Get the weights
    weights: ndarray = arange(height)[:, newaxis][::-1]

    # Calculate the weighted sum
    return int(np_sum(100 * image * weights / (height * width)))


if __name__ == "__main__":
    # Set up the screen capture
    sct = mss()
    default_monitor = sct.monitors[1]

    # Set up the PID controller
    pid: PID = PID(0.001, 0.001, 0.05, setpoint=1, output_limits=(-100, 100))

    # Set up the keyboard controller
    keyboard: Controller = Controller()

    while True:
        # Take a screenshot
        img: ScreenShot = sct.grab(default_monitor)

        # Convert the screenshot to a numpy array and then to a cv2 image
        haystack_image: ndarray = cv2.cvtColor(array(img), cv2.COLOR_RGB2BGR)

        # Perform color masking and get the bottom parts of the window
        haystack_image_masked: ndarray = color_range_mask_out(
            haystack_image, LIGHT_BLUE, WHITE
        )
        image_left_bottom, image_right_bottom = get_bottom_parts(
            haystack_image_masked,
            LEFT_BOTTOM_WINDOW_RECT,
            RIGHT_BOTTOM_WINDOW_RECT,
        )

        # Sum the pixels in the bottom parts of the window
        left_bottom_weighted_sum: float = weighted_sum(image_left_bottom)
        right_bottom_weighted_sum: float = weighted_sum(image_right_bottom)
        delta: float = left_bottom_weighted_sum - right_bottom_weighted_sum
        delta_pid: float = pid(delta)
        print("Delta:", delta, "Delta PID:", delta_pid)

        # Determine which direction to press
        key_to_press: int = Key.down
        if delta < -200:
            key_to_press = Key.left
        elif delta > 200:
            key_to_press = Key.right

        # Press button
        keyboard.press(key_to_press)
        sleep(abs(delta_pid) / 300)
        keyboard.release(key_to_press)

        # Reset the PID controller if the direction changes
        if (delta_pid < 0 and delta > 0) or (delta_pid > 0 and delta < 0):
            pid.reset()

        if SHOULD_SHOW_LIVE_FEED:
            rejoined_image: ndarray = cv2.hconcat(
                [image_left_bottom, image_right_bottom]
            )
            show_frame(
                rejoined_image, LIVE_FEED_WINDOW_NAME, rejoined_image.shape[:2][::-1]
            )

        # Stop recording when we press 'q'
        if cv2.waitKey(5) == ord("q"):
            break

    # Destroy all windows
    cv2.destroyAllWindows()
