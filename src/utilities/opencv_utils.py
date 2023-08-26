"""
This module contains OpenCV utilising functions that are used in the main script.
"""

import cv2
from numpy import array, ndarray

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

    marked_image: cv2.Mat = image.copy()
    for loc in location_list:
        # Determine the box positions
        top_left: tuple[int, int] = loc
        bottom_right: tuple[int, int] = (top_left[0] + width, top_left[1] + height)

        # Draw the box
        marked_image: cv2.Mat = cv2.rectangle(
            marked_image, top_left, bottom_right, match_line_color, match_line_type
        )
    return marked_image


def color_range_mask_out(
    image: ndarray,
    color_low: tuple[int, int, int],
    color_high: tuple[int, int, int],
) -> tuple[ndarray, ndarray]:
    """
    Mask out a specific color range.
    """

    # Only keep pixels that are in the color range
    mask: ndarray = cv2.inRange(image, array(color_low), array(color_high))

    # Invert the mask values to get the pixels we want to keep
    return cv2.bitwise_not(mask)


def get_bottom_parts(
    image: ndarray,
    left_bottom_window_rect: tuple[int, int, int, int],
    right_bottom_window_rect: tuple[int, int, int, int],
) -> tuple[ndarray, ndarray]:
    """
    Get the bottom parts of the window.
    """

    # Get the bottom parts of the window
    masked_left_bottom: ndarray = image[
        left_bottom_window_rect[1] : left_bottom_window_rect[1]
        + left_bottom_window_rect[3],
        left_bottom_window_rect[0] : left_bottom_window_rect[0]
        + left_bottom_window_rect[2],
    ]
    masked_right_bottom: ndarray = image[
        right_bottom_window_rect[1] : right_bottom_window_rect[1]
        + right_bottom_window_rect[3],
        right_bottom_window_rect[0] : right_bottom_window_rect[0]
        + right_bottom_window_rect[2],
    ]
    return (masked_left_bottom, masked_right_bottom)
