'''
Needle Haystack Algorithm implementation.
'''
from cv2 import matchTemplate
from cv2 import Mat, TM_SQDIFF_NORMED
from numpy import where

class NeedleHaystackAlgorithm:
    '''
    Class containing the Needle Haystack Algorithm implementation.
    '''

    def __init__(
        self,
        needle_image: Mat,
        match_threshold: float = 0.20,
        algorithm: int = TM_SQDIFF_NORMED
    ) -> None:
        '''
        Initialize the Needle Haystack Algorithm.
        '''

        self.needle_image: Mat = needle_image
        self.match_threshold: float = match_threshold
        self.algorithm: int = algorithm

    def get_needle_in_haystack(self, haystack_image: Mat) -> list[tuple[int, int]]:
        '''
        Get the needle in the haystack.
        '''

        # Perform Match Template to get potential match location list
        result: Mat = matchTemplate(
            haystack_image,
            self.needle_image,
            self.algorithm
        )
        locations: tuple = where(result <= self.match_threshold)
        location_list: list[tuple[int, int]] = list(zip(*locations[::-1]))
        return location_list

    def is_needle_in_haystack(self, haystack_img: Mat) -> bool:
        '''
        Check if needle is in haystack.
        '''

        location_list: list[tuple] = self.get_needle_in_haystack(haystack_img)
        return location_list and len(location_list[0]) > 0
