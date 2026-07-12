import numpy
import cv2
from typing import Literal


def difference_matcher(descriptors1: numpy.ndarray, descriptors2: numpy.ndarray,
        method: Literal['difference', 'ratio'] = 'difference',
        order: int = 2) -> list[cv2.DMatch]:

    # exist a bug that a keypoint may be matched with other keypoints for several times

    # descriptors1.shape = (m, 1, s**2)
    # descriptors2.shape = (1, n, s**2)
    descriptors1 = numpy.expand_dims(descriptors1, axis=1)
    descriptors2 = numpy.expand_dims(descriptors2, axis=0)

    # difference.shape = (m, n)
    difference: numpy.ndarray = numpy.linalg.norm(
        descriptors1 - descriptors2, ord=order, axis=-1)

    if method == 'difference':
        indices = numpy.argmin(difference, axis=-1)
        matches = [cv2.DMatch(index1, indices[index1],
                difference[index1, indices[index1]])
            for index1 in range(difference.shape[0])]

    elif method == 'ratio':
        indices = numpy.argpartition(difference, range(2), axis=-1)[..., :2]
        matches = [cv2.DMatch(index1, indices[index1, 0],
                difference[index1, indices[index1, 0]] \
                    / difference[index1, indices[index1, 1]])
            for index1 in range(difference.shape[0])]
    else:
        raise ValueError

    matches.sort(key=lambda match: match.distance)

    return matches


def opencv_bf_matcher(query_descriptors: numpy.ndarray,
    train_descriptors: numpy.ndarray,
    norm_type: int = cv2.NORM_L2, cross_check: bool = False):
    
    return sorted(cv2.BFMatcher.create(normType=norm_type, crossCheck=cross_check
            ).match(query_descriptors, train_descriptors),
        key=lambda match: match.distance)
