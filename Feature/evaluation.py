import numpy
import cv2


def evaluate(keypoints1: list[cv2.KeyPoint], keypoints2: list[cv2.KeyPoint],
        matches: list[cv2.DMatch], transform: numpy.ndarray,
        tolerance: float = 1e-2):

    true = numpy.empty(len(matches), dtype=numpy.bool_)
    # score = numpy.empty(len(matches), dtype=numpy.float32)

    for index, match in enumerate(matches):
        point = transform @ numpy.concatenate((
            keypoints1[match.queryIdx].pt, (1,)))
        true[index] = numpy.allclose(keypoints2[match.trainIdx].pt,
            point[:2]/point[-1], tolerance)
        # score[index] = match.distance
    # score = numpy.subtract(numpy.amax(score), score, out=score)

    # True Positive
    tps = numpy.cumsum(true, dtype=numpy.int_)
    # False Positive
    fps = range(1, len(matches)+1) - tps
    # sklearn.metrics.roc_curve

    auc = numpy.sum(numpy.diff(fps)*(tps[:-1]+tps[1:]))/(tps[-1]*fps[-1]*2)
    # numpy.trapz
    # sklearn.metrics.roc_auc_score, sklearn.metrics.auc

    return (fps, tps, auc)
