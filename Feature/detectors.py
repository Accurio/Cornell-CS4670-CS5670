import numpy
import cv2


def harris_detector(image: numpy.ndarray, block_size: int, aperture_size: int,
        k: float, threshold: float, disjoint_diameter: int) -> list[cv2.KeyPoint]:

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # OpenCV/imgproc/corner.cpp/cornerEigenValsVecs()
    scale = 1.0 / ((1 << (aperture_size - 1)) * block_size * 255)
    Dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, None, aperture_size, scale)
    Dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, None, aperture_size, scale)

    DxDx = Dx**2
    DxDy = Dx*Dy
    DyDy = Dy**2

    # OpenCV/imgproc/corner.cpp/cornerEigenValsVecs()
    cv2.boxFilter(DxDx, cv2.CV_32F, (block_size,)*2, DxDx, normalize=False)
    cv2.boxFilter(DxDy, cv2.CV_32F, (block_size,)*2, DxDy, normalize=False)
    cv2.boxFilter(DyDy, cv2.CV_32F, (block_size,)*2, DyDy, normalize=False)
    # cv2.GaussianBlur(DxDx, (block_size,)*2, sigma, DxDx)
    # cv2.GaussianBlur(DxDy, (block_size,)*2, sigma, DxDy)
    # cv2.GaussianBlur(DyDy, (block_size,)*2, sigma, DyDy)

    response = DxDx * DyDy - DxDy ** 2 - k * (DxDx + DyDy) ** 2

    threshold = response > numpy.amax(response) * threshold

    # non-maximum suppression
    maximum = (response == cv2.dilate(response, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (disjoint_diameter,)*2))) & threshold
    indices = numpy.argwhere(maximum)

    # apply a blur filter for keypoint angle calculation. sigma?
    cv2.GaussianBlur(Dx, (aperture_size,)*2, 4/aperture_size, Dx)
    cv2.GaussianBlur(Dy, (aperture_size,)*2, 4/aperture_size, Dy)

    # CW in OpenCV.KeyPoint.angle, CCW in numpy.arctan2()
    keypoints = [cv2.KeyPoint(numpy.float32(x), numpy.float32(y), disjoint_diameter,
            numpy.degrees(-numpy.arctan2(Dy[y, x], Dx[y, x])), response[y, x])
        for y, x in indices]

    keypoints.sort(key=lambda keypoint: keypoint.response, reverse=True)

    return keypoints
