import numpy
import cv2


def simple_descriptor(iamge: numpy.ndarray, keypoints: list[cv2.KeyPoint],
        sample_size: int) -> numpy.ndarray:

    iamge = cv2.cvtColor(iamge, cv2.COLOR_BGR2GRAY).astype(numpy.float32)

    pad_width = (sample_size//2, sample_size-1-sample_size//2)
    iamge = numpy.pad(iamge, (pad_width,)*2, 'reflect')

    descriptors = numpy.empty((len(keypoints), sample_size**2), dtype=iamge.dtype)

    for index, keypoint in enumerate(keypoints):
        x = int(keypoint.pt[0])
        y = int(keypoint.pt[1])
        descriptors[index] = iamge[y:y+sample_size, x:x+sample_size].reshape(-1)

    return descriptors


def mops_descriptor(image: numpy.ndarray, keypoints: list[cv2.KeyPoint],
        sample_size: int, space_size: int) -> numpy.ndarray:

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(numpy.float32)

    # times 1.5 for rotate k*45 degrees
    pad_width = int(sample_size/2 * space_size*1.5)
    start = pad_width - sample_size//2
    stop = start + sample_size
    image = numpy.pad(image, ((pad_width,)*2,)*2, mode="reflect")

    descriptors = numpy.empty((len(keypoints), sample_size**2), dtype=image.dtype)

    for index, keypoint in enumerate(keypoints):
        x = int(keypoint.pt[0])
        y = int(keypoint.pt[1])

        img = image[y:y+pad_width*2, x:x+pad_width*2]

        # img = cv2.GaussianBlur(img, ksize, sigma)

        img = cv2.warpAffine(img, cv2.getRotationMatrix2D(
            (pad_width,)*2, keypoint.angle, 1/space_size), None)

        descriptors[index] = img[start:stop, start:stop].reshape(-1)
        descriptors[index] -= numpy.mean(descriptors[index])
        descriptors[index] /= numpy.std(descriptors[index])

    return descriptors
