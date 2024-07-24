import os
import numpy
import cv2
import matplotlib.pyplot as plot
from typing import Literal

from detectors import harris_detector
from descriptors import simple_descriptor, mops_descriptor
from matchers import difference_matcher, opencv_bf_matcher
from evaluation import evaluate


################################################################################
# Parameters

# Image
dir_path = r"Project-2\resources"
image_1_path = "Yosemite-1.jpg"
image_2_path = "Yosemite-2.jpg"
transform_matrix_path = "Yosemite-1-to-2.txt"

# Detector
# Harris Detector
harris_detector_kwargs = dict(
    block_size = 2,
    aperture_size = 3,
    k = 0.04,
    threshold = 0.01,
    disjoint_diameter = 10,
)

# Descriptor
descriptor: Literal['Simple', 'MOPS']
descriptor = 'Simple'
# descriptor = 'MOPS'

# Simple Descriptor
simple_descriptor_kwargs = dict(
    sample_size = 16,
)

# MOPS Descriptor
mops_descriptor_kwargs = dict(
    sample_size = 8,
    space_size = 5,
)

# Matcher
matcher: Literal['Difference', 'BF']
matcher = 'Difference'
# matcher = 'BF'

# Difference Matcher
difference_matcher_kwargs = dict(
    # method = 'difference',
    method = 'ratio',
)

# OpenCV Brute-Force Matcher
bf_matcher_kwargs = dict(
    cross_check = True
)

# Display
display_detection: Literal[True, False] = False
display_matches: Literal[True, False] = True


################################################################################
# Main

def dict_str(dictionary: dict, format_string: str = '{}: {}', separator: str = ','):
    return separator.join(format_string.format(k, v) for k, v in dictionary.items())

image1 = cv2.imread(os.path.join(dir_path, image_1_path))
image2 = cv2.imread(os.path.join(dir_path, image_2_path))
transform_matrix = numpy.loadtxt(os.path.join(dir_path, transform_matrix_path))

# Detect
keypoints1 = harris_detector(image1, **harris_detector_kwargs)
keypoints2 = harris_detector(image2, **harris_detector_kwargs)
detector_text = "Descriptor: Harris\n" \
    + dict_str(harris_detector_kwargs, ' {}: {}', '\n')

# Describe
if descriptor == 'Simple':
    descriptor1 = simple_descriptor(image1, keypoints1, **simple_descriptor_kwargs)
    descriptor2 = simple_descriptor(image2, keypoints2, **simple_descriptor_kwargs)
    descriptor_text = "Descriptor: Simple\n" \
        + dict_str(simple_descriptor_kwargs, ' {}: {}', '\n')
elif descriptor == 'MOPS':
    descriptor1 = mops_descriptor(image1, keypoints1, **mops_descriptor_kwargs)
    descriptor2 = mops_descriptor(image2, keypoints2, **mops_descriptor_kwargs)
    descriptor_text = "Descriptor: MOPS\n" \
        + dict_str(mops_descriptor_kwargs, ' {}: {}', '\n')
else:
    raise ValueError

# Match
if matcher == 'Difference':
    matches = difference_matcher(descriptor1, descriptor2, **difference_matcher_kwargs)
    matcher_text = "Matcher: Difference\n" \
        + dict_str(difference_matcher_kwargs, ' {}: {}', '\n')
elif matcher == 'BF':
    matches = opencv_bf_matcher(descriptor1, descriptor2, **bf_matcher_kwargs)
    matcher_text = f"Matcher:\n OpenCV Brute-Force"
else:
    raise ValueError

# Evaluate
fp, tp, auc = evaluate(keypoints1, keypoints2, matches, transform_matrix)
evaluation_text = '\n'.join((f"Total: {len(matches)}",
    f" Correct: {tp[-1]}", f" Incorrect: {fp[-1]}", f"AUC: {auc:.2%}"))

# Display Detection
if display_detection:
    img1 = cv2.drawKeypoints(image1, keypoints1, None, (255, 255, 0),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2 = cv2.drawKeypoints(image2, keypoints2, None, (255, 255, 0),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow(f"Detection", numpy.concatenate((img1, img2), axis=1))

# Display Matches
if display_matches:
    img = cv2.drawMatches(image1, keypoints1, image2, keypoints2,
        matches[:len(matches)*tp[-1]//(tp[-1]+fp[-1])],
        None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow(f"{matcher_text} | {evaluation_text}", img)

# Display ROC
plot.plot(fp/fp[-1], tp/tp[-1])
plot.axline((0, 0), (1, 1), linewidth=0.5, color='g')
plot.grid(True)
plot.xticks(numpy.linspace(0.0, 1.0, 11))
plot.yticks(numpy.linspace(0.0, 1.0, 11))
plot.title("Receiver Operating Characteristic (ROC)")
plot.xlabel("1-Specificity / FPR")
plot.ylabel("Sensitivity / TPR")
plot.text(0.65, 0.04, '\n'.join((detector_text, descriptor_text, matcher_text)),
    backgroundcolor='w')
plot.text(0.35, 0.09, evaluation_text, backgroundcolor='w')
plot.tight_layout()
plot.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
