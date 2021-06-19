import math
import numpy as np
from skimage import color, transform


def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


def get_new_coords(x, y, alpha, img_shape):
    center_x, center_y = img_shape[1] // 2, img_shape[0] // 2
    x, y = x - center_x, y - center_y
    new_x = center_x + x * math.cos(alpha) + y * math.sin(alpha)
    new_y = center_y + y * math.cos(alpha) - x * math.sin(alpha)
    return new_x, new_y


def transform_face(image, image_eyes):
    image_center = tuple(np.array(image.shape[:2]) / 2)
    x1, y1 = image_eyes[0]
    x2, y2 = image_eyes[1]
    vector = (x2 - x1, y2 - y1)

    alpha = math.copysign(angle(vector, (1, 0)), vector[1])

    rotated = transform.rotate(image, alpha * 180 / np.pi)

    new_x1, new_y1 = get_new_coords(x1, y1, alpha, image.shape)
    new_x2, new_y2 = get_new_coords(x2, y2, alpha, image.shape)

    eye_distance = new_x2 - new_x1
    y_lower, y_upper = int(new_y1 - 2 * eye_distance), int(new_y1 + 2 * eye_distance)
    x_lower, x_upper = int(new_x1 - eye_distance), int(new_x2 + eye_distance)

    y_lower, x_lower = max(0, y_lower), max(0, x_lower)

    crop_img = transform.resize(rotated[y_lower:y_upper, x_lower:x_upper], [224,224,3])

    return crop_img
