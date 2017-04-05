import cv2
import numpy as np

def add_mosaic_rect(image, p1, p2, block_size=10, in_place=True):
    if in_place:
        img = image
    else:
        img = np.copy(image)
    if p1[0] >= p2[0] or p1[1] >= p2[1]:
        return img
    for x in xrange(p1[0], p2[0], block_size):
        for y in xrange(p1[1], p2[1], block_size):
            x2 = min(x+block_size, p2[0])
            y2 = min(y+block_size, p2[1])
            if x2>x and y2>y:
                img[y:y2, x:x2, 0] = np.mean(img[y:y2, x:x2, 0])
                img[y:y2, x:x2, 1] = np.mean(img[y:y2, x:x2, 1])
                img[y:y2, x:x2, 2] = np.mean(img[y:y2, x:x2, 2])
    return img

def add_mosaic_mask(image, mask, block_size=10, in_place=True):
    if in_place:
        img = image
    else:
        img = np.copy(image)
    maskrange = np.where(mask > 0)
    if maskrange[0].size == 0:
        return img
    p1 = [np.min(maskrange[1]), np.min(maskrange[0])]
    p2 = [np.max(maskrange[1]), np.max(maskrange[0])]
    if p1[0] >= p2[0] or p1[1] >= p2[1]:
        return img
    for x in xrange(p1[0], p2[0], block_size):
        for y in xrange(p1[1], p2[1], block_size):
            x2 = min(x+block_size, p2[0])
            y2 = min(y+block_size, p2[1])
            if x2>x and y2>y and np.sum(mask[y:y2, x:x2] > 0) * 2 > (x2-x)*(y2-y):
                img[y:y2, x:x2, 0] = np.mean(img[y:y2, x:x2, 0])
                img[y:y2, x:x2, 1] = np.mean(img[y:y2, x:x2, 1])
                img[y:y2, x:x2, 2] = np.mean(img[y:y2, x:x2, 2])
    return img

def auto_canny(image, sigma=0.33):
	v = np.median(image)
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	return edged