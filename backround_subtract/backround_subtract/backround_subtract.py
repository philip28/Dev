import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os
import glob

GRABCUT_ITER_COUNT = 1

def process_image(in_file_name, output_dir):
    _, file_name = os.path.split(in_file_name)
    img_src = cv2.imread(in_file_name)

    # resize image
    h, w = img_src.shape[:2]
    dim = (int(w/2), int(h/2))
    img_src = cv2.resize(img_src, dsize=dim, interpolation=cv2.INTER_CUBIC)

    #  crop to the field of view
    h, w = img_src.shape[:2]
    left = int(w/2-(w/2)*0.3)
    upper = int(h/2-(h/2)*0.90)
    right = int(w/2+(w/2)*0.3)
    lower = int(h/2+(h/2)*0.95)
    img = img_src[upper : lower, left : right]
    #out_file_name = os.path.join("C:\\train\\objects\\cocacola_pet_classic_0.5\\t", file_name)
    #cv2.imwrite(out_file_name, img)

    # coarse edge detection
    img_blur = cv2.GaussianBlur(img, (51,51), 0)
    canny = cv2.Canny(img_blur, 50, 150, apertureSize=5)
    #img_plot = plt.imshow(canny)
    #plt.show()

    # determine object boundaries
    rrow = cv2.reduce(canny, dim=0, rtype=cv2.REDUCE_MAX)
    rcol = cv2.reduce(canny, dim=1, rtype=cv2.REDUCE_MAX)
    row_nonz = np.nonzero(rrow)
    col_nonz = np.nonzero(rcol)
    left = row_nonz[1][0]
    if left > 5: left -= 5
    right = row_nonz[1][-1]
    if right < canny.shape[1]-7: right += 5
    upper = col_nonz[0][0]
    if upper > 5: upper -= 5
    lower = col_nonz[0][-1] + 5
    if lower < canny.shape[0]-7: lower += 5

    #cv2.rectangle(img, (left, upper), (right, lower), (255,255,0), 2)
    #img_plot = plt.imshow(img)
    #plt.show()

    # apply grabcut
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    rect = (left, upper, right-left, lower-upper)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount=GRABCUT_ITER_COUNT, mode=cv2.GC_INIT_WITH_RECT)

    # BW mask
    mask = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')

    # crop the image
    img = img[upper : lower, left : right]
    mask = mask[upper : lower, left : right]

    # remove small isolated areas
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # apply mask on the source image
    img = img * mask[:,:,np.newaxis]

    out_file_name, _ = os.path.splitext(file_name)
    out_file_name += ".png"
    out_file_name = os.path.join(output_dir, out_file_name)
    cv2.imwrite(out_file_name, img)

    #cv2.imshow('frame', img)
    #cv2.waitKey(0)
    #img_plot = plt.imshow(img)
    #plt.show()


def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    file_mask = os.path.join(input_dir, '*')
    file_list = glob.glob(file_mask)

    for file_name in file_list:
        print("processing ", file_name)
        process_image(file_name, output_dir)

if __name__ == "__main__":
  main()
