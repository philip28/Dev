import numpy as np
import os.path
import sys
import cv2

def main(argv):
    sum = np.array([0, 0, 0]).astype(np.int64)
    count = 0

    for root, subdirs, files in os.walk(argv[1]):
        print('--\nroot = ' + root)
        for file in files:
            full_path = os.path.join(root, file)
            image = cv2.imread(full_path)
            sum += np.sum(image, axis=(0,1))
            count += image.shape[0] * image.shape[1]
    
    avg = np.divide(sum, count).astype(np.float32)
    print("BGR!")
    print(avg)
    print(sum)
    print(count)

if __name__ == "__main__":
    main(sys.argv)
