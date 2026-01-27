import os
import random
import numpy as np
import tifffile

imgs_dir = "/home/hengweizhao/code/PU-Classification/toolbox/mean_std_cal/ABCD"

C0 = np.array([])
C1 = np.array([])
C2 = np.array([])
C3 = np.array([])
C4 = np.array([])
C5 = np.array([])

count = 0


def static(dir):
    global C0, C1, C2, C3, C4, C5, count
    for root, dirs, files in os.walk(dir, topdown=False):
        random.shuffle(files)
        for name in files[:4000]:
            if not name.lower().endswith('.tif'):
                continue 

            count += 1

            path = os.path.join(root, name)
            image = tifffile.imread(path)
            c0 = np.reshape(image[:, :, 0], (-1, 128 * 128))[0]
            c1 = np.reshape(image[:, :, 1], (-1, 128 * 128))[0]
            c2 = np.reshape(image[:, :, 2], (-1, 128 * 128))[0]
            c3 = np.reshape(image[:, :, 3], (-1, 128 * 128))[0]
            c4 = np.reshape(image[:, :, 4], (-1, 128 * 128))[0]
            c5 = np.reshape(image[:, :, 5], (-1, 128 * 128))[0]
            C0 = np.concatenate((C0, c0), axis=0)
            C1 = np.concatenate((C1, c1), axis=0)
            C2 = np.concatenate((C2, c2), axis=0)
            C3 = np.concatenate((C3, c3), axis=0)
            C4 = np.concatenate((C4, c4), axis=0)
            C5 = np.concatenate((C5, c5), axis=0)

            print(count)


static(imgs_dir)
print("***********************Final Results***********************")
print("0")
print(np.mean(C0))
print(np.std(C0))
print("1")
print(np.mean(C1))
print(np.std(C1))
print("2")
print(np.mean(C2))
print(np.std(C2))
print("3")
print(np.mean(C3))
print(np.std(C3))
print("4")
print(np.mean(C4))
print(np.std(C4))
print("5")
print(np.mean(C5))
print(np.std(C5))