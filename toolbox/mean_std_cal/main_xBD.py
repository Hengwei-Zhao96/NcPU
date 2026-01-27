import os
import random
import numpy as np
from PIL import Image

imgs_dir = "/home/hengweizhao/code/PU-Classification/data/xBD/xbd-all"

C0 = np.array([])
C1 = np.array([])
C2 = np.array([])

count = 0


def static(dir):
    global C0, C1, C2, count
    for root, dirs, files in os.walk(dir, topdown=False):
        random.shuffle(files)
        for name in files[:2000]:
            if not name.lower().endswith('.png'):
                continue 

            count += 1

            path = os.path.join(root, name)
            image = np.array(Image.open(path))
            c0 = np.reshape(image[:, :, 0], (-1, 64 * 64))[0]
            c1 = np.reshape(image[:, :, 1], (-1, 64 * 64))[0]
            c2 = np.reshape(image[:, :, 2], (-1, 64 * 64))[0]

            C0 = np.concatenate((C0, c0), axis=0)
            C1 = np.concatenate((C1, c1), axis=0)
            C2 = np.concatenate((C2, c2), axis=0)

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