from sklearn.datasets import fetch_openml
import numpy as np
mnist = fetch_openml('mnist_784', as_frame=False)
X = mnist.data
def move_image(dir, img):
    match dir:
        case "up":
            # row1 = img[:28]
            img = img[28:]
            img = np.concatenate((img, np.zeros(28)))
        case "down":
            # row28 = img[756:]
            img = img[:756]
            img = np.concatenate((np.zeros(28), img))
        case "right":
            for i in range(28):
                # last_pxl = img[i*28+27]
                img = np.delete(img,i*28+27)
                img = np.insert(img,i*28, 0)
        case "left":
            for i in range(28):
                # first_pcl = img[i*28]
                img = np.delete(img,i*28)
                img = np.insert(img,i*28+27, 0)
    return img