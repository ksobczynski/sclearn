from sklearn.datasets import fetch_openml
import numpy as np
mnist = fetch_openml('mnist_784', as_frame=False)
X = mnist.data
def move_image(dir, img):
    match dir:
        case "up":
            row1 = img[:28]
            img = img[28:]
            img = np.concatenate((img, row1))
            # img = img + dir
            # for i in range(28):
            #     img[i], img[755+i] = img[755]+i, img[i+i]
        case "down":
            row28 = img[755:]
            img = img[:755]
            img = np.concatenate((row28, img))
            # img = row28 + img
        case "right":
            for i in range(28):
                last_pxl = img[i*28+27]
                img.pop(i*28+27)
                img.insert(i*28)
        case "left":
            for i in range(28):
                first_pcl = img[i*28]
                img.pop(i*28)
                img.insert(i*28+27)
    return img
print(X[0][84+28+28:84+28+28+28])
W = move_image("up", X[0])
print(W[84+28:84+28+28])