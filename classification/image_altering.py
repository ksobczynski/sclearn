from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', as_frame=False)

print(mnist.data[0])

def move_image():
    return