import numpy as np
from PIL import Image as im

getpixels = lambda pic: np.array(pic.getdata()) \
            .reshape(pic.size[0], pic.size[1], 3)
pixelvec = lambda pixels: pixels.reshape(-1)
imagespath = 'dat/PetImages'
loadimage = lambda kind, num: im.open(f'{imagespath}/{kind}/{num}.jpg')
procimage = lambda pic: pixelvec(getpixels(pic.convert('RGB').resize((64, 64)))) / 255

sigmoid = lambda x: 1 / (1 + np.exp(-x))
dsigmoid = lambda x: sigmoid(x) * (1 - sigmoid(x))
relu = lambda x: np.maximum(0, x)
drelu = lambda x: (x > 0).astype(float)
slog = lambda x: np.log(x + 1e-8)
lossfunc = lambda y, yest: np.mean(- y * slog(yest) - (1 - y) * slog(1 - yest))

n = 2000
data = np.array([ procimage(loadimage(k, i)) for k in ('Dog', 'Cat') for i in range(1, n // 2 + 1) ])
res = np.array([ k for k in (1,0) for i in range(n // 2)  ])
order = np.random.permutation(n)
x = data[order]
y = res[order]

w1 = 0.01 + np.zeros((3, 64 * 64 * 3))
b1 = 0.01 + np.zeros(3)
w2 = 0.01 + np.zeros(3)
b2 = 0.01 + 0

for e in range(100):
    z1 = np.apply_along_axis(lambda _: x @ _, 1, w1)
    z1 = np.apply_along_axis(lambda _: _ + b1, 0, z1)
    x2 = relu(z1)
    z2 = w2 @ x2 + b2
    yest = sigmoid(z2)
    loss = lossfunc(y, yest)
    print(loss)
    learnrate = 0.01
    grad = yest - y

    dz2 = yest - y
    dw2 = np.dot(x2, dz2) / n
    db2 = dz2.mean()

    dx2 = np.outer(dz2, w2)
    dz1 = dx2 * drelu(z1).T
    dw1 = (dz1.T @ x) / n
    db1 = dz1.mean(axis=0)

    w1 -= learnrate * dw1
    b1 -= learnrate * db1
    w2 -= learnrate * dw2
    b2 -= learnrate * db2

m = 1000
x = np.array([ procimage(loadimage(k, i)) for k in ('Dog', 'Cat') for i in range(12000 + 1, 12000 + m // 2 + 1) ])
y = np.array([ k for k in (1,0) for i in range(m // 2) ])

z1 = np.apply_along_axis(lambda _: x @ _, 1, w1)
z1 = np.apply_along_axis(lambda _: _ + b1, 0, z1)
yest = np.round(sigmoid(w2 @ relu(x @ w1 + b1) + b2))
np.mean(y == yest)
