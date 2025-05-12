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

w = np.zeros(( 64 * 64 * 3))
b = 0

for e in range(100):
    z = x @ w + b
    yest = sigmoid(z)
    loss = lossfunc(y, yest)
    print(loss)
    learnrate = 0.01
    grad = yest - y
    w -= learnrate * grad @ x / n
    b -= learnrate * np.mean(grad)

m = 1000
x = np.array([ procimage(loadimage(k, i)) for k in ('Dog', 'Cat') for i in range(12000 + 1, 12000 + m // 2 + 1) ])
y = np.array([ k for k in (1,0) for i in range(m // 2) ])
yest = np.round(sigmoid(x @ w + b))
np.mean(y == yest)
