import numpy as np

num_of_data = 200
num_of_class = 5
# class 갯수,  class별 데이터 갯수, embbeding 차원

# make randomdata

data = list()
for i in range(num_of_class):
    # dd = np.random.normal( size=( num_of_data, 64))
    data.append(np.random.normal((i+2)**2, i*0.73, size=( num_of_data, 64)))

data = np.array(data)
data_reshpaed = data.reshape(( -1, 64))


# make_label
labels = []
for i in range(num_of_class):
    for j in range(num_of_data):
        labels.append(i)

from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

n_components = 2

model = TSNE(n_components=n_components)
x_2d = model.fit_transform(data_reshpaed)

from matplotlib import pyplot as plt
plt.figure(figsize=(10, 10))
colors = ['r', 'g', 'b', 'c', 'm']
# for i, c, label in zip(labels, colors, labels):
#     plt.scatter(x_2d[label == i, 0], x_2d[label == i, 1], c=c, label=label)

xs = x_2d[:,0]
ys = x_2d[:,1]

color_label = list()


for i in labels:
    color_label.append(colors[i])

plt.scatter(xs, ys, c=color_label)
plt.show()

# https://ichi.pro/ko/python-yejeleul-sayonghan-t-sne-sogae-99205845682385
# https://log-laboratory.tistory.com/340
# https://gaussian37.github.io/ml-concept-t_sne/
# https://rfriend.tistory.com/284
# https://skyeong.net/284
