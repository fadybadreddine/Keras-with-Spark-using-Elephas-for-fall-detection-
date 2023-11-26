# -*- coding: utf-8 -*-


from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive

pip install elephas

pip install matplotlib

import cv2
from matplotlib import pyplot as plt
import numpy as np

import numpy

imref = cv2.imread("/content/drive/My Drive/Dataset/ref.png", 0)
for a in range(1, 161):
    b = str(a).zfill(3)
    st = f"/content/drive/My Drive/datasetc/fall-01-cam0-rgb/fall-01-cam0-rgb-{b}.png"
    print(st)
    im = cv2.imread(st, 0)

    if im is None:
        print(f"Erreur lors de la lecture de l'image: {st}")
        continue

    imdiff = abs(cv2.subtract(imref, im))
    print(imdiff)
    lbp = local_binary_pattern(imdiff, 8, 1, method='nri_uniform')
    (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(60))
    hist = hist.astype("float")
    hist /= (hist.sum())

    lbp2 = local_binary_pattern(imdiff, 8, 2,method='nri_uniform')
    (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
    hist2 = hist2.astype("float")
    hist2 /= (hist2.sum())

    lbp3 = local_binary_pattern(imdiff, 8, 3,method='nri_uniform')
    (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
    hist3 = hist3.astype("float")
    hist3 /= (hist3.sum())
    if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit=hist

    else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit=numpy.vstack([voit,hist])
import pandas as pd

chemin = "/content/drive/My Drive/Dataset/fall-01-cam0-d/labels.csv"
df = pd.read_csv(chemin, index_col=0, sep=',')

d=df.to_numpy()
d=numpy.subtract(d,1)
t=numpy.hstack((voit,d))

import cv2
import numpy
from skimage.feature import local_binary_pattern
import pandas as pd

imref = cv2.imread("/content/drive/My Drive/Dataset/ref.png", 0)
for a in range(1, 97):
    b = str(a).zfill(3)
    st = f"/content/drive/My Drive/datasetc/fall-02-cam0-rgb/fall-02-cam0-rgb-{b}.png"
    print(st)
    im = cv2.imread(st, 0)

    if im is None:
        print(f"Erreur lors de la lecture de l'image: {st}")
        continue

    imdiff = abs(cv2.subtract(imref, im))
    print(imdiff)
    lbp = local_binary_pattern(imdiff, 8, 1, method='nri_uniform')
    (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(60))
    hist = hist.astype("float")
    hist /= (hist.sum())

    lbp2 = local_binary_pattern(imdiff, 8, 2,method='nri_uniform')
    (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
    hist2 = hist2.astype("float")
    hist2 /= (hist2.sum())

    lbp3 = local_binary_pattern(imdiff, 8, 3,method='nri_uniform')
    (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
    hist3 = hist3.astype("float")
    hist3 /= (hist3.sum())
    if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit=hist

    else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit=numpy.vstack([voit,hist])
import pandas as pd

chemin = "/content/drive/My Drive/Dataset/fall-02-cam-0-d/labels.csv"
df = pd.read_csv(chemin, index_col=0, sep=',')

d=df.to_numpy()
d=numpy.subtract(d,1)
t22=numpy.hstack((voit,d))

imref = cv2.imread("/content/drive/My Drive/Dataset/ref.png", 0)

for a in range(1, 209):
    b = str(a).zfill(3)
    st = f"/content/drive/My Drive/datasetc/fall-03-cam0-rgb/fall-03-cam0-rgb-{b}.png"
    print(st)
    im = cv2.imread(st, 0)

    if im is None:
        print(f"Erreur lors de la lecture de l'image: {st}")
        continue

    imdiff = abs(cv2.subtract(imref, im))
    print(imdiff)
    lbp = local_binary_pattern(imdiff, 8, 1, method='nri_uniform')
    (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(60))
    hist = hist.astype("float")
    hist /= (hist.sum())

    lbp2 = local_binary_pattern(imdiff, 8, 2,method='nri_uniform')
    (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
    hist2 = hist2.astype("float")
    hist2 /= (hist2.sum())

    lbp3 = local_binary_pattern(imdiff, 8, 3,method='nri_uniform')
    (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
    hist3 = hist3.astype("float")
    hist3 /= (hist3.sum())
    if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit5=hist

    else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit5=numpy.vstack([voit5,hist])


chemin = "/content/drive/My Drive/Dataset/fall-03-cam0-d/labels.csv"
df5 = pd.read_csv(chemin, index_col=0, sep=',')

d5=df5.to_numpy()
d5=numpy.subtract(d5,1)
t5=numpy.hstack((voit5,d5))

imref = cv2.imread("/content/drive/My Drive/Dataset/ref.png", 0)

for a in range(1, 97):
    b = str(a).zfill(3)
    st = f"/content/drive/My Drive/datasetc/fall-04-cam0-rgb/fall-04-cam0-rgb-{b}.png"
    print(st)
    im = cv2.imread(st, 0)

    if im is None:
        print(f"Erreur lors de la lecture de l'image: {st}")
        continue

    imdiff = abs(cv2.subtract(imref, im))
    print(imdiff)
    lbp = local_binary_pattern(imdiff, 8, 1, method='nri_uniform')
    (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(60))
    hist = hist.astype("float")
    hist /= (hist.sum())

    lbp2 = local_binary_pattern(imdiff, 8, 2,method='nri_uniform')
    (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
    hist2 = hist2.astype("float")
    hist2 /= (hist2.sum())

    lbp3 = local_binary_pattern(imdiff, 8, 3,method='nri_uniform')
    (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
    hist3 = hist3.astype("float")
    hist3 /= (hist3.sum())
    if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit6=hist

    else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit6=numpy.vstack([voit6,hist])


chemin = "/content/drive/My Drive/Dataset/fall-04-cam0-d/labels.csv"
df6 = pd.read_csv(chemin, index_col=0, sep=',')

d6=df6.to_numpy()
d6=numpy.subtract(d6,1)
t6=numpy.hstack((voit6,d6))

imref = cv2.imread("/content/drive/My Drive/Dataset/ref.png", 0)

for a in range(1, 145):
    b = str(a).zfill(3)
    st = f"/content/drive/My Drive/datasetc/fall-05-cam0-rgb/fall-05-cam0-rgb-{b}.png"
    print(st)
    im = cv2.imread(st, 0)

    if im is None:
        print(f"Erreur lors de la lecture de l'image: {st}")
        continue

    imdiff = abs(cv2.subtract(imref, im))
    print(imdiff)
    lbp = local_binary_pattern(imdiff, 8, 1, method='nri_uniform')
    (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(60))
    hist = hist.astype("float")
    hist /= (hist.sum())

    lbp2 = local_binary_pattern(imdiff, 8, 2,method='nri_uniform')
    (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
    hist2 = hist2.astype("float")
    hist2 /= (hist2.sum())

    lbp3 = local_binary_pattern(imdiff, 8, 3,method='nri_uniform')
    (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
    hist3 = hist3.astype("float")
    hist3 /= (hist3.sum())
    if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit7=hist

    else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit7=numpy.vstack([voit7,hist])


chemin = "/content/drive/My Drive/Dataset/fall-05-cam0-d/labels.csv"
df7 = pd.read_csv(chemin, index_col=0, sep=',')

d7=df7.to_numpy()
d7=numpy.subtract(d7,1)
t7=numpy.hstack((voit7,d7))

imref = cv2.imread("/content/drive/My Drive/Dataset/ref.png", 0)

for a in range(1, 97):
    b = str(a).zfill(3)
    st = f"/content/drive/My Drive/datasetc/fall-06-cam0-rgb/fall-06-cam0-rgb-{b}.png"
    print(st)
    im = cv2.imread(st, 0)

    if im is None:
        print(f"Erreur lors de la lecture de l'image: {st}")
        continue

    imdiff = abs(cv2.subtract(imref, im))
    print(imdiff)
    lbp = local_binary_pattern(imdiff, 8, 1, method='nri_uniform')
    (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(60))
    hist = hist.astype("float")
    hist /= (hist.sum())

    lbp2 = local_binary_pattern(imdiff, 8, 2,method='nri_uniform')
    (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
    hist2 = hist2.astype("float")
    hist2 /= (hist2.sum())

    lbp3 = local_binary_pattern(imdiff, 8, 3,method='nri_uniform')
    (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
    hist3 = hist3.astype("float")
    hist3 /= (hist3.sum())
    if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit8=hist

    else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit8=numpy.vstack([voit8,hist])


chemin = "/content/drive/My Drive/Dataset/fall-06-cam0-d/labels.csv"
df8 = pd.read_csv(chemin, index_col=0, sep=',')

d8=df8.to_numpy()
d8=numpy.subtract(d8,1)
t8=numpy.hstack((voit8,d8))

imref = cv2.imread("/content/drive/My Drive/Dataset/ref.png", 0)

for a in range(1, 145):
    b = str(a).zfill(3)
    st = f"/content/drive/My Drive/datasetc/fall-07-cam0-rgb/fall-07-cam0-rgb-{b}.png"
    print(st)
    im = cv2.imread(st, 0)

    if im is None:
        print(f"Erreur lors de la lecture de l'image: {st}")
        continue

    imdiff = abs(cv2.subtract(imref, im))
    print(imdiff)
    lbp = local_binary_pattern(imdiff, 8, 1, method='nri_uniform')
    (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(60))
    hist = hist.astype("float")
    hist /= (hist.sum())

    lbp2 = local_binary_pattern(imdiff, 8, 2,method='nri_uniform')
    (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
    hist2 = hist2.astype("float")
    hist2 /= (hist2.sum())

    lbp3 = local_binary_pattern(imdiff, 8, 3,method='nri_uniform')
    (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
    hist3 = hist3.astype("float")
    hist3 /= (hist3.sum())
    if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit9=hist

    else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit9=numpy.vstack([voit9,hist])


chemin = "/content/drive/My Drive/Dataset/fall-07-cam0-d/labels.csv"
df9 = pd.read_csv(chemin, index_col=0, sep=',')

d9=df9.to_numpy()
d9=numpy.subtract(d9,1)
t9=numpy.hstack((voit9,d9))

imref = cv2.imread("/content/drive/My Drive/Dataset/ref.png", 0)

for a in range(1, 81):
    b = str(a).zfill(3)
    st = f"/content/drive/My Drive/datasetc/fall-08-cam0-rgb/fall-08-cam0-rgb-{b}.png"
    print(st)
    im = cv2.imread(st, 0)

    if im is None:
        print(f"Erreur lors de la lecture de l'image: {st}")
        continue

    imdiff = abs(cv2.subtract(imref, im))
    print(imdiff)
    lbp = local_binary_pattern(imdiff, 8, 1, method='nri_uniform')
    (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(60))
    hist = hist.astype("float")
    hist /= (hist.sum())

    lbp2 = local_binary_pattern(imdiff, 8, 2,method='nri_uniform')
    (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
    hist2 = hist2.astype("float")
    hist2 /= (hist2.sum())

    lbp3 = local_binary_pattern(imdiff, 8, 3,method='nri_uniform')
    (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
    hist3 = hist3.astype("float")
    hist3 /= (hist3.sum())
    if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit10=hist

    else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit10=numpy.vstack([voit10,hist])


chemin = "/content/drive/My Drive/Dataset/fall-08-cam0-d/labels.csv"
df10 = pd.read_csv(chemin, index_col=0, sep=',')

d10=df10.to_numpy()
d10=numpy.subtract(d10,1)
t10=numpy.hstack((voit10,d10))

imref = cv2.imread("/content/drive/My Drive/Dataset/ref.png", 0)

for a in range(1, 177):
    b = str(a).zfill(3)
    st = f"/content/drive/My Drive/datasetc/fall-09-cam0-rgb/fall-09-cam0-rgb-{b}.png"
    print(st)
    im = cv2.imread(st, 0)

    if im is None:
        print(f"Erreur lors de la lecture de l'image: {st}")
        continue

    imdiff = abs(cv2.subtract(imref, im))
    print(imdiff)
    lbp = local_binary_pattern(imdiff, 8, 1, method='nri_uniform')
    (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(60))
    hist = hist.astype("float")
    hist /= (hist.sum())

    lbp2 = local_binary_pattern(imdiff, 8, 2,method='nri_uniform')
    (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
    hist2 = hist2.astype("float")
    hist2 /= (hist2.sum())

    lbp3 = local_binary_pattern(imdiff, 8, 3,method='nri_uniform')
    (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
    hist3 = hist3.astype("float")
    hist3 /= (hist3.sum())
    if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit11=hist

    else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit11=numpy.vstack([voit11,hist])


chemin = "/content/drive/My Drive/Dataset/fall-09-cam0-d/labels.csv"
df11 = pd.read_csv(chemin, index_col=0, sep=',')

d11=df11.to_numpy()
d11=numpy.subtract(d11,1)
t11=numpy.hstack((voit11,d11))

imref = cv2.imread("/content/drive/My Drive/Dataset/ref.png", 0)
for a in range(1, 129):
    b = str(a).zfill(3)
    st = f"/content/drive/My Drive/datasetc/fall-10-cam0-rgb/fall-10-cam0-rgb-{b}.png"
    print(st)
    im = cv2.imread(st, 0)

    if im is None:
        print(f"Erreur lors de la lecture de l'image: {st}")
        continue

    imdiff = abs(cv2.subtract(imref, im))
    print(imdiff)
    lbp = local_binary_pattern(imdiff, 8, 1, method='nri_uniform')
    (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(60))
    hist = hist.astype("float")
    hist /= (hist.sum())

    lbp2 = local_binary_pattern(imdiff, 8, 2,method='nri_uniform')
    (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
    hist2 = hist2.astype("float")
    hist2 /= (hist2.sum())

    lbp3 = local_binary_pattern(imdiff, 8, 3,method='nri_uniform')
    (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
    hist3 = hist3.astype("float")
    hist3 /= (hist3.sum())
    if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit12=hist

    else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit12=numpy.vstack([voit12,hist])


chemin = "/content/drive/My Drive/Dataset/fall-10-cam0-d/labels.csv"
df12 = pd.read_csv(chemin, index_col=0, sep=',')

d12=df12.to_numpy()
d12=numpy.subtract(d12,1)
t12=numpy.hstack((voit12,d12))

imref = cv2.imread("/content/drive/My Drive/Dataset/ref.png", 0)
for a in range(1, 129):
    b = str(a).zfill(3)
    st = f"/content/drive/My Drive/datasetc/fall-11-cam0-rgb/fall-11-cam0-rgb-{b}.png"
    print(st)
    im = cv2.imread(st, 0)

    if im is None:
        print(f"Erreur lors de la lecture de l'image: {st}")
        continue

    imdiff = abs(cv2.subtract(imref, im))
    print(imdiff)
    lbp = local_binary_pattern(imdiff, 8, 1, method='nri_uniform')
    (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(60))
    hist = hist.astype("float")
    hist /= (hist.sum())

    lbp2 = local_binary_pattern(imdiff, 8, 2,method='nri_uniform')
    (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
    hist2 = hist2.astype("float")
    hist2 /= (hist2.sum())

    lbp3 = local_binary_pattern(imdiff, 8, 3,method='nri_uniform')
    (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
    hist3 = hist3.astype("float")
    hist3 /= (hist3.sum())
    if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit12=hist

    else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit12=numpy.vstack([voit12,hist])


chemin = "/content/drive/My Drive/Dataset/fall-11-cam0-d/labels.csv"
df12 = pd.read_csv(chemin, index_col=0, sep=',')

d12=df12.to_numpy()
d12=numpy.subtract(d12,1)
t13=numpy.hstack((voit12,d12))

imref = cv2.imread("/content/drive/My Drive/Dataset/ref.png", 0)

for a in range(1, 97):
    b = str(a).zfill(3)
    st = f"/content/drive/My Drive/datasetc/fall-12-cam0-rgb/fall-12-cam0-rgb-{b}.png"
    print(st)
    im = cv2.imread(st, 0)

    if im is None:
        print(f"Erreur lors de la lecture de l'image: {st}")
        continue

    imdiff = abs(cv2.subtract(imref, im))
    print(imdiff)
    lbp = local_binary_pattern(imdiff, 8, 1, method='nri_uniform')
    (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(60))
    hist = hist.astype("float")
    hist /= (hist.sum())

    lbp2 = local_binary_pattern(imdiff, 8, 2,method='nri_uniform')
    (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
    hist2 = hist2.astype("float")
    hist2 /= (hist2.sum())

    lbp3 = local_binary_pattern(imdiff, 8, 3,method='nri_uniform')
    (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
    hist3 = hist3.astype("float")
    hist3 /= (hist3.sum())
    if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit14=hist

    else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit14=numpy.vstack([voit14,hist])


chemin = "/content/drive/My Drive/Dataset/fall-12-cam0-d/labels.csv"
df14 = pd.read_csv(chemin, index_col=0, sep=',')

d14=df14.to_numpy()
d14=numpy.subtract(d14,1)
t14=numpy.hstack((voit14,d14))

imref = cv2.imread("/content/drive/My Drive/Dataset/ref.png", 0)

for a in range(1, 81):
    b = str(a).zfill(3)
    st = f"/content/drive/My Drive/datasetc/fall-13-cam0-rgb/fall-13-cam0-rgb-{b}.png"
    print(st)
    im = cv2.imread(st, 0)

    if im is None:
        print(f"Erreur lors de la lecture de l'image: {st}")
        continue

    imdiff = abs(cv2.subtract(imref, im))
    print(imdiff)
    lbp = local_binary_pattern(imdiff, 8, 1, method='nri_uniform')
    (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(60))
    hist = hist.astype("float")
    hist /= (hist.sum())

    lbp2 = local_binary_pattern(imdiff, 8, 2,method='nri_uniform')
    (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
    hist2 = hist2.astype("float")
    hist2 /= (hist2.sum())

    lbp3 = local_binary_pattern(imdiff, 8, 3,method='nri_uniform')
    (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
    hist3 = hist3.astype("float")
    hist3 /= (hist3.sum())
    if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit15=hist

    else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit15=numpy.vstack([voit15,hist])


chemin = "/content/drive/My Drive/Dataset/fall-13-cam0-d/labels.csv"
df15 = pd.read_csv(chemin, index_col=0, sep=',')

d15=df15.to_numpy()
d15=numpy.subtract(d15,1)
t15=numpy.hstack((voit15,d15))

imref = cv2.imread("/content/drive/My Drive/Dataset/ref.png", 0)

for a in range(1, 49):
    b = str(a).zfill(3)
    st = f"/content/drive/My Drive/datasetc/fall-14-cam0-rgb/fall-14-cam0-rgb-{b}.png"
    print(st)
    im = cv2.imread(st, 0)

    if im is None:
        print(f"Erreur lors de la lecture de l'image: {st}")
        continue

    imdiff = abs(cv2.subtract(imref, im))
    print(imdiff)
    lbp = local_binary_pattern(imdiff, 8, 1, method='nri_uniform')
    (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(60))
    hist = hist.astype("float")
    hist /= (hist.sum())

    lbp2 = local_binary_pattern(imdiff, 8, 2,method='nri_uniform')
    (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
    hist2 = hist2.astype("float")
    hist2 /= (hist2.sum())

    lbp3 = local_binary_pattern(imdiff, 8, 3,method='nri_uniform')
    (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
    hist3 = hist3.astype("float")
    hist3 /= (hist3.sum())
    if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit16=hist

    else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit16=numpy.vstack([voit16,hist])


chemin = "/content/drive/My Drive/Dataset/fall-14-cam0-d/labels.csv"
df16 = pd.read_csv(chemin, index_col=0, sep=',')

d16=df16.to_numpy()
d16=numpy.subtract(d16,1)
t16=numpy.hstack((voit16,d16))

imref = cv2.imread("/content/drive/My Drive/Dataset/ref.png", 0)
for a in range(1, 65):
    b = str(a).zfill(3)
    st = f"/content/drive/My Drive/datasetc/fall-15-cam0-rgb/fall-15-cam0-rgb-{b}.png"
    print(st)
    im = cv2.imread(st, 0)

    if im is None:
        print(f"Erreur lors de la lecture de l'image: {st}")
        continue

    imdiff = abs(cv2.subtract(imref, im))
    print(imdiff)
    lbp = local_binary_pattern(imdiff, 8, 1, method='nri_uniform')
    (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(60))
    hist = hist.astype("float")
    hist /= (hist.sum())

    lbp2 = local_binary_pattern(imdiff, 8, 2,method='nri_uniform')
    (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
    hist2 = hist2.astype("float")
    hist2 /= (hist2.sum())

    lbp3 = local_binary_pattern(imdiff, 8, 3,method='nri_uniform')
    (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
    hist3 = hist3.astype("float")
    hist3 /= (hist3.sum())
    if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit17=hist

    else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit17=numpy.vstack([voit17,hist])


chemin = "/content/drive/My Drive/Dataset/fall-15-cam0-d/labels.csv"
df17 = pd.read_csv(chemin, index_col=0, sep=',')

d17=df17.to_numpy()
d17=numpy.subtract(d17,1)
t17=numpy.hstack((voit17,d17))

imref = cv2.imread("/content/drive/My Drive/Dataset/ref.png", 0)
for a in range(1, 49):
    b = str(a).zfill(3)
    st = f"/content/drive/My Drive/datasetc/fall-16-cam0-rgb/fall-16-cam0-rgb-{b}.png"
    print(st)
    im = cv2.imread(st, 0)

    if im is None:
        print(f"Erreur lors de la lecture de l'image: {st}")
        continue

    imdiff = abs(cv2.subtract(imref, im))
    print(imdiff)
    lbp = local_binary_pattern(imdiff, 8, 1, method='nri_uniform')
    (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(60))
    hist = hist.astype("float")
    hist /= (hist.sum())

    lbp2 = local_binary_pattern(imdiff, 8, 2,method='nri_uniform')
    (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
    hist2 = hist2.astype("float")
    hist2 /= (hist2.sum())

    lbp3 = local_binary_pattern(imdiff, 8, 3,method='nri_uniform')
    (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
    hist3 = hist3.astype("float")
    hist3 /= (hist3.sum())
    if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit18=hist

    else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit18=numpy.vstack([voit18,hist])


chemin = "/content/drive/My Drive/Dataset/fall-16-cam0-d/lables.csv"
df18 = pd.read_csv(chemin, index_col=0, sep=',')

d18=df18.to_numpy()
d18=numpy.subtract(d18,1)
t18=numpy.hstack((voit18,d18))

imref = cv2.imread("/content/drive/My Drive/Dataset/ref.png", 0)

for a in range(1, 97):
    b = str(a).zfill(3)
    st = f"/content/drive/My Drive/datasetc/fall-17-cam0-rgb/fall-17-cam0-rgb-{b}.png"
    print(st)
    im = cv2.imread(st, 0)

    if im is None:
        print(f"Erreur lors de la lecture de l'image: {st}")
        continue

    imdiff = abs(cv2.subtract(imref, im))
    print(imdiff)
    lbp = local_binary_pattern(imdiff, 8, 1, method='nri_uniform')
    (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(60))
    hist = hist.astype("float")
    hist /= (hist.sum())

    lbp2 = local_binary_pattern(imdiff, 8, 2,method='nri_uniform')
    (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
    hist2 = hist2.astype("float")
    hist2 /= (hist2.sum())

    lbp3 = local_binary_pattern(imdiff, 8, 3,method='nri_uniform')
    (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
    hist3 = hist3.astype("float")
    hist3 /= (hist3.sum())
    if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit19=hist

    else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit19=numpy.vstack([voit19,hist])


chemin = "/content/drive/My Drive/Dataset/fall-17-cam0-d/labels.csv"
df19 = pd.read_csv(chemin, index_col=0, sep=',')

d19=df19.to_numpy()
d19=numpy.subtract(d19,1)
t19=numpy.hstack((voit19,d19))

imref = cv2.imread("/content/drive/My Drive/Dataset/ref.png", 0)

for a in range(1, 65):
    b = str(a).zfill(3)
    st = f"/content/drive/My Drive/datasetc/fall-18-cam0-rgb/fall-18-cam0-rgb-{b}.png"
    print(st)
    im = cv2.imread(st, 0)

    if im is None:
        print(f"Erreur lors de la lecture de l'image: {st}")
        continue

    imdiff = abs(cv2.subtract(imref, im))
    print(imdiff)
    lbp = local_binary_pattern(imdiff, 8, 1, method='nri_uniform')
    (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(60))
    hist = hist.astype("float")
    hist /= (hist.sum())

    lbp2 = local_binary_pattern(imdiff, 8, 2,method='nri_uniform')
    (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
    hist2 = hist2.astype("float")
    hist2 /= (hist2.sum())

    lbp3 = local_binary_pattern(imdiff, 8, 3,method='nri_uniform')
    (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
    hist3 = hist3.astype("float")
    hist3 /= (hist3.sum())
    if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit20=hist

    else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit20=numpy.vstack([voit20,hist])


chemin = "/content/drive/My Drive/Dataset/fall-18-cam0-d/labes.csv"
df20 = pd.read_csv(chemin, index_col=0, sep=',')

d20=df20.to_numpy()
d20=numpy.subtract(d20,1)
t20=numpy.hstack((voit20,d20))

imref = cv2.imread("/content/drive/My Drive/Dataset/ref.png", 0)

for a in range(1, 97):
    b = str(a).zfill(3)
    st = f"/content/drive/My Drive/datasetc/fall-19-cam0-rgb/fall-19-cam0-rgb-{b}.png"
    print(st)
    im = cv2.imread(st, 0)

    if im is None:
        print(f"Erreur lors de la lecture de l'image: {st}")
        continue

    imdiff = abs(cv2.subtract(imref, im))
    print(imdiff)
    lbp = local_binary_pattern(imdiff, 8, 1, method='nri_uniform')
    (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(60))
    hist = hist.astype("float")
    hist /= (hist.sum())

    lbp2 = local_binary_pattern(imdiff, 8, 2,method='nri_uniform')
    (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
    hist2 = hist2.astype("float")
    hist2 /= (hist2.sum())

    lbp3 = local_binary_pattern(imdiff, 8, 3,method='nri_uniform')
    (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
    hist3 = hist3.astype("float")
    hist3 /= (hist3.sum())
    if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit21=hist

    else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit21=numpy.vstack([voit21,hist])


chemin = "/content/drive/My Drive/Dataset/fall-19-cam0-d/labels.csv"
df21 = pd.read_csv(chemin, index_col=0, sep=',')

d21=df21.to_numpy()
d21=numpy.subtract(d21,1)
t21=numpy.hstack((voit21,d21))

imref = cv2.imread("/content/drive/My Drive/Dataset/ref.png", 0)

for a in range(1, 97):
    b = str(a).zfill(3)
    st = f"/content/drive/My Drive/datasetc/fall-20-cam0-rgb/fall-20-cam0-rgb-{b}.png"
    print(st)
    im = cv2.imread(st, 0)

    if im is None:
        print(f"Erreur lors de la lecture de l'image: {st}")
        continue

    imdiff = abs(cv2.subtract(imref, im))
    print(imdiff)
    lbp = local_binary_pattern(imdiff, 8, 1, method='nri_uniform')
    (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(60))
    hist = hist.astype("float")
    hist /= (hist.sum())

    lbp2 = local_binary_pattern(imdiff, 8, 2,method='nri_uniform')
    (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
    hist2 = hist2.astype("float")
    hist2 /= (hist2.sum())

    lbp3 = local_binary_pattern(imdiff, 8, 3,method='nri_uniform')
    (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
    hist3 = hist3.astype("float")
    hist3 /= (hist3.sum())
    if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit22=hist

    else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit22=numpy.vstack([voit22,hist])


chemin = "/content/drive/My Drive/Dataset/fall-20-cam0-d/labels.csv"
df22 = pd.read_csv(chemin, index_col=0, sep=',')

d22=df22.to_numpy()
d22=numpy.subtract(d22,1)
t22=numpy.hstack((voit22,d22))

imref = cv2.imread("/content/drive/My Drive/Dataset/ref.png", 0)

for a in range(1, 49):
    b = str(a).zfill(3)
    st = f"/content/drive/My Drive/datasetc/fall-21-cam0-rgb/fall-21-cam0-rgb-{b}.png"
    print(st)
    im = cv2.imread(st, 0)

    if im is None:
        print(f"Erreur lors de la lecture de l'image: {st}")
        continue

    imdiff = abs(cv2.subtract(imref, im))
    print(imdiff)
    lbp = local_binary_pattern(imdiff, 8, 1, method='nri_uniform')
    (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(60))
    hist = hist.astype("float")
    hist /= (hist.sum())

    lbp2 = local_binary_pattern(imdiff, 8, 2,method='nri_uniform')
    (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
    hist2 = hist2.astype("float")
    hist2 /= (hist2.sum())

    lbp3 = local_binary_pattern(imdiff, 8, 3,method='nri_uniform')
    (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
    hist3 = hist3.astype("float")
    hist3 /= (hist3.sum())
    if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit23=hist

    else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit23=numpy.vstack([voit23,hist])


chemin = "/content/drive/My Drive/Dataset/fall-21-cam0-d/lables.csv"
df23 = pd.read_csv(chemin, index_col=0, sep=',')

d23=df23.to_numpy()
d23=numpy.subtract(d23,1)
t23=numpy.hstack((voit23,d23))

imref = cv2.imread("/content/drive/My Drive/Dataset/ref.png", 0)

for a in range(1, 49):
    b = str(a).zfill(3)
    st = f"/content/drive/My Drive/datasetc/fall-22-cam0-rgb/fall-22-cam0-rgb-{b}.png"
    print(st)
    im = cv2.imread(st, 0)

    if im is None:
        print(f"Erreur lors de la lecture de l'image: {st}")
        continue

    imdiff = abs(cv2.subtract(imref, im))
    print(imdiff)
    lbp = local_binary_pattern(imdiff, 8, 1, method='nri_uniform')
    (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(60))
    hist = hist.astype("float")
    hist /= (hist.sum())

    lbp2 = local_binary_pattern(imdiff, 8, 2,method='nri_uniform')
    (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
    hist2 = hist2.astype("float")
    hist2 /= (hist2.sum())

    lbp3 = local_binary_pattern(imdiff, 8, 3,method='nri_uniform')
    (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
    hist3 = hist3.astype("float")
    hist3 /= (hist3.sum())
    if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit24=hist

    else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit24=numpy.vstack([voit24,hist])


chemin = "/content/drive/My Drive/Dataset/fall-22-cam0-d/labels.csv"
df24 = pd.read_csv(chemin, index_col=0, sep=',')

d24=df24.to_numpy()
d24=numpy.subtract(d24,1)
t24=numpy.hstack((voit24,d24))

imref = cv2.imread("/content/drive/My Drive/Dataset/ref.png", 0)

for a in range(1, 65):
    b = str(a).zfill(3)
    st = f"/content/drive/My Drive/datasetc/fall-23-cam0-rgb/fall-23-cam0-rgb-{b}.png"
    print(st)
    im = cv2.imread(st, 0)

    if im is None:
        print(f"Erreur lors de la lecture de l'image: {st}")
        continue

    imdiff = abs(cv2.subtract(imref, im))
    print(imdiff)
    lbp = local_binary_pattern(imdiff, 8, 1, method='nri_uniform')
    (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(60))
    hist = hist.astype("float")
    hist /= (hist.sum())

    lbp2 = local_binary_pattern(imdiff, 8, 2,method='nri_uniform')
    (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
    hist2 = hist2.astype("float")
    hist2 /= (hist2.sum())

    lbp3 = local_binary_pattern(imdiff, 8, 3,method='nri_uniform')
    (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
    hist3 = hist3.astype("float")
    hist3 /= (hist3.sum())
    if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit25=hist

    else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit25=numpy.vstack([voit25,hist])


chemin = "/content/drive/My Drive/Dataset/fall-23-cam0-d/labels.csv"
df25 = pd.read_csv(chemin, index_col=0, sep=',')

d25=df25.to_numpy()
d25=numpy.subtract(d25,1)
t25=numpy.hstack((voit25,d25))

imref = cv2.imread("/content/drive/My Drive/Dataset/ref.png", 0)

for a in range(1, 65):
    b = str(a).zfill(3)
    st = f"/content/drive/My Drive/datasetc/fall-24-cam0-rgb/fall-24-cam0-rgb-{b}.png"
    print(st)
    im = cv2.imread(st, 0)

    if im is None:
        print(f"Erreur lors de la lecture de l'image: {st}")
        continue

    imdiff = abs(cv2.subtract(imref, im))
    print(imdiff)
    lbp = local_binary_pattern(imdiff, 8, 1, method='nri_uniform')
    (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(60))
    hist = hist.astype("float")
    hist /= (hist.sum())

    lbp2 = local_binary_pattern(imdiff, 8, 2,method='nri_uniform')
    (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
    hist2 = hist2.astype("float")
    hist2 /= (hist2.sum())

    lbp3 = local_binary_pattern(imdiff, 8, 3,method='nri_uniform')
    (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
    hist3 = hist3.astype("float")
    hist3 /= (hist3.sum())
    if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit26=hist

    else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit26=numpy.vstack([voit26,hist])


chemin = "/content/drive/My Drive/Dataset/fall-24-cam0-d/labels.csv"
df26 = pd.read_csv(chemin, index_col=0, sep=',')

d26=df26.to_numpy()
d26=numpy.subtract(d26,1)
t26=numpy.hstack((voit26,d26))

imref = cv2.imread("/content/drive/My Drive/Dataset/ref.png", 0)

for a in range(1, 81):
    b = str(a).zfill(3)
    st = f"/content/drive/My Drive/datasetc/fall-25-cam0-rgb/fall-25-cam0-rgb-{b}.png"
    print(st)
    im = cv2.imread(st, 0)

    if im is None:
        print(f"Erreur lors de la lecture de l'image: {st}")
        continue

    imdiff = abs(cv2.subtract(imref, im))
    print(imdiff)
    lbp = local_binary_pattern(imdiff, 8, 1, method='nri_uniform')
    (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(60))
    hist = hist.astype("float")
    hist /= (hist.sum())

    lbp2 = local_binary_pattern(imdiff, 8, 2,method='nri_uniform')
    (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
    hist2 = hist2.astype("float")
    hist2 /= (hist2.sum())

    lbp3 = local_binary_pattern(imdiff, 8, 3,method='nri_uniform')
    (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
    hist3 = hist3.astype("float")
    hist3 /= (hist3.sum())
    if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit27=hist

    else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit27=numpy.vstack([voit27,hist])


chemin = "/content/drive/My Drive/Dataset/fall-25-cam0-d/labels.csv"
df27 = pd.read_csv(chemin, index_col=0, sep=',')

d27=df27.to_numpy()
d27=numpy.subtract(d27,1)
t27=numpy.hstack((voit27,d27))

imref = cv2.imread("/content/drive/My Drive/Dataset/ref.png", 0)

for a in range(1, 65):
    b = str(a).zfill(3)
    st = f"/content/drive/My Drive/datasetc/fall-26-cam0-rgb/fall-26-cam0-rgb-{b}.png"
    print(st)
    im = cv2.imread(st, 0)

    if im is None:
        print(f"Erreur lors de la lecture de l'image: {st}")
        continue

    imdiff = abs(cv2.subtract(imref, im))
    print(imdiff)
    lbp = local_binary_pattern(imdiff, 8, 1, method='nri_uniform')
    (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(60))
    hist = hist.astype("float")
    hist /= (hist.sum())

    lbp2 = local_binary_pattern(imdiff, 8, 2,method='nri_uniform')
    (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
    hist2 = hist2.astype("float")
    hist2 /= (hist2.sum())

    lbp3 = local_binary_pattern(imdiff, 8, 3,method='nri_uniform')
    (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
    hist3 = hist3.astype("float")
    hist3 /= (hist3.sum())
    if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit28=hist

    else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit28=numpy.vstack([voit28,hist])


chemin = "/content/drive/My Drive/Dataset/fall-26-cam0-d/labels.csv"
df28 = pd.read_csv(chemin, index_col=0, sep=',')

d28=df28.to_numpy()
d28=numpy.subtract(d28,1)
t28=numpy.hstack((voit28,d28))

imref = cv2.imread("/content/drive/My Drive/Dataset/ref.png", 0)

for a in range(1, 81):
    b = str(a).zfill(3)
    st = f"/content/drive/My Drive/datasetc/fall-27-cam0-rgb/fall-27-cam0-rgb-{b}.png"
    print(st)
    im = cv2.imread(st, 0)

    if im is None:
        print(f"Erreur lors de la lecture de l'image: {st}")
        continue

    imdiff = abs(cv2.subtract(imref, im))
    print(imdiff)
    lbp = local_binary_pattern(imdiff, 8, 1, method='nri_uniform')
    (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(60))
    hist = hist.astype("float")
    hist /= (hist.sum())

    lbp2 = local_binary_pattern(imdiff, 8, 2,method='nri_uniform')
    (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
    hist2 = hist2.astype("float")
    hist2 /= (hist2.sum())

    lbp3 = local_binary_pattern(imdiff, 8, 3,method='nri_uniform')
    (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
    hist3 = hist3.astype("float")
    hist3 /= (hist3.sum())
    if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit29=hist

    else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit29=numpy.vstack([voit29,hist])


chemin = "/content/drive/My Drive/Dataset/fall-27-cam0-d/labels.csv"
df29 = pd.read_csv(chemin, index_col=0, sep=',')

d29=df29.to_numpy()
d29=numpy.subtract(d29,1)
t29=numpy.hstack((voit29,d29))

imref = cv2.imread("/content/drive/My Drive/Dataset/ref.png", 0)

for a in range(1, 65):
    b = str(a).zfill(3)
    st = f"/content/drive/My Drive/datasetc/fall-28-cam0-rgb/fall-28-cam0-rgb-{b}.png"
    print(st)
    im = cv2.imread(st, 0)

    if im is None:
        print(f"Erreur lors de la lecture de l'image: {st}")
        continue

    imdiff = abs(cv2.subtract(imref, im))
    print(imdiff)
    lbp = local_binary_pattern(imdiff, 8, 1, method='nri_uniform')
    (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(60))
    hist = hist.astype("float")
    hist /= (hist.sum())

    lbp2 = local_binary_pattern(imdiff, 8, 2,method='nri_uniform')
    (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
    hist2 = hist2.astype("float")
    hist2 /= (hist2.sum())

    lbp3 = local_binary_pattern(imdiff, 8, 3,method='nri_uniform')
    (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
    hist3 = hist3.astype("float")
    hist3 /= (hist3.sum())
    if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit30=hist

    else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit30=numpy.vstack([voit30,hist])


chemin = "/content/drive/My Drive/Dataset/fall-28-cam0-d/labels.csv"
df30 = pd.read_csv(chemin, index_col=0, sep=',')

d30=df30.to_numpy()
d30=numpy.subtract(d30,1)
t30=numpy.hstack((voit30,d30))

imref = cv2.imread("/content/drive/My Drive/Dataset/ref.png", 0)

for a in range(1, 97):
    b = str(a).zfill(3)
    st = f"/content/drive/My Drive/datasetc/fall-29-cam0-rgb/fall-29-cam0-rgb-{b}.png"
    print(st)
    im = cv2.imread(st, 0)

    if im is None:
        print(f"Erreur lors de la lecture de l'image: {st}")
        continue

    imdiff = abs(cv2.subtract(imref, im))
    print(imdiff)
    lbp = local_binary_pattern(imdiff, 8, 1, method='nri_uniform')
    (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(60))
    hist = hist.astype("float")
    hist /= (hist.sum())

    lbp2 = local_binary_pattern(imdiff, 8, 2,method='nri_uniform')
    (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
    hist2 = hist2.astype("float")
    hist2 /= (hist2.sum())

    lbp3 = local_binary_pattern(imdiff, 8, 3,method='nri_uniform')
    (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
    hist3 = hist3.astype("float")
    hist3 /= (hist3.sum())
    if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit31=hist

    else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit31=numpy.vstack([voit31,hist])


chemin = "/content/drive/My Drive/Dataset/fall-29-cam0-d/labels.csv"
df31 = pd.read_csv(chemin, index_col=0, sep=',')

d31=df31.to_numpy()
d31=numpy.subtract(d31,1)
t31=numpy.hstack((voit31,d31))

imref = cv2.imread("/content/drive/My Drive/Dataset/ref.png", 0)

for a in range(1, 65):
    b = str(a).zfill(3)
    st = f"/content/drive/My Drive/datasetc/fall-30-cam0-rgb/fall-30-cam0-rgb-{b}.png"
    print(st)
    im = cv2.imread(st, 0)

    if im is None:
        print(f"Erreur lors de la lecture de l'image: {st}")
        continue

    imdiff = abs(cv2.subtract(imref, im))
    print(imdiff)
    lbp = local_binary_pattern(imdiff, 8, 1, method='nri_uniform')
    (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(60))
    hist = hist.astype("float")
    hist /= (hist.sum())

    lbp2 = local_binary_pattern(imdiff, 8, 2,method='nri_uniform')
    (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
    hist2 = hist2.astype("float")
    hist2 /= (hist2.sum())

    lbp3 = local_binary_pattern(imdiff, 8, 3,method='nri_uniform')
    (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
    hist3 = hist3.astype("float")
    hist3 /= (hist3.sum())
    if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit32=hist

    else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit32=numpy.vstack([voit32,hist])


chemin = "/content/drive/My Drive/Dataset/fall-30-cam0-d/labels.csv"
df32 = pd.read_csv(chemin, index_col=0, sep=',')

d32=df32.to_numpy()
d32=numpy.subtract(d32,1)
t32=numpy.hstack((voit32,d32))

data177=numpy.concatenate((t,t22,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20,t21,t22,t23,t24,t25,t26,t27,t28,t29,t30),axis=0)

import pandas as pd

# Convertir le tableau numpy en DataFrame
df177 = pd.DataFrame(data177)

# Enregistrer le DataFrame en CSV
nom_fichier = "lbp.csv"
df177.to_csv(nom_fichier, index=False)

from google.colab import files
files.download(nom_fichier)

base_path = "/content/drive/My Drive/Dataset/fall-03-cam0-d/fall-03-cam0-d-"
num_images = 208
image_paths = [base_path + str(i).zfill(3) + ".png" for i in range(1, num_images+1)]

def extract_features_from_images(image_paths):
    features_list = []
    imref = cv2.imread(image_paths[0], 0)
    if imref is None:
        raise ValueError(f"L'image de référence à {image_paths[0]} n'a pas pu être chargée.")

    for path in image_paths:
        im = cv2.imread(path, 0)
        if im is None:
            raise ValueError(f"L'image à {path} n'a pas pu être chargée.")

        imdiff = abs(cv2.subtract(imref, im))
        lbp = local_binary_pattern(imdiff, 8, 1, method='nri_uniform')
        (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(60))
        hist = hist.astype("float")
        hist /= (hist.sum())

        lbp2 = local_binary_pattern(imdiff, 8, 2, method='nri_uniform')
        (hist2, _) = numpy.histogram(lbp2.ravel(), bins=numpy.arange(60))
        hist2 = hist2.astype("float")
        hist2 /= (hist2.sum())

        lbp3 = local_binary_pattern(imdiff, 8, 3, method='nri_uniform')
        (hist3, _) = numpy.histogram(lbp3.ravel(), bins=numpy.arange(60))
        hist3 = hist3.astype("float")
        hist3 /= (hist3.sum())

        hist_final = numpy.hstack([hist, hist2, hist3])
        features_list.append(hist_final)

    return features_list

features = extract_features_from_images(image_paths)

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("NomDeVotreApplication") \
    .getOrCreate()

from pyspark.ml.classification import RandomForestClassificationModel

# Charger le modèle sauvegardé depuis le chemin où vous l'avez enregistré
chemin_sauvegarde = "/content/drive/MyDrive/mon_modele_spark2"
modele_charge = RandomForestClassificationModel.load(chemin_sauvegarde)

from pyspark.ml.linalg import Vectors

# Convertir la liste de caractéristiques en DataFrame
rdd = spark.sparkContext.parallelize(features)
df = rdd.map(lambda x: (Vectors.dense(x,), )).toDF(["features"])

predictions = modele_charge.transform(df)

predicted_labels = predictions.select("prediction").collect()
predicted_labels = [row["prediction"] for row in predicted_labels]

image_predictions = list(zip(image_paths, predicted_labels))

for path, label in image_predictions:
    print(f"Chemin de l'image: {path} - Étiquette prédite: {label}")

labels = [label for _, label in image_predictions]


grouped_labels = [labels[i:i+16] for i in range(0, len(labels), 16)]


new_list = []
for group in grouped_labels:
    new_list.extend(group)


print(new_list)

from pyspark.sql import Row


rdd = spark.sparkContext.parallelize(grouped_labels)


df = rdd.map(lambda x: Row(*x)).toDF()


df.show()
