#coding=utf8
"""

# Author : forin-xyz
# Created Time : Aug 24 22:44:16 2019
# Description:
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals


import numpy as np
from radam import RAdam
from keras.models import Sequential
from keras import layers as L


X = np.random.standard_normal((64 * 1000, 25))
y = np.int64(np.sum(X * X, axis=1) > 25.)


model = Sequential()
model.add(L.Dense(40, input_shape=(25,), activation="relu"))
model.add(L.Dense(64, input_shape=(40,), activation="relu"))
model.add(L.Dense(32, input_shape=(64,), activation="relu"))
model.add(L.Dropout(0.2))
model.add(L.Dense(1, activation="sigmoid"))

model.compile(RAdam(1e-4), loss="binary_crossentropy", metrics=["acc"])
model.fit(X, y, epochs=50, validation_split=0.05)


del division
del print_function
del absolute_import
del unicode_literals
