# NECESSARY PYTHON LIBRARIES
import os
import numpy as np
import shutil

# DATASET PATH, CATEGORIES
DATASET = 'C:/Users/can_a/PycharmProjects/Assignment3/DatasetResized'
CATEGORIES = ['airport_inside', 'artstudio', 'bakery', 'bar', 'bathroom', 'bedroom', 'bookstore',
              'bowling', 'buffet', 'casino', 'church_inside', 'classroom', 'closet', 'clothingstore',
              'computerroom']

# CREATING TRAIN TEST SET (0.30 TEST, 0.70 TRAIN)
for i in CATEGORIES:
    os.makedirs('train/' + i)
    os.makedirs('test/' + i)
    src = DATASET + '/' + i

    all_images = os.listdir(src)
    np.random.shuffle(all_images)

    test_ratio = 0.3
    train_set, test_set = np.split(np.array(all_images), [int(len(all_images) * (1 - test_ratio))])

    train_set = [src + '/' + name for name in train_set.tolist()]
    test_set = [src + '/' + name for name in test_set.tolist()]

    for name in train_set:
        shutil.copy(name, 'train/' + i)

    for name in test_set:
        shutil.copy(name, 'test/' + i)





