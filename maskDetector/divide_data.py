import os
import splitfolders


splitfolders.ratio("Dataset/Dataset", output="Dataset_divided", seed=1337, ratio=(.85, 0.15))