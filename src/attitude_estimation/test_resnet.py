import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import time
import cv2
import math
import sys

resnet18 = torchvision.models.resnet18(pretrained=True)
print(type(resnet18))

# print(resnet18)