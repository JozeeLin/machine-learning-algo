#!/usr/bin/env python
# coding=utf-8
import copy
import itertools
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation

T = [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]

def draw_point(data):
    X,Y = [],[]
    for p in data:
        X.append(p[0])
        Y.append(p[1])
    plt.plot(X,Y,'bo')

def draw_line(xy_list):
    for xy in xy_list:
        x,y = xy
        plt.plot(x,y,'g',lw=2)
