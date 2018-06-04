#!/usr/bin/env python
# coding=utf-8
T = [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
T1 = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]

import KDTree

kd_tree = KDTree.build_kdtree(T,0,2)
print '-'*15,'[9,4]','-'*15
print KDTree.search_kdtree(kd_tree,0,[9,4],2)
kd_tree = KDTree.build_kdtree(T1,0,3)
print '-'*15,'[1.,1.,1.]','-'*15
print KDTree.search_kdtree(kd_tree,0,[1., 1., 1.],3)

#对照sklearn中实现的最近邻算法
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(T)
print '-'*15,'[9,4]','-'*15
print T
print neigh.kneighbors([[9,4]])

print '-'*15,'[1.,1.,1.]','-'*15
print T1
neigh.fit(T1)
print neigh.kneighbors([[1.,1.,1.]])
