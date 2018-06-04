#!/usr/bin/env python
# coding=utf-8

class node(object):
    def __init__(self,point):
        self.left = None
        self.right = None
        self.point = point
        self.parent = None #递归回退需要用到parent节点


    def set_left(self,left):
        if left == None: pass
        left.parent = self
        self.left = left

    def set_right(self,right):
        if right == None: pass
        right.parent = self
        self.right = right

def median(lst):
    m = len(lst)/2
    return lst[m], m

def build_kdtree(data,d,k):
    data = sorted(data,key=lambda x:x[d])
    p,m = median(data)
    tree = node(p)
    del data[m] #删除掉已经作为切分点的实例

    if m>0: tree.set_left(build_kdtree(data[:m], (d+1)%k, k))
    if len(data)>1: tree.set_right(build_kdtree(data[m:],(d+1)%k, k))
    return tree

def distance(a,b):
    print a,b
    return sum([(xi-xj)**2 for xi,xj in zip(a,b)])**0.5

def search_kdtree(tree, d, target,k):
    if target[d] < tree.point[d]:
        if tree.left != None:
            return search_kdtree(tree.left, (d+1)%k, target, k)
    else:
        if tree.right != None:
            return search_kdtree(tree.right,(d+1)%k, target, k)

    def update_best(t,best):
        if t == None: return
        t = t.point
        d = distance(t,target)
        if d < best[1]:
            best[1] = d #距离
            best[0] = t #实例点(节点)

    best = [tree.point, 100000.0]
    while tree.parent!=None:
        update_best(tree.parent.left, best)
        update_best(tree.parent.right, best)
        tree = tree.parent

    return best[0]


