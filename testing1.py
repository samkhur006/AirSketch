from bisect import *
l = [0.21, 0.13, 0.33, 0.91, 1.23, 1.42, 0.234]
l.sort()
print(l)
lb = bisect_left(l, 1.334-1, lo=0, hi=len(l))
print(l[lb])
ub = bisect_right(l, 1.334-1, lo=0, hi=len(l))
print(l[ub])
# print(bisect_left(l, 1.23-1, lo=0, hi=len(l)))
# print(bisect_right(l, 1.23-1, lo=0, hi=len(l)))