"""
    只是用来测试一些api
"""
import torch

a = [1, 2, 3]
b = [4, 5, 6]

c = a + b
a.extend(b)

print(c)
print(a)

