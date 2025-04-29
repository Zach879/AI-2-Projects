#Author: Zachary Reese
#File: fast.py
#Date: 2/2/2025-2/6/2025
#Due-Date: 2/7/2025 @ 5am
#Course: CPSC548
#Professor: Dr. Schwesinger
#Assignment: Project 0 (preliminary numpy practice, attendance verification, non-graded :( )
#Purpose: Convert professor supplied slow.py using python arrays and loops into efficient and simplified numpy operations.
import numpy as np

def norm(v):
    return np.sqrt(np.sum(v**2)) #normalizes the list v element-wise

def non_negative_filter(v):
    return np.where(v > 0, v ,0) #applies condition to each element

def sumproducts(v1, v2):
    return np.dot(v1, v2) #dot product between two vectors (simple matrix column/row addition)

#helpful reference explaining what mean squared error is: https://www.geeksforgeeks.org/mean-squared-error/
def mean_squared_error(v, p):
    v = np.array(v) #v is a 2d array.
    p = np.array(p).reshape(-1, 1) #reshape p into a single column with as many rows as needed for element wise matrix subtraction.
    return np.mean((v - p)**2) 

def centroid(v):
    return np.mean(v, axis=1) #average row-wise
