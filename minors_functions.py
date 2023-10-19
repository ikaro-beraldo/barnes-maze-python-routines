# Set of functions regarding minor calculations

import math
import numpy as np
import time
import sympy as sp


def isInside(circle_x, circle_y, rad, x, y):
     
    # Compare radius of circle
    # with distance of its center
    # from given point
    if ((x - circle_x) * (x - circle_x) +
        (y - circle_y) * (y - circle_y) <= rad * rad):
        return True;
    else:
        return False;

def mov_average(arr, window_size):
    # Program to calculate moving average 
    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []

    # Loop through the array to consider
    # every window of size 3
    while i < len(arr) - window_size + 1:

        # Store elements from i to i+window_size
        # in list to get the current window
        window = arr[i : i + window_size]

        # Calculate the average of current window
        window_average = round(sum(window) / window_size, 2)

        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)

        # Shift window to right by one position
        i += 1

    return moving_averages


# A utility function to calculate area
# of triangle formed by (x1, y1),
# (x2, y2) and (x3, y3)
 
def area(x1, y1, x2, y2, x3, y3):
 
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1)
                + x3 * (y1 - y2)) / 2.0)
 
 
# A function to check whether point P(x, y)
# lies inside the triangle formed by
# A(x1, y1), B(x2, y2) and C(x3, y3)
def isInside_triangle(triangle_matrix, x, y):
    
    x1 = triangle_matrix[0,0]
    y1 = triangle_matrix[0,1]
    x2 = triangle_matrix[1,0]
    y2 = triangle_matrix[1,1]
    x3 = triangle_matrix[2,0]
    y3 = triangle_matrix[2,1]
 
    # Calculate area of triangle ABC
    A = area (x1, y1, x2, y2, x3, y3)
 
    # Calculate area of triangle PBC
    A1 = area (x, y, x2, y2, x3, y3)
     
    # Calculate area of triangle PAC
    A2 = area (x1, y1, x, y, x3, y3)
     
    # Calculate area of triangle PAB
    A3 = area (x1, y1, x2, y2, x, y)
    
    soma = A1 + A2 + A3
     
    # Check if sum of A1, A2 and A3
    # is same as A
    if(A == A1 + A2 + A3):
        return True
    else:
        return False
    

# A function to check whether point P(x, y)
# lies inside the triangle formed by
# A(x1, y1), B(x2, y2) and C(x3, y3)
def isInside_triangle_dif(triangle_matrix, x, y):
    
    x1 = triangle_matrix[0,0]
    y1 = triangle_matrix[0,1]
    x2 = triangle_matrix[1,0]
    y2 = triangle_matrix[1,1]
    x3 = triangle_matrix[2,0]
    y3 = triangle_matrix[2,1]
 
    # Calculate area of triangle ABC
    A = area (x1, y1, x2, y2, x3, y3)
 
    # Calculate area of triangle PBC
    A1 = area (x, y, x2, y2, x3, y3)
     
    # Calculate area of triangle PAC
    A2 = area (x1, y1, x, y, x3, y3)
     
    # Calculate area of triangle PAB
    A3 = area (x1, y1, x2, y2, x, y)
    
    return abs((A1 + A2 + A3)-A)

def is_point_inside_ellipse(x, y, a, b, h, k, A):
    
    beg = time.time()
    # Convert parameters to regular Python floats
    a, b, h, k, A = float(a), float(b), float(h), float(k), float(A)

    # Translate and rotate the points to the ellipse's coordinate system
    x_translated = (x - h) * np.cos(A) + (y - k) * np.sin(A)
    y_translated = (x - h) * np.sin(A) - (y - k) * np.cos(A)

    # Check if the points are inside the ellipse (inequality check)
    is_inside = (x_translated / a)**2 + (y_translated / b)**2 < 1

    print(time.time()-beg)
    return is_inside

from scipy.optimize import fsolve

def ellipse_equation_x(x, y, a, b, h, k, A):
    cos_A = np.cos(A)
    sin_A = np.sin(A)
    return ((x - h) * cos_A + (y - k) * sin_A)**2 / a**2 + ((x - h) * sin_A - (y - k) * cos_A)**2 / b**2 - 1

def ellipse_equation_y(y, x, a, b, h, k, A):
    cos_A = np.cos(A)
    sin_A = np.sin(A)
    return ((x - h) * cos_A + (y - k) * sin_A)**2 / a**2 + ((x - h) * sin_A - (y - k) * cos_A)**2 / b**2 - 1

def find_line_ellipse_intersections(a, b, h, k, A, line_a, line_b, num_points=5):
    line_x = np.linspace(h - a, h + a, num_points)
    line_y = line_a * line_x + line_b

    intersection_points = []
    for x_val, y_val in zip(line_x, line_y):
        x_intersections = fsolve(ellipse_equation_x, x_val, args=(y_val, a, b, h, k, A))
        y_intersections = fsolve(ellipse_equation_y, y_val, args=(x_val, a, b, h, k, A))
        x_valid = np.all(np.abs(ellipse_equation_x(x_intersections, y_val, a, b, h, k, A)) < 1e-6)
        y_valid = np.all(np.abs(ellipse_equation_y(y_intersections, x_val, a, b, h, k, A)) < 1e-6)
        if x_valid and y_valid:
            intersection_points.append((x_intersections[0], y_intersections[0]))

    return np.array(intersection_points)

from numpy import ones,vstack
from numpy.linalg import lstsq

def solve_m_c_linear_equation(points):
    #points = [(1,5),(3,4)]
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords,ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords)[0]
    
    return m, c        


from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def are_points_inside_polygon(polygon_points, points_to_check):
    polygon = Polygon(polygon_points)
    results = [polygon.contains(Point(x, y)) for x, y in points_to_check]
    return results

# Example usage
polygon_points = np.asarray([(0, 0), (0, 1), (1, 1), (1, 0)])
points_to_check = np.asarray([(0.5, 0.5), (0.2, 0.2), (1.8, 0.8)])

results = are_points_inside_polygon(polygon_points, points_to_check)
print(results)  # Output: [True, True, False]

      
 
