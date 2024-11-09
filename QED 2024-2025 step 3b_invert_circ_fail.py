# Version 9/8/2024
# Reorganized functions into different groups
# Added find_circum_pt
# Added find_circum_pts
# Added reformat_pts_list
# Added inv_pts() so I can mass invert the circum points of the circle
# Deleted an extra unnecessary line under Defining points and Circle: invPt = invert_pt(pt)
# Tried to change check for point being inverted being on the center from returning None to returning (math.inf, math.inf), but encountered error: posx and posy should be finite values --> change back to None for now
# Added sympy library to program
# Added function find_circ()
# Added raise ValueError() checks for if less than 3 pts were to be used to def a circ and if one of the pts being inverted is at the center of the circle of inversion
# Tried to invert and plot a circle but encountered an error - ValueError: Exceeds the limit (4300 digits) for integer string conversion; use sys.set_int_max_str_digits() to increase the limit

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import math
import sympy as sp

xpts = np.array([0, 6])
ypts = np.array([0, 250])

# Create a figure and axis
fig, ax = plt.subplots()

# Define the center and radius of the circle of inversion
center = (0, 0)
radius = 3.6

# Define calculation functions
def find_circum_pt(cent, r, angle):
    '''Returns a tuple of a single point on a circle's circumference as specified by the parameters'''
    (x0, y0) = cent
    x1 = x0 + r * math.cos(angle)
    y1 = y0 + r * math.sin(angle)
    return (x1, y1)

def find_circum_pts(cent, r, n):
    '''Returns a list of tuples of n evenly spaced points around the circle's circumference, the circle is specified by the parameters'''
    pts = []
    incrAng = 2 * math.pi / n
    for mult in range(n):
        pts.append(find_circum_pt(cent, r, mult * incrAng))  # adds each circum point to the list
    return pts

def find_circ(circumPts):
    '''Returns a tuple of the circle's center and the circle's radius given a list of three or more tuple coordinates on the circle's circumference'''
    if len(circumPts) < 3:
        raise ValueError("At least 3 points are needed to find a circle.")
    (x1,y1),(x2,y2),(x3,y3) = circumPts[0],circumPts[1],circumPts[2]
    # Define
    x0,y0,r = sp.symbols('x0 y0 r')
    # Equations for the distance from each point to the center being equal to r
    eq1 = sp.Eq((x1 - x0)**2 + (y1 - y0)**2, r**2)
    eq2 = sp.Eq((x2 - x0)**2 + (y2 - y0)**2, r**2)
    eq3 = sp.Eq((x3 - x0)**2 + (y3 - y0)**2, r**2)
    # Solve
    solutions = sp.solve([eq1, eq2, eq3], (x0, y0, r))
    # A check in case circle center becomes infinity, figure out how to tackle inf later
    if not solutions:
        raise ValueError("No solutions found for center and radius in find_circ() - is the center at infinity?")
    center = (solutions[x0], solutions[y0])
    radius = solutions[r]
    return center, radius

# Define the inversion functions
def invert_pt(coords):
    '''Returns a tuple of coordinates of the inverted point
    coords = tuple (x,y) coordinates of point being inverted'''
    if coords == (0, 0):
        # A check if we're inverting the radius -> becomes inf, will tackle inf later
        raise ValueError("One of the points you're trying to invert is at the center of the circle of inversion.")
    (x, y) = coords
    xNew = radius**2 * x / (x**2 + y**2)
    yNew = radius**2 * y / (x**2 + y**2)
    return (xNew, yNew)

def invert_pts(pts):
    '''Returns a list of tuples of the inverted points'''
    invPts = []
    for pt in pts:
        invPts.append(invert_pt(pt))  # There's a check in invert_pt() in case of one pt being inverted is at the center of the circle of inversion
    return invPts

def invert_circle(cent, r):
    '''Returns the coordinates of the inverted circle's center and the radius of the inverted circle.
    cent = coordinates of the center of the circle being inverted
    r = radius of the circle being inverted'''
    circPts = find_circum_pts(cent,r,3)  # only uses 3 points to determine a circle
    invCircPts = invert_pts(circPts)
    newCent,newRad = find_circ(invCircPts)
    return newCent,newRad

# Define graphing functions
def reformat_pts_list(pts):
    '''Reformats a list of tuple coordinates and returns two lists: one list for all the x-coords, and one list for all the y-coords'''
    xs = []
    ys = []
    for pt in pts:
        xs.append(pt[0])
        ys.append(pt[1])
    return xs, ys

def plot_pts(pts):
    '''Plots a list of tuple coordinates'''
    xs, ys = reformat_pts_list(pts)
    plt.plot(xs, ys, "o")

def plot_invert_pts(label, coords):
    '''Plots the point specified by coords and its inverted point with its specified label
    label = what the original point should be labeled as - by default, will label inverted point with the same but with an extra apostrophe
    coords = coordinates of point being inverted'''
    invCoords = invert_pt(coords)
    plt.plot(coords[0], coords[1], "o")
    plt.text(coords[0], coords[1], label, fontsize=12, ha="right")
    if invCoords != None:
        plt.plot(invCoords[0], invCoords[1], "o")
        plt.text(invCoords[0],invCoords[1],label + "'",fontsize=12,ha="right")

def graph_circle(cent, r, outlineCol='black', fillCol='none'):
    '''Graphs a circle'''
    circle = Circle(cent, r, edgecolor=outlineCol, facecolor=fillCol)
    ax.add_patch(circle)
    ax.set_xlim(-radius - 1, radius + 1)
    ax.set_ylim(-radius - 1, radius + 1)
    ax.set_aspect('equal')

def plot_invert_circle(cent,r,outlineCol1,outlineCol2,fillCol1="none",fillCol2="none"):
    '''Plots the circle specified by its center and radius and its inverted version and colors them as specified in the parameters - the first colors color the original'''
    graph_circle(cent, r, outlineCol1, fillCol1)
    newCent, newRad = invert_circle(cent, r)
    graph_circle(newCent, newRad, outlineCol2, fillCol2)

# Graph the circle of inversion
graph_circle(center, radius)

# Define the points and circle
pt = (3.4, -2.1)
cent2 = (4, -0.5)
r2 = 1
circumPts = find_circum_pts(cent2, r2, 8)

# Graph the points and circle
plt.plot(0, 0, "o")
plt.text(0, 0, "O", fontsize=12, ha="right")
plot_pts(circumPts)
plot_pts(invert_pts(circumPts))
plot_invert_pts("P", pt)
plot_invert_circle(cent2, r2, 'blue', 'red')

# Print inverted coordinates
print("P' = " + str(invert_pt(pt)))
newCenter, newRadius = invert_circle(cent2, r2)
print("New center: " + str(newCenter))
print("New radius: " + str(newRadius))

# Format the graph
plt.grid()
plt.title("Circle Inversion")
plt.show()
