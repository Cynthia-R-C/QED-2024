# Version 9/18/2024
# Define circle D, created circleD_x() and circleD_y()
# Defined circle A and respective functions
# Set default graphing colors to blue and red in Figure class
# Added plot_all() in Figure class
# Defined circles B and C
# Defined child class of Figure called Circle and created init and methods
# Got unexpected error because Circle class has same name as matplotlib Circle patch
# Renamed Circle class to myCircle
# Fixed a few bugs (e.g. used radius instead of self.radius in para_x and para_y)
# Created default radius
# Deleted circleD_x, circleD_y, circleA_x, etc.

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math
import numpy as np

# Create a figure and axis
fig, ax = plt.subplots()

# Define the center and radius of the circle of inversion
center = (0, 0)
radius = 2

# Define limits
xlims = [-20,20]    # when plotting an infinite graph like a line, this defines how far in the x axis both ways the graph will stretch'
tValues = np.linspace(xlims[0],xlims[1],500)  # for getting tons of close points to graph parametrics

# Define tolerance value bc of rounding errors
tolerance = 1e-9    # arbitrary small number for now


# Figure class
class Figure():
    '''Creates a figure according to a defined function and allows you to graph it and use it'''
    
    def __init__(self, xFunction, yFunction, color='blue', invColor='red'):
        '''Initializes variables'''
        self.xFunc = xFunction
        self.yFunc = yFunction
        self.col = color
        self.invCol = invColor

    def find_pts(self):
        '''Returns two lists, one of xs, one of ys, of a parametric graph for the predefined range of t values'''
        xs = []
        ys = []
        for t in tValues:
            xs.append(self.xFunc(t))
            ys.append(self.yFunc(t))
        return xs, ys

    def plot(self):
        '''Plots the figure in the graph'''
        xs,ys = self.find_pts()
        plt.plot(xs,ys,c=self.col)

    def invert_pts_xylist(self):
        '''Returns two lists of the inverted points in the predefined range of t values: one of the x values, and one of the y values, to be used for parametrics'''
        pts = []
        for t in tValues:
            pts.append((self.xFunc(t),self.yFunc(t)))
        invPts = invert_pts(pts)
        xt = []
        yt = []
        for invPt in invPts:
            xt.append(invPt[0])
            yt.append(invPt[1])
        return xt, yt

    def plot_inverse(self):
        '''Plots the figure inverted across the circle'''
        xt,yt = self.invert_pts_xylist()
        plt.plot(xt,yt,c=self.invCol)

    def plot_all(self):
        '''Plots the figure and the figure's inverse'''
        self.plot()
        self.plot_inverse()


# Circle class
class myCircle(Figure):    # purpose is to automate making circles so I can test the Apollonian Gasket, generalizes circle eq and applies them

    def __init__(self, cent, rad=1, color='blue',invColor='red'):
        '''Initializes variables'''
        self.center = cent
        self.radius = rad
        self.col = color
        self.invCol = invColor
        self.set_funcs()

    def x_para(self, t):
        '''Is the parametric equation for the x variable of the circle'''
        return self.center[0] + self.radius * math.cos(t)

    def y_para(self, t):
        '''Is the parametric equation for the x variable of the circle'''
        return self.center[1] + self.radius * math.sin(t)

    def set_funcs(self):
        '''Sets the parametric functions to the functions of the object'''
        self.xFunc = self.x_para
        self.yFunc = self.y_para


# Define calculation functions
def test_func_x(t):
    '''A test function for parametrics for x'''
    return 4 + math.cos(t)

def test_func_y(t):
    '''A test function for parametrics for y'''
    return -0.5 + math.sin(t)



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
    plt.plot(coords[0], coords[1], "o", c="black")
    plt.text(coords[0], coords[1], label, fontsize=12, ha="right")
    if invCoords != None:
        plt.plot(invCoords[0], invCoords[1], "o", c="black")
        plt.text(invCoords[0],invCoords[1],label + "'",fontsize=12,ha="right")

def graph_circle(cent, r, outlineCol='black', fillCol='none'):
    '''Graphs a circle'''
    print("Graphing a circle with center " + str(cent) + " and radius " + str(r))
    circle = Circle(cent, r, edgecolor=outlineCol, facecolor=fillCol)
    ax.add_patch(circle)
    ax.set_xlim(-radius - 1, radius + 1)
    ax.set_ylim(-radius - 1, radius + 1)
    ax.set_aspect('equal')


# Graph the circle of inversion
graph_circle(center, radius)

# Define the example point
pt = (3, 0)

# # Graph the center and the example points
plt.plot(0, 0, "o")
plt.text(0, 0, "O", fontsize=12, ha="right")
plot_invert_pts("P", pt)
print("P' = " + str(invert_pt(pt)))

# Using parametrics
circleD = myCircle((0,1),1,"purple","orange")
circleD.plot_all()

# Plotting Apollonian Gasket
for x in range(-20,21,2):
    for y in range(3,44,2):
        smallCirc = myCircle((x,y))
        smallCirc.plot_all()


##circleA = myCircle((0,3))
##circleA.plot_all()
##circleB = myCircle((-2,3))
##circleB.plot_all()
##circleC = myCircle((2,3))
##circleC.plot_all()

# Format the graph
plt.grid()
plt.title("Circle Inversion")
plt.show()
