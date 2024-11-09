# Version 10/8/2024
# Added get_center & get_radius in myCircle class
# Changed invCenter and invR parameters in all relevant functions and methods to invCircle
# Deleted center and radius variables for circle of inversion

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math
import numpy as np

# Create a figure and axis
fig, ax = plt.subplots()

# Define limits
xlims = [-20,20]    # when plotting an infinite graph like a line, this defines how far in the x axis both ways the graph will stretch'
tValues = np.linspace(0,7,500)  # for getting tons of close points to graph parametrics
tDist = 7

# Define tolerance value bc of rounding errors
tolerance = 1e-9    # arbitrary small number for now


# Figure class
class Figure():
    '''Creates a figure according to a defined function and allows you to graph it and use it'''
    
    def __init__(self, xFunction, yFunction, invCircle=None, color='blue', invColor='red'):
        '''Initializes variables'''
        self.xFunc = xFunction
        self.yFunc = yFunction
        self.invCircle = invCircle   # circle of inversion
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
        invPts = invert_pts(pts,self.invCircle)
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

    def __init__(self, cent, rad=1, invCircle=None, color='blue',invColor='red'):
        '''Initializes variables'''
        self.center = cent
        self.radius = rad
        self.invCircle = invCircle   # cricle of inversion
        self.col = color
        self.invCol = invColor
        self.set_funcs()

    def x_para(self, t):
        '''Is the parametric equation for the x variable of the circle'''
        #print(t)
        return self.center[0] + self.radius * math.cos(t)

    def y_para(self, t):
        '''Is the parametric equation for the x variable of the circle'''
        return self.center[1] + self.radius * math.sin(t)

    def set_funcs(self):
        '''Sets the parametric functions to the functions of the object'''
        self.xFunc = self.x_para
        self.yFunc = self.y_para

    def get_center(self):
        '''Returns a tuple of the circle's center'''
        return self.center

    def get_radius(self):
        '''Returns the radius'''
        return self.radius

# Triangle class
class EquiTriangle(Figure):

    def __init__(self, cornerCoords, sideLength, direction=1, invCircle=None, color='blue',invColor='red'):
        '''Initializes variables'''
        self.x0 = cornerCoords[0]
        self.y0 = cornerCoords[1]
        self.n = sideLength
        self.orientation = direction  # can be 1 or -1; 1 = up,  -1 = down
        self.invCircle = invCircle   # cricle of inversion
        self.col = color
        self.invCol = invColor
        self.set_funcs()

    def x_para(self, t):
        '''Is the parametric equation for the x variable of the equilateral triangle'''
        d = t/ (tDist/3)
        # print("t = " + str(t), ", d = " + str(d))
        if 0 <= d and d <= 1:
            return self.x0 + self.n/2 * d
        elif d <= 2:
            return self.x0 + self.n/2 + self.n/2 * (d-1)
        elif d <= 3:
            return self.x0 + self.n*(3-d)
        else:
            raise ValueError("The value of t > 7 for x. The value of d is " + str(d) + "and the value of t is " + str(t))
    

    def y_para(self, t):
        '''Is the parametric equation for the y variable of the equilateral triangle'''
        d = t/ (tDist/3)
        if 0 <= d and d <= 1:
            return self.y0 + self.orientation * math.sqrt(3)/2 * self.n*d
        elif d <= 2:
            return self.y0 + self.orientation * math.sqrt(3)/2 * self.n * (2-d)
        elif d <= 3:
            return self.y0
        else:
            raise ValueError("The value of t > 7 for x. The value of d is " + str(d) + "and the value of t is " + str(t))
        
    def set_funcs(self):
        '''Sets the parametric functions to the functions of the object'''
        self.xFunc = self.x_para
        self.yFunc = self.y_para

# Line class
class myLine(Figure):

    def __init__(self, coords1, coords2, invCircle=None, color='blue', invColor='red'):
        '''Initializes variables'''
        (self.x0,self.y0) = coords1
        (self.x1,self.y1) = coords2
        #self.slope = (self.y1 - self.y0) / (self.x1 - self.x0)
        self.invCircle = invCircle   # cricle of inversion
        self.col = color
        self.invCol = invColor
        self.set_funcs()

    def x_para(self,t):
        '''Parametric equation for x variable of the line'''
        return self.x0 + (self.x1 - self.x0) * t

    def y_para(self,t):
        '''Parametric equation for y variable of the line'''
        return self.y0 + (self.y1 - self.y0) * t

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
def invert_pt(coords,invCircle):
    '''Returns a tuple of coordinates of the inverted point
    coords = tuple (x,y) coordinates of point being inverted
    invCircle = circle object representing circle of inversion'''
    if coords == invCircle.get_center():
        # A check if we're inverting the radius -> becomes inf, will tackle inf later
        raise ValueError("One of the points you're trying to invert is at the center of the circle of inversion.")
    (x, y) = coords
    (h,k) = invCircle.get_center()
    radius = invCircle.get_radius()
    xNew = h + radius**2 * (x - h) / ((x - h)**2 + (y - k)**2)
    yNew = k + radius**2 * (y - k) / ((x - h)**2 + (y - k)**2)
    return (xNew, yNew)

def invert_pts(pts, invCircle):
    '''Returns a list of tuples of the inverted points'''
    invPts = []
    for pt in pts:
        invPts.append(invert_pt(pt,invCircle))  # There's a check in invert_pt() in case of one pt being inverted is at the center of the circle of inversion
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

def plot_invert_pts(label, coords, invCircle):
    '''Plots the point specified by coords and its inverted point with its specified label
    label = what the original point should be labeled as - by default, will label inverted point with the same but with an extra apostrophe
    coords = coordinates of point being inverted'''
    invCoords = invert_pt(coords, invCircle)
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

# Graphing fractals

def graph_apollonian(invCircle, gasketType="circle"):
    '''Graphs an Apollonian Gasket, currently two types: circle and triangle.
    gasketType is circle by default.
    invCircle = circle of inversion
    invR = radius of circle of inversion'''

    # The sizes of the shapes plotted below are automatically determined according to the circle of inversion

    # Define points
    (h,k) = invCircle.get_center()
    invR = invCircle.get_radius()
##    (h,k) = invCenter

    # Graph the circle the fractal will be in
    circleD = myCircle((h,k + invR/2),invR/2,invCircle,"purple","orange")
    circleD.plot_all()

    if gasketType == "circle":
        x = h - 10*invR
        y = k + invR/2 + invR
        # ranges plot a 21 by 21 square of triangles above line of inversion
        for yNum in range(21):  # notice xNum and yNum are NOT the values of x and y but the Numth iteration of x and y
            for xNum in range(21):
                smallCirc = myCircle((x,y),invR/2,invCircle)
                smallCirc.plot_all()
                x += invR
            x = h - 10*invR
            y += invR

    if gasketType == "triangle":
        x = h - 10*invR - invR/2
        y = k + invR
        # ranges plot a 21 by 21 square of triangles above line of inversion
        for yNum in range(21):  # notice xNum and yNum are NOT the values of x and y but the Numth iteration of x and y
            for xNum in range(21):
                upTri = EquiTriangle((x,y),invR,1,invCircle)
                upTri.plot_all()
                xDown = x + invR / 2
                yDown = y + invR / 2 * math.sqrt(3)
                downTri = EquiTriangle((xDown,yDown),invR,-1,invCircle)
                downTri.plot_all()
                x += invR
            x = -21
            y += invR / 2 * math.sqrt(3)
    


# Create and graph the circle of inversion
defaultInv = myCircle((0,0),2)
#defaultInv.plot()
inv1 = myCircle((1,0),1, None, "purple")
inv1.plot()

# Define the example point
pt = (3, 0)

# # Graph the center and the example points for center of inversion = (0,0)
plt.plot(0, 0, "o")
plt.text(0, 0, "O", fontsize=12, ha="right")
plot_invert_pts("P", pt, defaultInv)
print("P' = " + str(invert_pt(pt,defaultInv)))

# Using parametrics

# Plotting Apollonian Gasket
#graph_apollonian(invCircle)

# Plotting equilateral triangle

# Imperfect Triangle Apollonian
##for x in range(-21,20,2):
##    for y in range(2,43,2):
##        testTri = EquiTriangle((x,y),2,1)
##        testTri.plot_all()

# Triangle Apollonian
#graph_apollonian(invCircle,"triangle")

# Sierpinski Triangle
mainTriangle = EquiTriangle((-1,0),2)  # no circle of inversion bc this triangle isn't being inverted
mainTriangle.plot()
line1 = myLine((0,0),(1/2,math.sqrt(3)/2),inv1)
line1.plot_all()
line2 = myLine((1/2,0),(3/4,math.sqrt(3)/4),inv1,"orange","green")
line2.plot_all()

# Format the graph
plt.grid()
plt.title("Circle Inversion")
plt.show()
