# Version 9/7/2024
# Added function to graph circles more easily (called graph_circle())
# Deleted accidental repetition of graphing the circle of inversion found in step 
# Removed the parameter 'r' from inverting points functions because I can just use the variable 'radius' I made when defining the circle of inversion, did this to make the new invert_circle() function 'r' parameter clearer
# Corrected single indents in functions to double indents
# added an import math line
# found that invert_circle() function was incorrect: both the radius and the center were incorrect

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import math

xpts = np.array([0,6])
ypts = np.array([0,250])

# Create a figure and axis
fig, ax = plt.subplots()

# Define the center and radius of the circle
center = (0, 0)
radius = 3.6

# Define functions
def invert_pt(coords):
    '''Returns a tuple of coordinates of the inverted point
    coords = tuple (x,y) coordinates of point being inverted'''
    if coords == (0,0):
        return None           # add a check if we're inverting the radius -> becomes inf
    (x,y) = coords
    xNew = radius**2 * x / (x**2 + y**2)
    yNew = radius**2 * y / (x**2 + y**2)
    return (xNew, yNew)

def invert_circle(cent, r):
    '''Returns the coordinates of the inverted circle's center and the radius of the inverted circle.
    cent = coordinates of the center of the circle being inverted
    r = radius of the circle being inverted'''
    (x0,y0) = cent
    (x1,y1) = (x0+r,y0)  # finds another point on the circumference of the circle
    invCent = invert_pt(cent)   # finds new inverted center of circle
    invBord = invert_pt((x1,y1))
    newRad = math.sqrt((invBord[0]-invCent[0])**2 + (invBord[1]-invCent[1])**2)
    return invCent,newRad

def plot_invert_pts(label, coords):
    '''Plots the point specified by coords and its inverted point with its specified label
    label = what the original point should be labeled as - by default, will label inverted point with the same but with an extra apostrophe
    coords = coordinates of point being inverted'''
    invCoords = invert_pt(coords)
    plt.plot(coords[0],coords[1],"o")
    plt.text(coords[0],coords[1],label,fontsize=12,ha="right")
    if invCoords != None:
        plt.plot(invCoords[0],invCoords[1],"o")
        plt.text(invCoords[0],invCoords[1],label+"'",fontsize=12,ha="right")

def graph_circle(cent, r, outlineCol='black', fillCol='none'):
    '''Graphs a circle'''
    circle = Circle(cent, r, edgecolor=outlineCol, facecolor=fillCol)
    ax.add_patch(circle)
    ax.set_xlim(-radius-1, radius+1)
    ax.set_ylim(-radius-1, radius+1)
    ax.set_aspect('equal')

def plot_invert_circle(cent, r, outlineCol1, outlineCol2, fillCol1="none", fillCol2="none"):
    '''Plots the circle specified by its center and radius and its inverted version and colors them as specified in the parameters - the first colors color the original'''
    graph_circle(cent,r,outlineCol1,fillCol1)
    newCent,newRad = invert_circle(cent,r)
    graph_circle(newCent,newRad,outlineCol2,fillCol2)

# Graph the circle of inversion
graph_circle(center, radius)

# Define the points and circle
pt = (3.4,-2.1)
invPt = invert_pt(pt)
cent2 = (4,-0.5)
r2 = 1

# Graph the points and circle
plt.plot(0,0,"o")
plt.text(0,0,"O",fontsize=12,ha="right")
plot_invert_pts("P",pt)
plot_invert_circle(cent2,r2,'blue','red')

# Print inverted coordinates
print("P' = " + str(invert_pt(pt)))
newCenter,newRadius = invert_circle(cent2,r2)
print("New center: " + str(newCenter))
print("New radius: " + str(newRadius))

# Format the graph
plt.grid()
plt.title("Circle Inversion")
plt.show()
