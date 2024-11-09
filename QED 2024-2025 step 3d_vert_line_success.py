# Version 9/14/2024
# Added early return statement to find_circ() to return "inf","inf" for center and radius if it is a line
# Moved running plot_inf_line() from find_circ() to plot_invert_circle()
# Had invert_circle() also return an additional list of the circumPts so plot_invert_circle() would be able to run plot_inf_line()
# Added a lineCol parameter to plot_inf_line() so outline of inverted plot would always be red or whatever I chose it to be
# Added trash variable circPts in the print inverted coordinates section because invert_circle() now had to return three things

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math

# Create a figure and axis
fig, ax = plt.subplots()

# Define the center and radius of the circle of inversion
center = (0, 0)
radius = 3.6

# Define limits
xlims = [-10,10]    # when plotting an infinite graph like a line, this defines how far in the x axis both ways the graph will stretch'

# Define tolerance value bc of rounding errors
tolerance = 1e-9    # arbitrary small number for now

# Define calculation functionsF
def find_slope(pt1,pt2):
    '''Takes in two tuple coordinates and returns the slope of the two points'''
    if abs(pt2[0] - pt1[0]) < tolerance:    # if it's close enough to zero bc rounding error
        # raise ValueError("Uh oh: the line through " + str(pt1) + " and " + str(pt2) + " is a vertical line.")
        return 'undef'
    m = (pt2[1]-pt1[1]) / (pt2[0]-pt1[0])
    return m

def is_collinear(pt1,pt2,pt3):
    '''Finds if the three tuple coordinates submitted are collinear or not and returns a boolean'''
    # Slopes
    m1 = find_slope(pt1,pt2)
    m2 = find_slope(pt2,pt3)
    m3 = find_slope(pt1,pt3)

    # If any slopes are vertical
    if m1 == 'undef' or m2 == 'undef' or m3 == 'undef':
        return m1 == m2 == m3     # I can't figure out how I would incorporate tolerance into this so I'll just do this for now

    return abs(m1 - m2) < tolerance and abs(m2 - m3) < tolerance

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
    '''Returns a tuple of the circle's center (unless the circle is a line, then it returns "inf") and the circle's radius ("inf" if it's a line) given a list of three or more tuple coordinates on the circle's circumference'''
    print("Running find_circ() for the following points: " + str(circumPts))
    # Error checks
    if len(circumPts) < 3:
        raise ValueError("At least 3 points are needed to find a circle.")
    (x1,y1),(x2,y2),(x3,y3) = circumPts[0],circumPts[1],circumPts[2]

    if is_collinear(circumPts[0],circumPts[1],circumPts[2]):
        # raise ValueError("The three circumference points are collinear - the result is a line")
        # I can use this part to actually draw a line and correctly model inversion
        # xs,ys = reformat_pts_list([circumPts[0],circumPts[1]])
        # plt.plot(xs, ys)
        print("In find_circ: the circumPts are collinear.")
        # plot_inf_line(circumPts[0],circumPts[1])

        return "inf","inf"   # returns it to plot_invert_circle(), where a conditional checks for it

    # Define variables needed for Cramer's Rule
    a = 2 * (x2 - x1)
    b = 2 * (y2 - y1)
    u = (x2**2 + y2**2) - (x1**2 + y1**2)
    c = 2 * (x3 - x1)
    d = 2 * (y3 - y1)
    v = (x3**2 + y3**2) - (x1**2 + y1**2)

    # Solve using Cramer's Rule
    h = (d*u - b*v) / (a*d - b*c)
    k = (a*v - c*u) / (a*d - b*c)
    cent = (h,k)

    # Solve for radius
    r = math.dist(cent,(x1,y1))

    return cent, r

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
    '''Returns the coordinates of the inverted circle's center, the radius of the inverted circle, and a list of the circumference points of the inverted circle.
    cent = coordinates of the center of the circle being inverted
    r = radius of the circle being inverted'''
    print("Running invert_circle() for center = " + str(cent) + " and radius = " + str(r))
    circPts = find_circum_pts(cent,r,3)  # only uses 3 points to determine a circle
    invCircPts = invert_pts(circPts)
    newCent,newRad = find_circ(invCircPts)
    return newCent,newRad,invCircPts

# Define graphing functions
def plot_inf_line(p1,p2,lineCol):
    '''Takes in two tuple coordinates on a line and a line color and plots the line visually extending endlessly in both directions such that the line does not appear to be a segment.'''
    print("Running plot_inf_line() for p1 = " + str(p1) + " and p2 = " + str(p2))
    m = find_slope(p1,p2)
    if m == 'undef':    # if it's a vertical line
        print("The slope between " + str(p1) + " and " + str(p2) + " is undefined.")
        plt.axvline(x=p1[0], c = lineCol)
    else:
        b = p1[1] - m * p1[0]    # calculates the intercept of the line
        xs = xlims
        ys = [m * x + b for x in xs]
        plt.plot(xs,ys, c = lineCol)

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
    print("Graphing a circle with center " + str(cent) + " and radius " + str(r))
    circle = Circle(cent, r, edgecolor=outlineCol, facecolor=fillCol)
    ax.add_patch(circle)
    ax.set_xlim(-radius - 1, radius + 1)
    ax.set_ylim(-radius - 1, radius + 1)
    ax.set_aspect('equal')

def plot_invert_circle(cent,r,outlineCol1,outlineCol2,fillCol1="none",fillCol2="none"):
    '''Plots the circle specified by its center and radius and its inverted version and colors them as specified in the parameters - the first colors color the original'''
    graph_circle(cent, r, outlineCol1, fillCol1)
    newCent, newRad, circumPts = invert_circle(cent, r)
    if newCent == "inf" and newRad == "inf":
        plot_inf_line(circumPts[0],circumPts[1], outlineCol2)
    else:
        graph_circle(newCent, newRad, outlineCol2, fillCol2)

# Graph the circle of inversion
graph_circle(center, radius)

# Define the points and circle
pt = (3.4, -2.1)
cent2 = (3.6, 0)
r2 = math.dist(cent2,center)
circumPts = find_circum_pts(cent2, r2, 3)

# Graph the points and circle
plt.plot(0, 0, "o")
plt.text(0, 0, "O", fontsize=12, ha="right")
plot_pts(circumPts)
plot_pts(invert_pts(circumPts))
plot_invert_pts("P", pt)
plot_invert_circle(cent2, r2, 'blue', 'red')

# Print inverted coordinates
print("P' = " + str(invert_pt(pt)))
newCenter, newRadius, circPts = invert_circle(cent2, r2)   # circPts is useless, it's just because invert_circle() now returns 3 points so I kind of had to add that dumb variable there
print("New center: " + str(newCenter))
print("New radius: " + str(newRadius))

# Format the graph
plt.grid()
plt.title("Circle Inversion")
plt.show()
