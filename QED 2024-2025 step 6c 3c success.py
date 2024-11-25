# Version 11/24/2024
# Made transform_to_create_future_layers()
# Added set_center() in myCircle class
# Fixed tDist location
# Imported copy module for making deepcopies
# Fixed issue of transform_to_create_future_layers() changing the location of the original up2dicts[firstR] every time
# Tested cases 2, 5, 6, 7 - all worked as intended!! Runtime = 1 minute (with print statements removed)

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math
import cmath
import numpy as np
import sympy as sp
from sympy.solvers import solve
from sympy import Symbol
from sympy.abc import x, y
from fractions import Fraction
import copy

# Create a figure and axis
fig, ax = plt.subplots()

# Define limits
xlims = [-20,20]    # when plotting an infinite graph like a line, this defines how far in the x axis both ways the graph will stretch'
tDist = 15
tValues = np.linspace(0,tDist,500)  # for getting tons of close points to graph parametrics

# Define tolerance value bc of rounding errors
tolerance = 1e-7    # Change to 1 for tests #5 and #6


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

    def set_color(self, color):
        '''Sets the color of the figure'''
        self.col = color

    def get_color(self):
        '''Returns a string of the color of the figure'''
        return self.col

    def get_invCirc(self):
        '''Returns the invCircle of the Figure'''
        return self.invCircle


# Line class
class myLine(Figure):

    def __init__(self, coords1, coords2, invCircle=None, color='blue', invColor='red'):
        '''Initializes variables; coords1 is the starting point of the ray'''
        (self.x0,self.y0) = coords1
        (self.x1,self.y1) = coords2
        #self.slope = (self.y1 - self.y0) / (self.x1 - self.x0)
        self.invCircle = invCircle   # circle of inversion
        self.col = color
        self.invCol = invColor
        self.set_funcs()

    def __str__(self):
        '''Returns a string specifying the line'''
        return "myLine with pt1 = " + str((self.x0,self.y0)) + ", p2 = " + str((self.x1,self.y1)) + ", and invCirc " + str(self.invCircle)

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

    def get_pt1(self):
        '''Returns the coords1 entered for myLine'''
        return (self.x0,self.y0)

    def get_pt2(self):
        '''Returns the coords2 entered for myLine'''
        return (self.x1,self.y1)


# Circle class
class myCircle(Figure):    # purpose is to automate making circles so I can test the Apollonian Gasket, generalizes circle eq and applies them

    def __init__(self, cent, rad=1, invCircle=None, color='blue',invColor='red'):
        '''Initializes variables'''
        self.center = cent
        self.radius = rad
        self.invCircle = invCircle   # circle of inversion
        self.col = color
        self.invCol = invColor
        self.set_funcs()

    def __str__(self):
        return "myCircle at " + str(self.center) + ", r = " + str(self.radius)

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

    def set_center(self,newCent):
        '''Allows the user to manually set the center'''
        self.center = newCent

    def get_center(self):
        '''Returns a tuple of the circle's center'''
        return self.center

    def get_radius(self):
        '''Returns the radius'''
        return self.radius

    def get_inv(self):
        '''Returns an object that is the result of this circle inverted over the circle of inversion;
        Returns a circle type object if the result is a circle; else returns a line type object'''

        (x0,y0) = self.invCircle.get_center()
        R = self.invCircle.get_radius()
        (x,y) = self.center
        r = self.radius

        if (x - x0)**2 + (y - y0)**2 <= (r + tolerance)**2 and (x - x0)**2 + (y - y0)**2 >= (r - tolerance)**2:  # if this circle passes through the center of inversion
            xt,yt = self.invert_pts_xylist()
            pt1 = (xt[0],yt[0])
            pt2 = (xt[1],yt[1])
            return myLine(pt1,pt2,self.invCircle)
        
        else:
            D = math.sqrt((x-x0)**2 + (y-y0)**2)
            rp = (R**2 * r) / (D**2 - r**2)

            # Finding the invCent
            circumPts = find_circum_pts(self.center,self.radius,3)
            invCircumPts = invert_pts(circumPts,self.invCircle)
            invCent,rj = find_circ(invCircumPts)  # rj is unused in this case - find_circ was an old function
            
            return myCircle(invCent,rp,self.invCircle)
        

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


# Old calculation functions re-added in
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
    '''Returns a tuple of the circle's center and the circle's radius given a list of three or more tuple coordinates on the circle's circumference
    Assumes the circle is not a line'''
    #print("Running find_circ() for the following points: " + str(circumPts))
    # Error checks
    if len(circumPts) < 3:
        raise ValueError("At least 3 points are needed to find a circle.")
    (x1,y1),(x2,y2),(x3,y3) = circumPts[0],circumPts[1],circumPts[2]

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
            x = h - 10*invR - invR/2
            y += invR / 2 * math.sqrt(3)


# Generalizing Apollonian Gaskets

    # First 4 circles

def is_collinear(pt1, pt2, pt3):
    '''Returns a boolean of if the 3 points are collinear'''
    if pt1[0] == pt2[0] or pt2[0] == pt3[0]:
        return pt1[0] == pt2[0] and pt2[0] == pt3[0]
    else:
        return (pt1[1]-pt2[1])/(pt1[0]-pt2[0]) == (pt2[1]-pt3[1])/(pt2[0]-pt3[0])   # possibility for rounding error?

def calc_rd(ra,rb,rc):
    '''Returns list of solutions to solving for rd'''

    # Trying it with sympy
    rd = Symbol('rd')
    solutions = solve((1/ra + 1/rb + 1/rc - 1/rd)**2 - 2 * (1/ra**2 + 1/rb**2 + 1/rc**2 + 1/rd**2),rd)
    for i in range(len(solutions)):
        solutions[i] = float(solutions[i])    # there were a lot of issues with cmath evaluations later because the float types of sympy and regular Python were different
    return solutions

def find_possible_rds(circA, circB, circC):
    '''Returns the radius of the circle circumscribing the first three circles of an Apollonian Gasket'''
    ra = circA.get_radius()
    rb = circB.get_radius()
    rc = circC.get_radius()
    (ha,ka) = circA.get_center()
    (hb,kb) = circB.get_center()
    (hc,kc) = circC.get_center()

    # Check if the circles are collinear with identical radii
    if is_collinear((ha,ka),(hb,kb),(hc,kc)) and ra == rb and rb == rc:
        return "inf"

    # Now plug it into Descartes
    solutions = calc_rd(ra,rb,rc)

    # Add condition here: rd > 0; rd actually works (plug back in) condition will be eval in find_circD
    for s in solutions:
        if s <= 0:
            solutions.remove(s)

    return solutions

def find_d_center(circA, circB, circC, rd):   # This will NOT RUN if rd is infinite
    '''Returns the a tuple of center coordinates of the circle circumscribing the first three circles of an Apollonian gasket'''
    (ha,ka) = circA.get_center()
    (hb,kb) = circB.get_center()
    (hc,kc) = circC.get_center()
    ra = circA.get_radius()
    rb = circB.get_radius()
    rc = circC.get_radius()

    # Let these be the curvatures of circles A, B, and C
    ca = 1/ra
    cb = 1/rb
    cc = 1/rc
    cd = 1/rd

    # Using Descartes' complex number equation for center
    za = complex(ha,ka)
    zb = complex(hb,kb)
    zc = complex(hc,kc)

    # To check for the type - for testing
    #print(type(za), type(zb), type(zc), type(ca), type(cb), type(cc), type(cd))


    # A list of solutions to account for the plus or minus?
    zds = [(za*ca + zb*cb + zc*cc + 2*cmath.sqrt(ca*cb*za*zb + cb*cc*zb*zc + ca*cc*za*zc))/-cd, (za*ca + zb*cb + zc*cc - 2*cmath.sqrt(ca*cb*za*zb + cb*cc*zb*zc + ca*cc*za*zc))/-cd]

    solutions = []
    for zd in zds:
        print(zd)
        solutions.append((zd.real,zd.imag))

    return solutions

def is_between(p, p1, p2):
    '''Returns a boolean for if p is between points p1 and p2 (points are tuples)'''
    xBetween = (p1[0] <= p[0] and p[0] <= p2[0]) or (p2[0] <= p[0] and p[0] <= p1[0])
    yBetween = (p1[1] <= p[1] and p[1] <= p2[1]) or (p2[1] <= p[1] and p[1] <= p1[1])
    return xBetween and yBetween

def is_within_tolerance(p1,p2):
    '''Returns if the two points should be counted as the same point'''
    return math.dist(p1,p2) < tolerance

def rd_works(circA, circB, circC, rd, center):
    '''Returns a boolean for if a certain rd works'''

    (hd,kd) = center
    print("Running rd_works for rd = " + str(rd) + ", center " + str((hd,kd)))   # for testing

    # Testing points on line distance for each circle

    counter = 0     # for testing, keeping track of which circle it is
    
    for circ in (circA,circB,circC):
        counter += 1    # for testing
        (h,k) = circ.get_center()
        r = circ.get_radius()
        
        # First find the equation of the line

        equationType = ""    # if "v" the slope is vertical, if "m" then has slope
        
        if hd - h == 0:  # then the slope is a vertical line

            # Testing
            print("The slope is a vertical line")
            print("hd,kd: " + str((hd,kd)))
            print("rd: " + str(rd))
            print("(h,k): " + str((h,k)))
            print("r: " + str(r))
            circLetter = ""
            if counter == 1:
                circLetter = "A"
            elif counter == 2:
                circLetter = "B"
            elif counter == 3:
                circLetter = "C"
            else:
                circLetter = "somehow counter > 3"
            print("circLetter: " + circLetter)
            # Testing End
            
            equationType = "v"

        else:
            equationType = "m"
            slope = (kd-k)/(hd-h)

        # Now solve for intersections

        # Intersections with A/B/C
        if equationType == "v":    # if the slope is vertical
            interEs = solve([x-h,(x-h)**2 + (y-k)**2 - r**2], [x,y], dict=True)
        else:
            interEs = solve([y-k - (slope * (x-h)),(x-h)**2 + (y-k)**2 - r**2], [x,y], dict=True)

        # Eliminate solutions that are between the radii of A/B/C and D - the incorrect intersections
        for solution in interEs:
            x1 = solution[x]
            y1 = solution[y]
            if is_between((x1,y1),(h,k),(hd,kd)):
                interEs.remove(solution)

        print("interEs: " + str(interEs))   # for testing

        # Intersections with D
        if equationType == "v":    # if the slope is vertical
            interDs = solve([x-h,(x-hd)**2 + (y-kd)**2 - rd**2], [x,y], dict=True)
        else:
            interDs = solve([y-k - (slope * (x-h)),(x-hd)**2 + (y-kd)**2 - rd**2], [x,y], dict=True)

        # Eliminate solutions that are on the other side of circle D - the incorrect intersections
        for solution in interDs:
            x1 = solution[x]
            y1 = solution[y]
            if math.dist((x1,y1),(h,k)) > rd:   # if it's on the other side of circle D; won't be >= bc (h,k)!=(hd,kd)
                interDs.remove(solution)

        print("interDs: " + str(interDs))   # for testing

        # Theoretically there should now be only one solution in interEs and one solution in interDs
        if len(interEs) != 1 or len(interDs) != 1:
            raise ValueError("One of interEs and interDs doesn't only have 1 solution in it!")
        else:
            x1 = interEs[0][x]
            y1 = interEs[0][y]
            x2 = interDs[0][x]
            y2 = interDs[0][y]
            
            if is_within_tolerance((x1,y1),(x2,y2)):   # check if the two should be counted as the same point
                return True
            else:
                return False
            

def find_circD(circA, circB, circC):
    '''Returns the center and radius of the circle circumscribing the first three circles of the Apollonian gasket'''
    rds = find_possible_rds(circA, circB, circC)
    print("rds from find_possible_rds: " + str(rds))   # for testing
    (ha,ka) = circA.get_center()
    (hb,kb) = circB.get_center()
    (hc,kc) = circC.get_center()
    ra = circA.get_radius()
    rb = circB.get_radius()
    rc = circC.get_radius()

    # First check to see if rd is infinite - if so, DON'T run find_d_center
    if rds == 'inf':
        return 'inf','inf'

    if len(rds) == 0:
        print("No solutions for rd reported from find_possible_rds.")   # for testing
        return None, None

    # To keep track of solutions that work, since find_d_center can now output more than one solution and thus
    # find_d_center()[0] won't work for returning anymore
    validSolutions = {}   # will be a dictionary of r:center pairs, e.g. {1:(0,0), 4:(1,3)}

    for rd in rds:    # tests if each rd works: if it doesn't, it's removed from the rd possibilities
        centerSolutions = find_d_center(circA, circB, circC, rd)
        print("For rd = " + str(rd) + ", the center solutions are " + str(centerSolutions))
        for (hd,kd) in centerSolutions:
            if not rd_works(circA, circB, circC, rd, (hd,kd)):
                print("Center D = " + str((hd,kd)) + " is being removed.")
                centerSolutions.remove((hd,kd))
        if len(centerSolutions) != 1:   # if there aren't any possible centers left or more than 1 left
            print("centerSolutions: " + str(centerSolutions))
            print("rd = " + str(rd) + " is being removed.")
            rds.remove(rd)
        else:
            validSolutions[rd] = centerSolutions[0]    # adds a r:center pair to validSolutions

    if len(rds)!= 1:     # if there isn't only one possibility for rd left
        print("There isn't only 1 possibility for rd: the possibilities are " + str(rds))   # for testing
        return None, None
    else:
        rd = rds[0]
        (hd,kd) = validSolutions[rd]   # draws on the valid center for rd stored in validSolutions

    print("Circ D returned from find_circD: rd = " + str(rd) + ", cent = " + str((hd,kd)))
    return (hd,kd), rd

def find_tangent_lines(circA, circB, circC):
    '''Is for the condition of circD having an infinite radius - will only be run if so
    Returns solutions of coordinates of the top and bottom tangent points to circle A and the slope'''
    ra = circA.get_radius()
    rb = circB.get_radius()
    #rc = circC.get_radius()
    (ha,ka) = circA.get_center()
    (hb,kb) = circB.get_center()
    #(hc,kc) = circC.get_center()

    # Here, m represents the slope of the tangent lines that will be drawn

    # Address cases here: m = vertical or m = horizontal (making -1/m undefined)
    if hb-ha == 0:   # if m is vertical
        m = 'undef'
        solutions = [{x: ra, y: 0},{x: -1*ra, y: 0}]
    elif kb-ka == 0:  # if m is horizontal
        m = 0
        solutions = [{x: 0, y: ra},{x: 0, y: -1*ra}]
    else:
        m = (kb-ka)/(hb-ha)
        solutions = solve([y/x + 1/m, sp.sqrt(x**2 + y**2) - ra], [x,y], dict=True)   # remember, this solves for the CHANGE in x and y from the center of A to the tangent points
    print("dx and dy solutions: " + str(solutions))     # testing
    return solutions,m

def find_unscaled_ABC(ra,rb,rc):
    '''Returns (ha,ka),(hb,kb),(hc,kc);
    Assumes rd != inf'''
    (ha,ka) = (0,0)
    (hb,kb) = (ra+rb,0)

    # Find center of C here and add check with Triangle Inequality
    s1 = ra+rb
    s2 = ra+rc
    s3 = rb+rc

    if not (s1+s2>s3 and s1+s3>s2 and s2+s3>s1):    # if doesn't satisfy Triangle Inequality
        raise ValueError("The three initial radii provided are invalid.")
    else:
        solutions = solve([x**2 + y**2 - (ra + rc)**2, (x - hb)**2 + y**2 - (rb + rc)**2], [x,y], dict=True)

    # Remove upper solution of circ C to adjust for new def of A,B,C set on pg. 36 of research records
    for solution in solutions:
        if not (solution[y] <= ka and solution[y] <= kb):
            solutions.remove(solution)

    if len(solutions) != 1:    # if the solutions list is now empty or if there's more than 1 left (idk how that could be, but just in case)
        raise ValueError("The three radii provided are not initial - no single circC center can be found.")
    
    (hc,kc) = (solutions[0][x],solutions[0][y])
    return (ha,ka), (hb,kb), (hc,kc)

def scale_transl_circE(centE,re,centD,rd):
    '''Returns scaled (he,ke), re of one of A,B,C specified in the centers and radii provided
    Assums rd != inf'''

    # Testing
    print("centD used in scale/transl: " + str(centD))
    print("rd used in scale/transl: " + str(rd))

    # Translation + Scaling
    he = (centE[0] - centD[0]) / rd
    ke = (centE[1] - centD[1]) / rd

    # Scaling
    re /= rd

    return (he,ke),re

def rd_within_ABC_cents(centA,centB,centC,centD):
    '''Returns a boolean: if centD is inclusively within the bounds of centA, centB, and centC,
    thus meaning are centA, centB, and centC really three INITIAL gasket circles;
    Assumes that rd != inf'''
    
    (ha,ka) = centA
    (hb,kb) = centB
    (hc,kc) = centC
    (hd,kd) = centD

    # For testing
    print("Running rd_within_ABC_cents():")
    print("(ha,ka) = " + str((ha,ka)))
    print("(hb,kb) = " + str((hb,kb)))
    print("(hc,kc) = " + str((hc,kc)))
    print("(hd,kd) = " + str((hd,kd)))
    
    yesAB = kd - ka <= (ka-kb)/(ha-hb) * (hd - ha)    # if centD is within bounds of line from centA to centB
    yesBC = kd - kb >= (kc-kb)/(hc-hb) * (hd - hb)    # if centD is within bounds of line from centC to centB
    yesAC = kd - ka >= (ka-kc)/(ha-hc) * (hd - ha)    # if centD is within bounds of line from centA to centB

    # For testing
    print("yesAB: " + str(yesAB))
    print("yesBC: " + str(yesBC))
    print("yesAC: " + str(yesAC))

    return yesAB and yesBC and yesAC
    

def first_four(ra, rb, rc, circO, config="circle",graph=True,returnLists=False):
    '''Finds the first four circles of the Apollonian gasket - also graphs circle O
    If config="straight", will run calculations for a straight configuration; else, runs calculations for a circular config
    If returnLists=True, will return lists of final ABCDs and inverted versions and circO
    If graph=True, will graph the final ABCDs and inverted versions'''

    if graph:
        circO.plot()

    # For find_new_pos later
    listABCD = []
    invABCDs = []

    if config == "straight":     # for cases where rd = inf
        if not ra == rb and rb == rc:
            raise ValueError("The three initial radii for a straight config are invalid.")
        else:    # the scaling for rd = inf has been moved to the bottom
            (ha,ka) = (0,0)
            (hb,kb) = (2*ra,0)
            (hc,kc) = (4*ra,0)

    else:
        (ha,ka), (hb,kb), (hc,kc) = find_unscaled_ABC(ra,rb,rc)
        # Scaling and translating is down at the end for this because we need to find and validate rd before using it

    # Circs A, B, and C are redefined later after scaling and translation
    circA = myCircle((ha,ka), ra, circO)
    circB = myCircle((hb,kb), rb, circO)
    circC = myCircle((hc,kc), rc, circO)

    # Testing
    print("Circ A orig center: " + str((ha,ka)))
    print("Circ B orig center: " + str((hb,kb)))
    print("Circ C orig center: " + str((hc,kc)))

    centD,rd = find_circD(circA, circB, circC)
    (ha,ka) = circA.get_center()

    # Add check for circD to see if it returns 'inf','inf' or None, None
    if centD == None:   # since it's both or neither I really only have to check one of them
        raise ValueError("The three initial circles for the gasket are invalid.")   # might change this later when I start the user input to have the user reenter the circles

        # comment out above portion when testing and uncomment below pass
        #pass
        
    elif centD == 'inf':

        # Scaling and Translating for Straight Configs
        if config == "straight":
            (ha,ka) = (-2,0)   # there's only really one possible config for a straight config without orientation
            (hb,kb) = (0,0)
            (hc,kc) = (2,0)
            ra = 1
            rb = 1
            rc = 1

        # Testing
        print("Circ A center: " + str((ha,ka)))
        print("Circ B center: " + str((hb,kb)))
        print("Circ C center: " + str((hc,kc)))

        circA = myCircle((ha,ka), ra, circO, "purple")
        circB = myCircle((hb,kb), rb, circO)
        circC = myCircle((hc,kc), rc, circO, "green")

        if graph:   # if the graph function is turned
            circA.plot_all()
            circB.plot_all()
            circC.plot_all()

        # Plotting circ D

        ys = []  # for find_new_pos later; now I know there's really only going to be horizontal lines, so I only need to keep track of yms
        
        solutions, m = find_tangent_lines(circA, circB, circC)
        for solution in solutions:
            #  Each solution in the list is a dictionary
            dx = solution[x]   # delta x = change in x
            dy = solution[y]   # delta y = change in y
            xm = ha + dx       # is the coordinates of one of the two tangent points (top and bottom)
            ym = ka + dy
            ys.append(ym)    # Assumes only horizontal lines are possible

            # Testing
            print("(xm, ym) = " + str((xm,ym)))
            print("m = " + str(m))

            if graph:

                # Add cases here for if m = 'undef' or if m = 0
                if m == 0:    # draw horizontal tangent lines
                    print("Plotting horizontal tangent lines")
                    plt.axhline(ym,xlims[0],xlims[1],color="gray")  # just use same x limits for y limits
                elif m == 'undef':    # draw vertical tangent lines
                    print("Plotting vertical tangent lines")
                    plt.axvline(xm,xlims[0],xlims[1],color="gray")   
                else:
                    print("Plotting a sloped line")
                    plt.axline((float(xm),float(ym)),slope=float(m),color="gray")

        # For find_new_pos later

        # A check just in case
        if len(ys) != 2:
            raise ValueError("len(ys) != 2: ys = " + str(ys))

        # Note for this if rd = inf the circ D element will be a list of lines
        listABCD = [circA,circB,circC,[myLine((xlims[0],ys[0]),(0,ys[0]),circO),myLine((xlims[0],ys[1]),(0,ys[1]),circO)]]

        # If the initial config is a straight config, there is only one possible location for the two lines resulting from
        # the inversion of circle D per restrictions set on the representation of a gasket
        invABCDs = [circA.get_inv(),circB.get_inv(),circC.get_inv(),[myLine((xlims[0],-1),(0,-1),circO),myLine((xlims[0],0),(0,0),circO)]]

    else:
        
        # Scaling and Translating Non-Straight Configs
        
        circCount = 0   # for keeping track of which of circA, B, and C is in the loop below
        for cent in [(ha,ka),(hb,kb),(hc,kc)]:
            circCount += 1
            rs = {1: ra, 2:rb, 3:rc, 4: "count is somehow past 3"}
            cent, r = scale_transl_circE(cent,rs[circCount],centD,rd)
            if circCount == 1:
                (ha,ka) = cent
                ra = r
            elif circCount == 2:
                (hb,kb) = cent
                rb = r
            elif circCount == 3:
                (hc,kc) = cent
                rc = r
            else:
                print("circCount is somehow past 3")
        print("Plotting circle D: center (0,0), radius 1")   # for testing
        circD = myCircle((0,0),1,circO, "gray")

        # Testing
        print("Circ A center: " + str((ha,ka)))
        print("Circ B center: " + str((hb,kb)))
        print("Circ C center: " + str((hc,kc)))

        circA = myCircle((ha,ka), ra, circO, "purple")
        circB = myCircle((hb,kb), rb, circO)
        circC = myCircle((hc,kc), rc, circO, "green")

        # Final check for invalid initial 3 circles: if all else passes but the 3 circs aren't the INITIAL ones
        if not rd_within_ABC_cents((ha,ka),(hb,kb),(hc,kc),(0,0)):
            raise ValueError("The 3 circles you entered are not initial circles.")

        if graph:
            circA.plot_all()
            circB.plot_all()
            circC.plot_all()
            circD.plot_all()

        # For find_new_pos later
        listABCD = [circA,circB,circC,circD]

        # If the initial config is a straight config, there is only one possible location for the two lines resulting from
        # the inversion of circle D per restrictions set on the representation of a gasket
        invABCDs = [circA.get_inv(),circB.get_inv(),circC.get_inv(),circD.get_inv()]

    # For find_new_pos later
    if returnLists:
        return listABCD, invABCDs, circO


    # Ford Circles

def find_third_radius(rp,rq):
    '''Returns the radius of a smaller circle tangent to circle P, circle Q, and a horizontal line tangent to both circles'''
    k = 1/math.sqrt(rp) + 1/math.sqrt(rq)    # used existing equations
    rr = (1/k)**2
    return rr

def find_third_center(centP,rp,rr,vloc,hloc=1):
    '''Returns the center of the third smaller circle with same setting as previous function
    vloc = 1 or -1, specifying top or bottom respectively
    hloc = 1 or -1, specifying right or left circle respectively'''
    (hp,kp) = centP
    if hloc == 1:
        hr = hp + 2 * math.sqrt(rr*rp)
    elif hloc == -1:
        hr = hp - 2 * math.sqrt(rr*rp)
    else:
        raise ValueError("Invalid hloc entered for find_third_center: " + str(hloc))
    if vloc == 1:
        kr = kp + (rp - rr)
    elif vloc == -1:
        kr = kp - (rp - rr)
    else:
        raise ValueError("Invalid vloc entered for find_third_center: " + str(vloc))
    return (hr,kr)

def find_third_circ(circP, circQ, vloc, hloc=1):
    '''Returns the center and radius of a smaller circle tangent to circle A, circle B, and a horizontal line tangent
    to both circles;
    vloc = 1 or -1, specifying the line being on top or on bottom respectively
    hloc = 1 or -1, specifying right or left circle respectively'''
    rp = circP.get_radius()
    (hp, kp) = circP.get_center()
    rq = circQ.get_radius()
    rr = find_third_radius(rp,rq)
    (hr,kr) = find_third_center((hp,kp),rp,rr,vloc,hloc)
    return (hr,kr),rr

def find_rdp(circ1,circ2,circ3):
    '''Returns the list of possible radii of the circle externally tangent to 3 other surrounding circles;
    Assumes the circles are not both collinear and of the same radius; run in find_Dprime()
    Assumes the third lim is not a line but a circle'''
    r1 = circ1.get_radius()
    r2 = circ2.get_radius()
    r3 = circ3.get_radius()
    (h1,k1) = circ1.get_center()
    (h2,k2) = circ2.get_center()
    (h3,k3) = circ3.get_center()

    # Now plug it into Descartes
    rdp = Symbol('rdp')
    solutions = solve((1/r1 + 1/r2 + 1/r3 + 1/rdp)**2 - 2 * (1/r1**2 + 1/r2**2 + 1/r3**2 + 1/rdp**2),rdp)
    for i in range(len(solutions)):
        solutions[i] = float(solutions[i])  # I've had a lot of issues with sympy datatypes, best to convert to float to be safe

    # Get rid of negative solutions
    for s in solutions:
        if s <= 0:
            solutions.remove(s)
        elif s >= r1 or s >= r2 or s >= r3:  # the solution to the error that took me 3 hours to figure out... pg. 44 of records
            solutions.remove(s)

    if len(solutions) != 1:
        print("There isn't only one possible solution for rdp. The solutions are: " + str(solutions))
        print("r1 = " + str(r1) + ", r2 = " + str(r2) + ", r3 = " + str(r3))
    return solutions

def one_Dp_intersection_works(circ,rdp,center,counter):
    '''Returns a boolean for if a certain Dp center and radius works for a certain limit,
    which is specified by the circ parameter; the counter is for ease of testing'''
    (hdp,kdp) = center
    (h,k) = circ.get_center()
    r = circ.get_radius()

    if counter == 1:
        limNum = "lim1"
    elif counter == 2:
        limNum = "lim2"
    elif counter == 3:
        limNum = "lim3"
    else:
        limNum = "somehow counter > 3"

    print("\nChecking intersections for " + limNum)
    print("lim center = " + str((h,k)) + ", lim radius = " + str(r))
    print("Dp center = " + str((hdp,kdp)) + ", rdp = " + str(rdp))
    
    # First find the equation of the line

    equationType = ""    # if "v" the slope is vertical, if "m" then has slope
    
    if hdp - h == 0:  # then the slope is a vertical line

        # Testing
        print("The slope is a vertical line")
        print("hdp,kdp: " + str((hdp,kdp)))
        print("rdp: " + str(rdp))
        print("(h,k): " + str((h,k)))
        print("r: " + str(r))
        circLetter = str(counter)
        circLetter = "somehow counter > 3"
        print("circLetter: " + circLetter)
        # Testing End
        
        equationType = "v"

    else:
        equationType = "m"
        slope = (kdp-k)/(hdp-h)

    # Now solve for intersections

    # Intersections with 1/2/3
    if equationType == "v":    # if the slope is vertical
        interEs = solve([x-h,(x-h)**2 + (y-k)**2 - r**2], [x,y], dict=True)
    else:
        interEs = solve([y-k - (slope * (x-h)),(x-h)**2 + (y-k)**2 - r**2], [x,y], dict=True)

    # Eliminate solutions that are NOT between the radii of 1/2/3 and Dp - the incorrect intersections
    for solution in interEs:
        x1 = solution[x]
        y1 = solution[y]
        if not is_between((x1,y1),(h,k),(hdp,kdp)):
            interEs.remove(solution)

    print("interEs for Dp: " + str(interEs))   # for testing

    # Intersections with D
    if equationType == "v":    # if the slope is vertical
        interDs = solve([x-h,(x-hdp)**2 + (y-kdp)**2 - rdp**2], [x,y], dict=True)
    else:
        
        interDs = solve([y-k - (slope * (x-h)),(x-hdp)**2 + (y-kdp)**2 - rdp**2], [x,y], dict=True)

    # Eliminate solutions for interDs in the same way
    for solution in interDs:
        x1 = solution[x]
        y1 = solution[y]
        if not is_between((x1,y1),(h,k),(hdp,kdp)):
            interDs.remove(solution)

    print("interDs for Dp: " + str(interDs))   # for testing

    # Theoretically there should now be only one solution in interEs and one solution in interDs
    if len(interEs) != 1 or len(interDs) != 1:
        return False
    else:
        x1 = interEs[0][x]
        y1 = interEs[0][y]
        x2 = interDs[0][x]
        y2 = interDs[0][y]
        
        if is_within_tolerance((x1,y1),(x2,y2)):   # check if the two should be counted as the same point
            return True
        else:
            return False


def Dp_cent_works(circ1, circ2, circ3, rdp, center):
    '''Returns a boolean for if a certain Dp center and radius works; run in find_Dprime
    Assumes the third limit is a circle'''

    (hdp,kdp) = center
    print("\nRunning Dp_cent_works for rdp = " + str(rdp) + ", center " + str((hdp,kdp)))   # for testing

    # Testing points on line distance for each circle

    counter = 0     # for testing, keeping track of which circle it is
    limNum = ""    # Also for testing
    worksList = []  # A list of the boolean values of if each of the intersections of Dprime with the limits work
    
    for circ in [circ1,circ2,circ3]:
        counter += 1    # for testing
        worksList.append(one_Dp_intersection_works(circ,rdp,center,counter))

    print("worksList: " + str(worksList))  # testing

    # If intersections with all the limits are valid, the center works
    if worksList[0] == True and worksList[1] == True and worksList[2] == True:
        print("Valid Dp cent: rdp = " + str(rdp) + ", center = " + str(center) + " works!")
        return True
    else:
        print("Invalid Dp cent: rdp = " + str(rdp) + ", center = " + str(center) + " doesn't work.")
        return False
        

def find_Dp_centers(lim1,lim2,lim3,rdp):
    '''Returns the list of tuples of center coordinates of the circle externally tangent to the three limits;
    Assumes the third limit isn't a line but a circle; run in find_Dprime'''
    (h1,k1) = lim1.get_center()
    (h2,k2) = lim2.get_center()
    (h3,k3) = lim3.get_center()
    r1 = lim1.get_radius()
    r2 = lim2.get_radius()
    r3 = lim3.get_radius()

    # Let these be the curvatures of circles A, B, and C
    c1 = 1/r1
    c2 = 1/r2
    c3 = 1/r3
    cdp = 1/rdp

    # Using Descartes' complex number equation for center
    z1 = complex(h1,k1)
    z2 = complex(h2,k2)
    z3 = complex(h3,k3)


    # A list of solutions to account for the plus or minus
    zdps = [(z1*c1 + z2*c2 + z3*c3 + 2*cmath.sqrt(c1*c2*z1*z2 + c2*c3*z2*z3 + c1*c3*z1*z3))/cdp, (z1*c1 + z2*c2 + z3*c3 - 2*cmath.sqrt(c1*c2*z1*z2 + c2*c3*z2*z3 + c1*c3*z1*z3))/cdp]

    solutions = []
    for zdp in zdps:
        solutions.append((zdp.real,zdp.imag))

##    for solution in solutions:
##        if not Dp_cent_works(lim1,lim2,lim3,rdp,solution):
##            solutions.remove(solution)
##
##    if len(solutions) != 1:
##        raise ValueError("There isn't only 1 possible solution for Dp center. The solutions are: " + str(solutions))

    print("\nDp possible centers found for rdp = " + str(rdp) + ": " + str(solutions))
    return solutions


def find_Dprime(lim1,lim2,lim3,lim3Type="circle"):
    '''Returns the center and radius of circle Dprime given:
    lim1,lim2: myCircle type
    lim3: myCircle type or an integer 1 or -1 specifying vloc of line; hloc = right for finding Dprime
    lim3Type: can be "circle" or "line"'''

    if lim3Type == "line":   # if lim3 is the top or bottom line
        (h,k),r = find_third_circ(lim1,lim2,lim3)
    else:    # if lim3 is a circle
        rs = find_rdp(lim1,lim2,lim3)

        # Eliminate impossible solutions
        for r in rs:
            possibleCents = find_Dp_centers(lim1,lim2,lim3,r)
            for cent in possibleCents:
                if not Dp_cent_works(lim1, lim2, lim3, r, cent):
                    possibleCents.remove(cent)
            print("\n\nConsidering possibility r = " + str(r) + ", cents = " + str(possibleCents))
            print("len(possibleCents) in eliminating impossible solutions: " + str(len(possibleCents)) + "\n\n")
            if len(possibleCents) != 1:
                print("There isn't not only one possible center for rdp = " + str(r) + ": the possible centers are " + str(possibleCents))
                print("Removing rdp = " + str(r) + " as a possibility.")
                rs.remove(r)
            else:
                print("There is only one possible center for rdp = " + str(r) + ". It is a valid possibility.")

        # Now choose the remaining r
        if len(rs) != 1:
            print("\n\nPossible radius,center pairs:")
            for r in rs:
                possibleCents = find_Dp_centers(lim1,lim2,lim3,r)
                for cent in possibleCents:  
                    if not Dp_cent_works(lim1, lim2, lim3, r, cent):
                        possibleCents.remove(cent)
                    print("\nr = " + str(r) + ", cents = " + str(possibleCents))
                    print("len(possibleCents) = " + str(len(possibleCents)))
            raise ValueError("There isn't only 1 possible rdp: possible rdps = " + str(rs))
        else:
            r = rs[0]
            possibleCents = find_Dp_centers(lim1,lim2,lim3,r)
            for cent in possibleCents:  
                if not Dp_cent_works(lim1, lim2, lim3, r, cent):
                    possibleCents.remove(cent)
            (h,k) = possibleCents[0]  # if r is still in rs, that means there's only 1 center possible after elimination

    print("\nDprime found: center " + str((h,k)) + ", radius " + str(r) + "\n")
    return (h,k),r

def find_three_circs(currCirc,limsList,lim3Type,circO):
    '''Returns three myCircle objects that represent the next three recursive Ford circles given the limits of currCircle;
    circT = bottom left, circU = top, circV = bottom right; or if the line is on top circT = top left, circU = bottom, circV = top right'''

    circs = {}     # will be a dictionary of myCircle: limsList for that circle
    for limpair in [(1,3),(1,2),(2,3)]:   # numbers represent lim#; e.g. 1 = lim1, 2 = lim2 etc.; in order that would yield circs T,U,V

        # For figuring out if currCirc is lim1, lim2, or lim3
        limOrders = ['','','']
        newLimType = ""    # because one of the next circles might not have a line as a limit

        # First assign the locations of the lims we already know the numbers of
        for i in limpair:
            limOrders[i-1] = limsList[i-1]

        # Then assign currCirc to the only remaining empty spot left
        limOrders[limOrders.index('')] = currCirc

        print("\n\nFor limpair = " + str(limpair))
        print("limOrders in find_three_circles: ",end="")  # testing
        for lim in limOrders:
            print(str(lim) + ", ",end="")
        print("\n")

        # Finally, determine the newLimType of the new circle
        numInts = 0   # number of integer limits in limOrders
        for lim in limOrders:
            if type(lim) == int:
                numInts += 1
        if numInts != 0:   # if one of the limits is indeed an integer, thus a line
            newLimType = "line"
        else:
            newLimType = "circle"

        # Now find the circle and add it to the list
        print("\nRunning find_Dprime for above limOrders")
        (h,k),r = find_Dprime(limOrders[0],limOrders[1],limOrders[2],newLimType)
        circM = myCircle((h,k),r,circO)
        circs[circM] = limOrders

    return circs    # returns a dictionary with circ:lims pairs, e.g. {myCircDatatype: [circ1,circ2,currCirc], anotherCirc: ...}


def next_three(currCirc,limsList,countNum,circO,graph=True):
    '''The recursive function to graph and store the next iteration of three Ford circles beginning the third layer
    of Ford Circle generation.
    Enter an integer, 1 or -1 (specifying top or bottom), for lim3, if lim3 is a line.'''

    print("\n\n\nRunning next_three for " + str(currCirc))

    # For plugging into functions
    if type(limsList[2]) == int:
        lim3Type = "line"
    else:
        lim3Type = "circle"

    print("lim3Type = " + lim3Type)  # for testing

    # Find the next three circles and their respective limits to plug into next_three again
    circsTUV = find_three_circs(currCirc,limsList,lim3Type,circO)
    # Looks like {circT: limsT, circU: limsU, circV: limsV}

    # Testing
    print("\ncountNum = " + str(countNum))
    print("circsTUV's lims = ")
    for circ in circsTUV.keys():
        print("For " + str(circ) + ", lims are: ",end="")
        for lim in circsTUV[circ]:
            print(str(lim) + ", ")
        print("\n")
    print("\n")

    # Create the dictionary to store {circT: {circX:...}, circU: {circX:...}, circV:...}
    TUVdict = {}

    if not countNum == 1:   # if this isn't the last recursion to do
        #circCounter = 0    # to keep track of whether it's circle T, U, or V; e.g. for naming keys in the TUVdict
        for circ in circsTUV.keys():   # for each of circs T, U, and V
            #circCounter += 1
            if graph:
                circ.plot()
            subdict = next_three(circ, circsTUV[circ], countNum - 1, circO,graph)  # runs the recursive function for each of T, U, V

            # To add later subdict of subdicts generated recursively to TUVdict            
            TUVdict[circ] = subdict    # I'm not defining the keys by a callable name anymore; this is a problem for when inverting Ford circles

    else:    # if this IS the last recursion to do
        for circ in circsTUV.keys():   # for each of circs T, U, and V
            #circCounter += 1
            if graph:
                circ.plot()
            # Theoretically, per recursion, T,U,V is now being stored as the three "next" options for circP, e.g.
            # {circP: {circT: ..., circU: ..., circV: ...}}
            # So as an end to the recursion, we'll fill in the "..." with NoneTypes instead of more dictionaries
            TUVdict[circ] = None

    return TUVdict

def transform_to_create_future_layers(currDict,n,dx,vloc,graph=True):
    '''Returns a new dictionary of the same circles but with their centers transformed
    accordingly
    dx = x coordinate central position (e.g. change in x from the currDict)
    vloc = top or bottom
    graph = boolean for whether to graph it or not'''
    #print("Running transform_to_create...")
    #circCount = 0
    for nextCirc in currDict.keys():
        #circCount += 1
        #print("circCount: " + str(circCount) + ", iteration = " + str(n))
        currCent = nextCirc.get_center()
        newCent = (currCent[0] + dx, currCent[1]*vloc)
        nextCirc.set_center(newCent)
        if graph:
            nextCirc.plot()
        if n != 1:   # if this isn't the last iteration
            currDict[nextCirc] = transform_to_create_future_layers(currDict[nextCirc],n-1,dx,vloc,graph)
        else:
            pass
    return currDict

def create_Ford_Circles(circO, recurseNum,graph=True,returnList=False):
    '''Graphs and stores an initial configuration of Ford Circles'''
    circles = []    # stores the circles; one list of two dictionaries (top/bottom) for each layer (except layer 0 and 1)


    # Zeroth Layer - aka tangent lines
    layer0 = []
    layer0s = {}   # only one dictionary for layer 0
    layer0s["line1"] = myLine((xlims[0],1),(0,1),circO)
    layer0s["line2"] = myLine((xlims[0],-1),(0,-1),circO)
    if graph:
        layer0s["line1"].plot()
        layer0s["line2"].plot()
    layer0.append(layer0s)
    circles.append(layer0)
    
    # First Layer
    layer1 = []
    layer1s = {}   # only one dictionary for layer 1
    circCounter = 0
    for x in range(xlims[0], xlims[1] + 1, 2):  # Because xlims[0] is already neg, just multiply by positive number; circles start from circ1, then circ2, etc.
        circCounter += 1
        layer1s["circ" + str(circCounter)] = myCircle((x,0),1,circO)
        if graph:
            layer1s["circ" + str(circCounter)].plot()
    layer1.append(layer1s)
    circles.append(layer1)

    # Second Layer
    layer2 = []
    #circles.append(layer2)    # will really be appending tree dict nests in next section to circles
    up2 = {}
    layer2.append(up2)
    down2 = {}
    layer2.append(down2)
    for vloc in [1,-1]:    # means top and bottom
        circCounter = 0
        if vloc == 1:
            dictLoc = 0   # to access up2 from layer2
        else:
            dictLoc = 1   # to access down2 from layer2
        for circNum in range(1,len(layer1s)):    # should go from 1 to n-1 if the length of layer1s is n
            circCounter += 1
            circP = layer1s["circ" + str(circNum)]
            circQ = layer1s["circ" + str(circNum + 1)]
            (hr,kr),rr = find_third_circ(circP,circQ,vloc)
            layer2[dictLoc]["circ" + str(circCounter)] = myCircle((hr,kr),rr,circO)
            if graph:
                layer2[dictLoc]["circ" + str(circCounter)].plot()

    # Rest of Layers
    # Now entering the tree dictionary nesting
    layer2dicts = []
    circles.append(layer2dicts)
    up2dicts = {}
    layer2dicts.append(up2dicts)
    down2dicts = {}
    layer2dicts.append(down2dicts)

    # Doing it once for the leftmost configuration
    # circNum = 1 in this case

    # Defining initial limits and currCirc: refer to pg. 39 research records diagrams for P,Q,R, line L definitions
    firstP = layer1s["circ1"]
    firstQ = layer1s["circ2"]
    firstR = layer2[0]["circ1"]
    up2dicts[firstR] = next_three(firstR,[firstP,firstQ,1],recurseNum,circO,graph)
    down2dicts[firstR] = transform_to_create_future_layers(copy.deepcopy(up2dicts[firstR]),recurseNum,0,-1,graph)
    
    # Do it for the rest
    for vloc in [1,-1]:    # means top and bottom
        circCounter = 0   # for keeping track of which # circle it is in layer 2
        if vloc == 1:
            dictLoc = 0   # to access up2 from layer2
        else:
            dictLoc = 1   # to access down2 from layer2
        for circNum in range(2,len(layer2[dictLoc])+1):    # should go from 2 to n if the length of one top/bottom of layer2 is n
            circCounter += 1

            # Defining initial limits and currCirc: refer to pg. 39 research records diagrams for P,Q,R, line L definitions
            circR = layer2[dictLoc]["circ" + str(circNum)]

            # Calculate dx
            dx = 2 * (circNum - 1)   # calculates the amount needed to translate the circles horizontally

            # Make a copy so the changes aren't accidentally layered on top of the original dictionary - see pg. 54 research records
            copyDict = copy.deepcopy(up2dicts[firstR])
            print("Location of copy: " + str(list(copyDict.keys())[0]))

            if vloc == 1:
                up2dicts[circR] = transform_to_create_future_layers(copyDict,recurseNum,dx,1,graph)
            elif vloc == -1: 
                down2dicts[circR] = transform_to_create_future_layers(copyDict,recurseNum,dx,-1,graph)
            else:
                raise ValueError("Somehow vloc is not 1 or -1??")

            print("New location of copy: " + str(list(copyDict.keys())[0]))
    

##    # Running the recursions
##    for vloc in [1,-1]:    # means top and bottom
##        circCounter = 0   # for keeping track of which # circle it is in layer 2
##        if vloc == 1:
##            dictLoc = 0   # to access up2 from layer2
##        else:
##            dictLoc = 1   # to access down2 from layer2
##        for circNum in range(1,len(layer2[dictLoc])+1):    # should go from 1 to n if the length of one top/bottom of layer2 is n
##            circCounter += 1
##
##            # Defining initial limits and currCirc: refer to pg. 39 research records diagrams for P,Q,R, line L definitions
##            circP = layer1s["circ" + str(circNum)]  
##            circQ = layer1s["circ" + str(circNum + 1)]
##            circR = layer2[dictLoc]["circ" + str(circNum)]
##
##            # Recurse
##            if vloc == 1:
##                up2dicts[circR] = next_three(circR,[circP,circQ,1],recurseNum,circO,graph)
##            else:
##                down2dicts[circR] = next_three(circR,[circP,circQ,-1],recurseNum,circO,graph)

    if returnList:
        return circles


# Scaling & Positioning Initial Gasket Circles

def scale_and_transl(fig, ratio, dx, dy, figType="circle",invCirc="None"):
    '''Returns a new object after scaling and translating a figure given the ratio, dx, dy, and the invCircle;
    Figure type can only be circle or line'''

    if figType == "circle":

        # Find radius
        newR = fig.get_radius() * ratio

        # Find center
        newCent = ((fig.get_center()[0] + dx) * ratio, (fig.get_center()[1] + dy) * ratio)

        return myCircle(newCent,newR,invCirc)

    elif figType == "line":

        # Find points
        newPt1 = ((fig.get_pt1()[0] + dx) * ratio, (fig.get_pt1()[1] + dy) * ratio)
        newPt2 = ((fig.get_pt2()[0] + dx) * ratio, (fig.get_pt2()[1] + dy) * ratio)

        return myLine(newPt1, newPt2, invCirc)
        

def find_new_pos_straight(circsABCD,invCircles,circO):
    '''Returns a new list of circsABCD and a new circO to adjust to the default Ford Circle
    config created in the program given the invCircles entered;
    Assumes this is a straight ABC config case, in which rd = inf'''
    print("\n\nRunning find_new_pos_straight()...\n\n")  # for tests

    newABCDs = []
    newInvs = []

    # Picking a random circle, e.g. circle A
    rRatio = 1 / invCircles[0].get_radius()
    dx = 0 - invCircles[0].get_center()[0]
    dy = 0 - invCircles[0].get_center()[1]

    # Applying the translation / scaling

    # First make the new circO to redefine all the circles towards (as the new circle of inversion)
    newCircO = scale_and_transl(circO,rRatio,dx,dy,"circle")   # it has no invCircle because it IS the invCircle
    
    for fig in circsABCD:  # for the non-inverted circles

        if type(fig) == list:   # if it's looking at the list of lines for circD
            figList = []
            for figure in fig:
                newFigure = scale_and_transl(figure, rRatio, dx, dy, "line", newCircO)
                figList.append(newFigure)
            newABCDs.append(figList)

        else:   # if it isn't looking at the list of lines for circD
            newFig = scale_and_transl(fig, rRatio, dx, dy, "circle", newCircO)
            newABCDs.append(newFig)

    for fig in invCircles:  # for the inverted circles

        if type(fig) == list:   # if it's looking at the list of lines for circD
            figList = []
            for figure in fig:
                newFigure = scale_and_transl(figure, rRatio, dx, dy, "line", newCircO)
                figList.append(newFigure)
            newInvs.append(figList)

        elif type(fig) == myLine:  # if one of A/B/C had inverted to a line
            newFig = scale_and_transl(fig, rRatio, dx, dy, "line", newCircO)
            newInvs.append(newFig)

        else:   # if it isn't looking at the list of lines for circD
            print("Calling scale_and_transl w/ circle figtype on " + str(fig) + " with index " + str(invCircles.index(fig)))
            newFig = scale_and_transl(fig, rRatio, dx, dy, "circle", newCircO)
            newInvs.append(newFig)

    return newABCDs, newInvs, newCircO

def r_n(n):
    '''Returns the radius of the nth Ford circle tangent to one of the top/bottom lines'''
    return 1/n**2


def find_nth_cent(n,hloc,vloc):
    '''Finds the center of the Ford circle of the nth layer tangent to one of the lines
    hloc = -1 or 1, specifies what x direction the layer goes in
    vloc = -1 or 1, specifies if the nth circle is on the top or bottom
    Algorithm derived on pg. 49 of research records'''

    # Just in case
    if (vloc != 1 and vloc != -1) or (hloc != 1 and hloc != -1):
        raise ValueError("You entered vloc or hloc wrong in find_nth_cent(). vloc = " + str(vloc) + ", hloc = " + str(hloc))

    if n == 1:   # for the first case
        return (0,0)

    # x coord - calculation doesn't change with different vlocs
    xn = 0
    for k in range(2,n+1):   # the summation derived on pg. 49 of records
        xn += math.sqrt((r_n(k) + r_n(k-1))**2 - (r_n(k-1) - r_n(k))**2)
    xn *= hloc

    # y coord - depends on the vloc
    if vloc == 1:
        yn = 1 - r_n(n)
    else:
        yn = r_n(n) - 1

    return (xn,yn)


def find_new_pos_circ(circsABCD,invCircles,circO):
    '''Returns a new list of circsABCD and a new circO to adjust to the default Ford Circle
    config created in the program given the invCircles entered;
    Assumes this is a circular ABC config case, in which rd != inf'''
    print("\n\nRunning find_new_pos_circ()...\n\n") # for tests
    print("circsABCD: [",end="")
    for fig in circsABCD:
        if type(fig) == list:
            for figure in fig:
                print(str(figure) + ", ",end="")
        else:
            print(str(fig) + ", ",end="")
    print("]\n")
    print("invCircles: [",end="")
    for fig in invCircles:
        if type(fig) == list:
            for figure in fig:
                print(str(figure) + ", ",end="")
        else:
            print(str(fig) + ", ",end="")
    print("]")

    newABCDs = []
    newInvs = []

    # First find the two smallest tangent inverted circles; doesn't calc circD because inverted circD = line
    print("Finding two smallest tangent inverted circles...")
    
    maxLoc = 0
    maxR = 0
    possibleLocs = [0,1,2]   # not including the loc for inversion of circD

    # Finding max radius out of A,B,C and omitting it to find the two smallest
    print("Finding max radius out of A,B,C and omitting to find the two smallest...")
    for i in range(len(invCircles)-1):  # the -1 is to not include circD's inversion
        if type(invCircles[i]) == myLine:  # if the inverted result is a line
            maxLoc = i
            break
        elif invCircles[i].get_radius() > maxR:
            maxR = invCircles[i].get_radius()
            maxLoc = i

    smallestLocs = []
    for loc in possibleLocs:   # no need to worry about 3 radii being the same bc this is the non-straight config case
        if loc != maxLoc:
            smallestLocs.append(loc)

    smallestR = min(invCircles[smallestLocs[0]].get_radius(), invCircles[smallestLocs[1]].get_radius())
    largerR = max(invCircles[smallestLocs[0]].get_radius(), invCircles[smallestLocs[1]].get_radius())
    print("\nsmallestR: " + str(smallestR))
    print("largerR: " + str(largerR) + "\n")
    rRatio = smallestR / largerR
    rRatioFrac = Fraction(rRatio).limit_denominator(10000000)
    print("\nrRatioFrac = " + str(rRatioFrac))

    # These names may be misleading - smallestN doesn't refer to the layer number of the smaller layer number but to the smaller circle, which means a higher layer number
    print("Finding smallestN and largerN...")
    smallestN = math.sqrt(rRatioFrac.denominator)   # because when the fraction is in simplest terms it becomes a ratio of the larger's curvature to the smaller's curvature
    largerN = math.sqrt(rRatioFrac.numerator)
    print("smallestN: " + str(smallestN))
    print("largerN: " + str(largerN) + "\n")

    # Check in case this fraction thing doesn't work out
    print("Checking if smallestN and largerN work...")
    if not (smallestN.is_integer() and largerN.is_integer()):
        raise ValueError("One of the parts of the rRatioFrac isn't a square number! rRatioFrac = " + str(rRatioFrac))

    # Now start getting the centers - I'll just use the second smallest circles as the reference point here because that means
    # less iterations to calculate in find_nth_cent

    # Find the loc of the larger circ in invCircles
    print("Finding the loc of the larger circ in invCircles and checking for errors...")    
    largerLocs = []  # reason I'm making a list here is I wanna see if the code below returns more than 1 possible loc; if so, I'll raise an error
    smallerLocs = []
    for loc in smallestLocs:
        if invCircles[loc].get_radius() == largerR:
            largerLocs.append(loc)
        else:
            smallerLocs.append(loc)
    print("\nlargerLocs = " + str(largerLocs))
    print("smallerLocs = " + str(smallerLocs) + "\n")

    # Check for errors
    if len(largerLocs) != 1 and largerR != smallestR:  # account for if largerR = smallerR
        raise ValueError("Returned not only 1 val for largerLocs; largerLocs = " + str(largerLocs))
    elif len(smallerLocs) != 1 and largerR != smallestR:
        raise ValueError("Returned not only 1 val for smallerLocs; smallerLocs = " + str(smallerLocs))
    elif largerR == smallestR:
        largerLoc = largerLocs[0]
        smallestLoc = largerLocs[1]
    else:
        largerLoc = largerLocs[0]
        smallestLoc = smallerLocs[0]
    print("\nsmallestCirc: " + str(invCircles[smallestLoc]))
    print("largerCirc: " + str(invCircles[largerLoc]) + "\n")
        
    # Get the positions
    print("Getting the current positions...")
    currCent = invCircles[largerLoc].get_center()
    currR = largerR
    print("currCent: " + str(currCent))
    print("currR: " + str(currR) + "\n")

    # Find hloc and vloc
    print("Finding the hloc and vloc...")
    if invCircles[smallestLoc].get_center()[0] > currCent[0]:
        hloc = 1
    else:
        hloc = -1
    if invCircles[smallestLoc].get_center()[1] > currCent[1]:
        vloc = 1
    else:
        vloc = -1
    print("hloc = " + str(hloc) + ", vloc = " + str(vloc) + "\n")

    # Find target positions
    print("Finding the target positions...")
    targetCent = find_nth_cent(int(largerN),hloc,vloc)
    targetR = r_n(int(largerN))
    print("targetCent = " + str(targetCent))
    print("targetR = " + str(targetR) + "\n")

    # Find differences
    print("Finding differences...")
    ratio = targetR / currR
    dx = targetCent[0]/ratio - currCent[0]  # calculations derived on pg. 53 of records
    dy = targetCent[1]/ratio - currCent[1]
    print("ratio = " + str(ratio))
    print("dx = " + str(dx))
    print("dy = " + str(dy) + "\n")

    # Scale and translate

    # First make the new circO to redefine all the circles towards (as the new circle of inversion)
    print("\nScaling and translating...\n")
    print("Finding newCircO...")
    newCircO = scale_and_transl(circO,ratio,dx,dy,"circle")   # it has no invCircle because it IS the invCircle

    print("Finding new noninverted ABCDs...")
    for fig in circsABCD:  # for the non-inverted circles
        newFig = scale_and_transl(fig, ratio, dx, dy, "circle", newCircO)
        newABCDs.append(newFig)
        print("Found a new noninverted A/B/C/D.")

    print("Finding new inverted ABCDs...")
    for i in range(len(invCircles)):  # for the inverted circles
        fig = invCircles[i]
        if type(fig) == myLine:  # if one of A/B/C had inverted to a line
            newFig = scale_and_transl(fig, rRatio, dx, dy, "line", newCircO)
        elif i != len(invCircles) - 1:   # if we're not looking at the inversion of circle D here
            newFig = scale_and_transl(fig,ratio,dx,dy,"circle",newCircO)
            newInvs.append(newFig)
        else:   # if we are looking at the inversion of circle D
            newFig = scale_and_transl(fig,ratio,dx,dy,"line",newCircO)
        newInvs.append(newFig)

    return newABCDs, newInvs, newCircO

def find_new_pos(circsABCD,invCircles,circO):
    '''Returns a new list of circsABCD and a new circO to adjust to the default Ford Circle
    config created in the program given the invCircles entered'''

    if type(circsABCD[len(circsABCD)-1]) == list:   # if it is a straight config
        return find_new_pos_straight(circsABCD,invCircles,circO)
    else:
        return find_new_pos_circ(circsABCD,invCircles,circO)

def graph_new_pos(ra,rb,rc,circO,config="circle",graphInversions=False,returnCircO=False,graphCircO=True):
    '''Graphs a the new calculated positions of circles ABCD according to the default Ford Circle config;
    Graphing inverison is toggleable
    If config="straight" will run calculations for a straight config; else, runs calculations for a circular one'''
    print("\n\nRunning first_four()...\n\n")
    listABCD, invABCDs, circO = first_four(ra,rb,rc,circO,config,False,True)  # first_four(ra,rb,rc,circO,config=None,graph=True,returnLists=False)
    print("\n\nRunning find_new_pos()...\n\n")
    newABCDs,newInvs,newCircO = find_new_pos(listABCD,invABCDs,circO)  # detects the configuration inside this func
    print("\n\nDone running find__new_pos.\n\n")

    # Setting colors
    print("\n\nSetting colors...\n\n")
    newCircO.set_color("black")
    print("newCircO col: " + str(newCircO.get_color()))
    newABCDs[0].set_color("purple")
    newABCDs[2].set_color("green")
    print("CircA col: " + str(newABCDs[0].get_color()))
    print("CircB col: " + str(newABCDs[1].get_color()))
    print("CircC col: " + str(newABCDs[2].get_color()))
    if config == "straight":
        for fig in newABCDs[3]:  # for each circ D line
            fig.set_color("gray")
            print("CircD Line col: " + str(fig.get_color()))
    else:
        newABCDs[3].set_color("gray")
        print("CircD col: " + str(newABCDs[2].get_color()))

    # Graphing
    if graphCircO:
        newCircO.plot()
    if graphInversions:
        print("\n\nGraphing new pos for original and inversions...\n\n")
        for fig in newABCDs:
            if type(fig) == list:   # if it's a straight config and circD's a list of lines
                for figure in fig:
                    figure.plot_all()
            else:
                fig.plot_all()
    else:
        print("\n\nGraphing new pos for originals...\n\n")
        for fig in newABCDs:
            if type(fig) == list:   # if it's a straight config and circD's a list of lines
                for figure in fig:
                    figure.plot()
            else:
                fig.plot()

    if returnCircO:
        return newCircO


# Inverting the Ford Circles

def invert_future_layers(currDict,n,graphFord=True):
    '''Inverts and graphs the future layers of the Ford Circles'''

    for nextCirc in currDict.keys():
        if graphFord:
            nextCirc.plot_all()
        else:
            nextCirc.plot_inverse()
        if n != 1:   # if this isn't the last iteration
            invert_future_layers(currDict[nextCirc],n-1,graphFord)
        else:
            break


def generate_gasket(ra,rb,rc,circO,layers=5,config="circle",graphFord=True):
    '''Generates and graphs an Apollonian gasket with the specified initial radii;
    Function to graph Ford circle inversions also is toggleable'''
    
    newCircO = graph_new_pos(ra,rb,rc,circO,config,returnCircO=True,graphCircO=False)

    # Check for invalid user input
    if layers <= 2:
        raise ValueError("Layers must > 2.")
    elif not layers.is_integer():
        raise ValueError("Layers must be an integer value.")
    
    fordCircs = create_Ford_Circles(newCircO,layers-2,graph=False,returnList=True)
    print("\n\nFord circs list: " + str(fordCircs))

    # Layer 0
    layer0dict = fordCircs[0][0]
    
    print("\n\nLayer 0: " + str(layer0dict))
    for line in layer0dict.keys():
        print(line + ": " + str(layer0dict[line]))
    
    #for line in layer0dict.values():

    # not graphing inversion to circle D anymore
    if graphFord:
        layer0dict["line1"].plot_all()
        layer0dict["line2"].plot()
    else:
        layer0dict["line1"].plot_inverse()

    # Layer 1
    layer1dict = fordCircs[1][0]
    for circ in layer1dict.values():
        if graphFord:
            circ.plot_all()
        else:
            circ.plot_inverse()

    # Layer 2 and Beyond
    layer2dicts = fordCircs[2]
    up2dict = layer2dicts[0]
    down2dict = layer2dicts[1]

    for direcDict in [up2dict,down2dict]:
        for circ in direcDict.keys():  # graph layer 2 circles
            if graphFord:
                circ.plot_all()
            else:
                circ.plot_inverse()
            invert_future_layers(direcDict[circ],layers-2,graphFord)  # call the recursive function
        

# Past Apollonian (the triangle/regular apollonian functions are under Graphing Fractals above)

def graph_three_circle_apollonian():
    '''Graphs a default 3 circle apollonian gasket'''
    # Graphs circles O and D
    circleO = myCircle((0,0),2,None,"black")
    circleO.plot()
    circleD = myCircle((0,1),1,circleO)
    circleD.plot_all()
    

# Create and graph the circle of inversion
##defaultInv = myCircle((0,0),2)
##defaultInv.plot()
##inv1 = myCircle((0,1),1, None, "black")
##inv1.plot()
gasketInv = myCircle((0,-1),2,None,"black")

# Ford Circle Set
#create_Ford_Circles(gasketInv,3)

# Testing first four circles of gasket

# Debugging tests for Test #1
##newCirc = scale_and_transl(myCircle((0,-6)),0.5,0,3)
##print("newCirc center = " + str(newCirc.get_center()))
##print("newCirc radius = " + str(newCirc.get_radius()))
##testCirc = myCircle((0.48,0.26),0.932/2,gasketInv)
##print("\nInverted testCirc = " + str(testCirc.get_inv()) + "\n\n\n")

# Modified Test #1
##r = 1/(1 + 2 * math.sqrt(3) / 3)
##first_four(r, r, r, gasketInv,graph=False)
##graph_new_pos(r,r,r,gasketInv,graphInversions=True)
##generate_gasket(r,r,r,gasketInv,graphFord=False)


### Modified Test Case #2
##r = 1
##first_four(r,r,r,gasketInv,"straight",graph=False)
##graph_new_pos(r,r,r,gasketInv,"straight",graphInversions=True)
##generate_gasket(r,r,r,gasketInv,config="straight",graphFord=False,graphCircO=False)


### Determining test case #5
##circP = myCircle((-32,0),32)
##circQ = myCircle((32,0),32)
##circR = myCircle((0,-56.58),33)
##(hd,kd) = find_d_center(circP,circQ,circR,69.6811)
##print("Center of D for test case #5: " + str((hd,kd)))

### Modified Test Case #515
##first_four(32,32,33,gasketInv)
##graph_new_pos(32,32,33,gasketInv,graphInversions=True)
##generate_gasket(32,32,33,gasketInv,graphFord=False)


# Modified Test Case #6
##first_four(2,1,3.1415926,gasketInv)
##graph_new_pos(2,1,3.2425926,gasketInv,graphInversions=True)
##generate_gasket(2,1,3.1415926,gasketInv,graphFord=False)

### Test #7
##first_four(1/10,1/15,1/19,gasketInv)
##graph_new_pos(1/10,1/15,1/19,gasketInv,graphInversions=True)
generate_gasket(1/10,1/15,1/19,gasketInv,layers=7,graphFord=False)

### Test #8
##first_four(1/18,1/23,1/27,gasketInv)
##graph_new_pos(1/18,1/23,1/27,gasketInv,graphInversions=True)

# Define the example point
##pt = (3, 0)

# # Graph the center and the example points for center of inversion = (0,0)
##plt.plot(0, 0, "o")
##plt.text(0, 0, "O", fontsize=12, ha="right")
##plot_invert_pts("P", pt, defaultInv)
##print("P' = " + str(invert_pt(pt,defaultInv)))

# Using parametrics

# Plotting Apollonian Gasket
#graph_apollonian(inv1)
## graph_three_circle_apollonian()

# Plotting equilateral triangle

# Imperfect Triangle Apollonian
##for x in range(-21,20,2):
##    for y in range(2,43,2):
##        testTri = EquiTriangle((x,y),2,1)
##        testTri.plot_all()

# Triangle Apollonian
# graph_apollonian(inv1,"triangle")

# Format the graph
plt.grid()
plt.title("Circle Inversion")
plt.show()

