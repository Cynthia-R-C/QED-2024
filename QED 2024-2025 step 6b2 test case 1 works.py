# Version 10/27/2024
# Added cmath
# Deleted some old pre-sympy calculation code
# Created separate function out of find_circD: rd_works()
# Added is_between()
# Added is_within_tolerance()
# Changed tolerance to 2.2e-9

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math
import cmath
import numpy as np
from sympy.solvers import solve
from sympy import Symbol
from sympy.abc import x, y

# Create a figure and axis
fig, ax = plt.subplots()

# Define limits
xlims = [-20,20]    # when plotting an infinite graph like a line, this defines how far in the x axis both ways the graph will stretch'
tValues = np.linspace(0,7,500)  # for getting tons of close points to graph parametrics
tDist = 7

# Define tolerance value bc of rounding errors
tolerance = 2.2e-9


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
            x = h - 10*invR - invR/2
            y += invR / 2 * math.sqrt(3)


# Generalizing Apollonian Gaskets

    # Ford Circles

def find_third_radius(ra,rb):
    '''Returns the radius of a smaller circle tangent to circle A, circle B, and a horizontal line tangent to both circles'''
    k = 1/math.sqrt(ra) + 1/math.sqrt(rb)    # used existing equations
    rc = (1/k)**2
    return rc

def find_third_center(centA,ra,rc):
    '''Returns the center of the third smaller circle with same setting as previous function'''
    (ha,ka) = centA
    hc = ha + 2 * math.sqrt(rc*ra)
    kc = ka - ra + rc
    return (hc,kc)

def find_third_circ(circA, circB):
    '''Returns the center and radius of a smaller circle tangent to circle A, circle B, and a horizontal line tangent to both circles'''
    ra = circA.get_radius()
    (ha, ka) = circA.get_center()
    rb = circB.get_radius()
    rc = find_third_radius(ra,rb)
    (hc,kc) = find_third_center((ha,ka),ra,rc)
    return (hc,kc),rc


    # First 4 circles

def is_collinear(pt1, pt2, pt3):
    '''Returns a boolean of if the 3 points are collinear'''
    return (pt1[0] == pt2[0] and pt2[0] == pt3[0]) or (pt1[1]-pt2[1])/(pt1[0]-pt2[0]) == (pt2[1]-pt3[1])/(pt2[0]-pt3[0])

def calc_rd(ra,rb,rc):
    '''Returns list of solutions to solving for rd'''

    # Trying it with sympy
    rd = Symbol('rd')
    solutions = solve((1/ra + 1/rb + 1/rc - 1/rd)**2 - 2 * (1/ra**2 + 1/rb**2 + 1/rc**2 + 1/rd**2),rd)
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

    # Now plug it into this abomination of an equation from Wolfram
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

##    # Let these be the curvatures of circles A, B, and C
##    ca = 1/ra
##    cb = 1/rb
##    cc = 1/rc
##    cd = 1/rd
##
##    # Using Descartes' complex number equation for center
##    za = complex(ha,ka)
##    zb = complex(hb,kb)
##    zc = complex(hc,kc)
##
##    # A list of solutions to account for the plus or minus?
##    zds = [(za*ca + zb*cb + zc*cc + 2*cmath.sqrt(ca*cb*za*zb + cb*cc*zb*zc + ca*cc*za*zc))/cd, (za*ca + zb*cb + zc*cc - 2*cmath.sqrt(ca*cb*za*zb + cb*cc*zb*zc + ca*cc*za*zc))/cd]
##
##    solutions = []
##    for zd in zds:
##        print(zd)
##        solutions.append((zd.real,zd.imag))
##
##    return solutions

    # Variables for Cramer's Rule
    a = ha - hb
    b = ka - kb
    u = 1/2 * (rb**2 - ra**2 + 2 * rd * (rb - ra) + hb**2 + kb**2 - ha**2 - ka**2)
    c = ha - hc
    d = ka - kc
    v = 1/2 * (rc**2 - ra**2 + 2 * rd * (rc - ra) + hc**2 + kc**2 - ha**2 - ka**2)

    # Using Cramer's Rule
    hd = (d*u - b*v) / (a*d - b*c)
    kd = (a*v - c*u) / (a*d - b*c)

    return (hd,kd)

def is_between(p, p1, p2):
    '''Returns a boolean for if p is between points p1 and p2 (points are tuples)'''
    xBetween = (p1[0] <= p[0] and p[0] <= p2[0]) or (p2[0] <= p[0] and p[0] <= p1[0])
    yBetween = (p1[1] <= p[1] and p[1] <= p2[1]) or (p2[1] <= p[1] and p[1] <= p1[1])
    return xBetween and yBetween

def is_within_tolerance(p1,p2):
    '''Returns if the two points should be counted as the same point'''
    return math.dist(p1,p2) < tolerance

def rd_works(circA, circB, circC, rd):
    '''Returns a boolean for if a certain rd works'''
    (hd,kd) = find_d_center(circA, circB, circC, rd)
    print("For rd = " + str(rd) + ", the center is (" + str(hd) + ", " + str(kd) + ")")

##        # Adjustment for testing Descartes' complex ver
##        solutions = find_d_center(circA, circB, circC, rd)
##        for cent in solutions:
##            print("For rd = " + str(rd) + ", the center is (" + str(cent[0]) + ", " + str(cent[1]) + ")")

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
##            print("hd,kd: " + str((hd,kd)))
##            print("rd: " + str(rd))
##            print("(h,k): " + str((h,k)))
##            print("r: " + str(r))
##            circLetter = ""
##            if counter == 1:
##                circLetter = "A"
##            elif counter == 2:
##                circLetter = "B"
##            elif counter == 3:
##                circLetter = "C"
##            else:
##                circLetter = "somehow counter > 3"
##            print("circLetter: " + circLetter)
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

    for rd in rds:    # tests if each rd works: if it doesn't, it's removed from the rd possibilites
        if not rd_works(circA, circB, circC, rd):
            rds.remove(rd)

##        # Check to see if rd is valid here - plug back in to find intersections
##        interAs = solve([(x - ha)**2 + (y - ka)**2 - ra**2, (x - hd)**2 + (y - kd)**2 - rd**2], [x, y], dict=True)  # solutions for intersection of circles A and D
##        interBs = solve([(x - hb)**2 + (y - kb)**2 - rb**2, (x - hd)**2 + (y - kd)**2 - rd**2], [x, y], dict=True)  # solutions for intersection of circles B and D
##        interCs = solve([(x - hc)**2 + (y - kc)**2 - rc**2, (x - hd)**2 + (y - kd)**2 - rd**2], [x, y], dict=True)  # solutions for intersection of circles C and D
##        
##        # Testing
##        print("interAs = " + str(interAs))
##        print("interBs = " + str(interBs))
##        print("interCs = " + str(interCs))
##
##        # Using sympy solve results as estimates in nsolve to account for rounding errors
##        newInterAs = []
##        newInterBs = []
##        newInterCs = []
##        
##        if len(interAs) != 0:
##            for solution in newInterAs:
##                solution[x] = x1
##                solution[y] = y1
##                newInterAs += nsolve([(x - ha)**2 + (y - ka)**2 - ra**2, (x - hd)**2 + (y - kd)**2 - rd**2], (x1, y1), dict=True)
##
##        if len(interBs) != 0:
##            for solution in newInterBs:
##                solution[x] = x1
##                solution[y] = y1
##                newInterBs += nsolve([(x - hb)**2 + (y - kb)**2 - rb**2, (x - hd)**2 + (y - kd)**2 - rd**2], (x1, y1), dict=True)
##
##        if len(interCs) != 0:
##            for solution in newInterCs:
##                solution[x] = x1
##                solution[y] = y1
##                newInterAs += nsolve([(x - hc)**2 + (y - kc)**2 - rc**2, (x - hd)**2 + (y - kd)**2 - rd**2], (x1, y1), dict=True)
##
##        # Testing
##        print("newInterAs = " + str(newInterAs))
##        print("newInterBs = " + str(newInterBs))
##        print("newInterCs = " + str(newInterCs))
##
##
##        counter = 0     # keeps track of which circle the bottom is referring to - made for testing
##        for solutions in [newInterAs,newInterBs,newInterCs]:
##
##            # for testing
##            counter += 1
##            circ = ""
##            if counter == 1:
##                circ = "A"
##            elif counter == 2:
##                circ = "B"
##            else:
##                circ = "C"
##            print("Intersections of rd = " + str(rd) + " with circle " + circ + " are " + str(solutions))
##            # end testing part
##             
##            if len(solutions) != 1:       # the sympy returns a list of dictionaries with one x,y solution in each
##                rds.remove(rd)
##                break

    if len(rds)!= 1:     # if there isn't only one possibility for rd left
        print("There isn't only 1 possibility for rd: the possibilities are " + str(rds))   # for testing
        return None, None
    else:
        rd = rds[0]
        (hd,kd) = find_d_center(circA, circB, circC, rd)
    
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
    m = (kb-ka)/(hb-ha)
    solutions = solve([y/x + 1/m, math.sqrt(x**2 + y**2) - ra], [x,y], dict=True)   # remember, this solves for the CHANGE in x and y from the center of A to the tangent points
    return solutions,m

def graph_first_four(circA, circB, circC, circO):
    '''Graphs the first four circles of the Apollonian gasket - assumes circle O is already graphed'''
    circA.plot_all()
    circB.plot_all()
    circC.plot_all()

    centD,rd = find_circD(circA, circB, circC)
    (ha,ka) = circA.get_center()

    # Add check for circD to see if it returns 'inf','inf' or None, None
    if centD == None:   # since it's both or neither I really only have to check one of them
        #raise ValueError("The three initial circles for the gasket are invalid.")   # might change this later when I start the user input to have the user reenter the circles

        # commented out above portion for testing
        pass
        
    elif centD == 'inf':
        solutions, m = find_tangent_lines(circA, circB, circC)
        for solution in solutions:
            #  Each solution in the list is a dictionary
            dx = solution[x]   # delta x = change in x
            dy = solution[y]   # delta y = change in y
            xl = ha + dx       # the l is an L not a one; is the coordinates of one of the two tangent points (top and bottom)
            yl = ka + dy
            plt.plot.axline((xl,yl),None,m,color="red")
    else:
        circD = myCircle(centD,rd,circO)
        circD.plot_all()
    

# Past Apollonian (the triangle/regular apollonian functions are under Graphing Fractals above)

def graph_three_circle_apollonian():
    '''Graphs a default 3 circle apollonian gasket'''
    # Graphs circles O and D
    circleO = myCircle((0,0),2,None,"black")
    circleO.plot()
    circleD = myCircle((0,1),1,circleO)
    circleD.plot_all()
    


# Create and graph the circle of inversion
#defaultInv = myCircle((0,0),2)
#defaultInv.plot()
#inv1 = myCircle((0,1),1, None, "black")
#inv1.plot()
gasketInv = myCircle((0,-1),2,None,"black")

# Testing first four circles of gasket

# Test #1
r = 1/(1 + 2 * math.sqrt(3) / 3)
circA = myCircle((0,2*r*math.sqrt(3)/3), r, gasketInv)
circB = myCircle((-1*r, -1 * r*math.sqrt(3)/3), r, gasketInv)
circC = myCircle((r, -1 * r*math.sqrt(3)/3), r, gasketInv)
graph_first_four(circA, circB, circC, gasketInv)

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
