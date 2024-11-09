# Version 9/6/2024
# made a function to graph the point and the inverted point more easily, added a check in case we were trying to invert the radius of the circle of inversion


import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

xpts = np.array([0,6])
ypts = np.array([0,250])

# Create a figure and axis
fig, ax = plt.subplots()

# Define the center and radius of the circle
center = (0, 0)
radius = 1

# Graph the circle of inversion
circle = Circle(center, radius, edgecolor='blue', facecolor='none')
ax.add_patch(circle)
ax.set_xlim(-radius-1, radius+1)
ax.set_ylim(-radius-1, radius+1)
ax.set_aspect('equal')

# Define functions
def invert_pt(coords, r):
  '''Returns a tuple of coordinates of the inverted point'''
  if coords == (0,0):
      return None           # add a check if we're inverting the radius -> becomes inf
  (x,y) = coords
  xNew = r**2 * x / (x**2 + y**2)
  yNew = r**2 * y / (x**2 + y**2)
  return (xNew, yNew)

def plot_invert_pts(label, coords, r):
  '''Plots the point specified by coords and its inverted point with its specified label'''
  invCoords = invert_pt(coords,r)
  plt.plot(coords[0],coords[1],"o")
  plt.text(coords[0],coords[1],label,fontsize=12,ha="right")
  if invCoords != None:
      plt.plot(invCoords[0],invCoords[1],"o")
      plt.text(invCoords[0],invCoords[1],label+"'",fontsize=12,ha="right")

# Graph the circle
circle = Circle(center, radius, edgecolor='blue', facecolor='none')
ax.add_patch(circle)
ax.set_xlim(-radius-1, radius+1)
ax.set_ylim(-radius-1, radius+1)
ax.set_aspect('equal')

# Define the points
pt = (0,0)
invPt = invert_pt(pt,radius)

# Graph the points
plt.plot(0,0,"o")
plt.text(0,0,"O",fontsize=12,ha="right")
plot_invert_pts("P",pt,radius)

# Format the graph
plt.grid()
plt.title("Circle Inversion")
plt.show()
