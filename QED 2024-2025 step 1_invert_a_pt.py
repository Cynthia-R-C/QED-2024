# Version 9/5/2024
# Simplest beginning, figured out how to invert a point


import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

xpts = np.array([0,6])
ypts = np.array([0,250])

# Create a figure and axis
fig, ax = plt.subplots()

def invert_pt(coords, r):
  '''Returns a tuple of coordinates of the inverted point'''
  (x,y) = coords
  xNew = r**2 * x / (x**2 + y**2)
  yNew = r**2 * y / (x**2 + y**2)
  return (xNew, yNew)

# Define the center and radius of the circle
center = (0, 0)
radius = 1

# Graph the circle
circle = Circle(center, radius, edgecolor='blue', facecolor='none')
ax.add_patch(circle)
ax.set_xlim(-radius-1, radius+1)
ax.set_ylim(-radius-1, radius+1)
ax.set_aspect('equal')

# Define the points
pt = (3.4,-2.1)
invPt = invert_pt(pt,radius)

# Graph the points
plt.plot(0,0,"o")
plt.text(0,0,"O",fontsize=12,ha="right")
plt.plot(pt[0],pt[1],"o")
plt.text(pt[0],pt[1],"P",fontsize=12,ha="right")
plt.plot(invPt[0],invPt[1],"o")
plt.text(invPt[0],invPt[1],"P'",fontsize=12,ha="right")

# Format the graph
plt.grid()
plt.title("Circle Inversion")
plt.show()
