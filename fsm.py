# Implementing the parallel version of the Fast Sweeping Method in a uniform grid and in shared memory environment.
import numpy as np
from typing import Dict, Tuple

# Initialization.
D = 2											# Number of dimensions.
M = 11											# Number of grid points per dimension.
U_old: np.ndarray = np.zeros( (M, M) )
U_new: np.ndarray = np.full( (M, M), np.finfo(float).max )	# Uniform grid with large initial values in its nodes.
Gamma: Dict[Tuple[int, int], float] = dict()
Gamma[(5, 5)] = 0.0								# The interface: coordinates are discrete (i,j), i for x, j for y.
												# Then, the grid, U, x are rows, and y are columns.
h = 1. / ( M - 1 )								# Domain spacing (assuming a 1:1 ratio).

for location in Gamma:
	U_new[location] = Gamma[location]  			# Fixed known values for interface on grid.

def update( U: np.ndarray, i: int, j: int ):
	"""
	Update the solution to the Eikonal equation value at U[i,j] using a Godunov finite difference scheme.
	:param U: Discretized solution.
	:param i: Index in x direction.
	:param j: Index in y direction.
	"""
	if Gamma.get( (i, j) ) is not None:			# Do not modify the solution at (or adjacent to) the interface.
		return

	# Finite difference along x.
	if i == 0:									# Left domain border?
		uh_xmin = U[1, j]
	elif i == M - 1:							# Right domain border?
		uh_xmin = U[M - 2, j]
	else:										# Interior point.
		uh_xmin = min( U[i-1, j], U[i+1, j] )

	# Finite difference along y.
	if j == 0:									# Bottom domain border?
		uh_ymin = U[i, 1]
	elif j == M - 1:							# Top domain border?
		uh_ymin = U[i, M - 2]
	else:										# Interior point.
		uh_ymin = min( U[i, j-1], U[i, j+1] )

	# Solving the Godunov finite difference equation.
	a, b = uh_xmin, uh_ymin
	if abs( a - b ) >= h:
		uBar = min( a, b ) + h
	else:
		uBar = ( a + b + np.sqrt( 2. * h ** 2 - ( a - b ) ** 2 ) ) / 2.

	# Update u_ij to be the smaller between computed u and u_ij^old.
	U[i, j] = min( uBar, U[i, j] )


# Indices for traversing the field.
I: np.ndarray = np.array( range( M ) )
J: np.ndarray = np.array( range( M ) )

# Use the max of L1 norm to check for convergence.
errorNorm = 1
TOL = 1e-16
while errorNorm > TOL:
	U_old = np.array( U_new )

	# 2^D Gauss-Seidel iterations.
	for ordering in range( 2 ** D ):
		for i in I:								# Along x-axis.
			for j in J:							# Along y-axis.
				update( U_new, i, j )

		# Efficient axis rotation by fliping coordinate indices.
		if not ( ordering % 2 ):
			I = I[::-1]
		else:
			J = J[::-1]

	# New L1 error norm.
	errorNorm = np.max( np.mean( np.abs( U_new - U_old ), axis=0 ) )