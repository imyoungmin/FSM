import numpy as np
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from multiprocessing import Pool
import time

"""
Implementation of the Fast Sweeping Method in a uniform grid and in a shared memory environment.
"""
class FSM:

	def __init__( self, gridPointsPerDim, minVal=-0.5, maxVal=+0.5 ):
		"""
		Constructor.
		:param gridPointsPerDim: Number of grid points per dimension.
		:param minVal: Minimum domain value.
		:param maxVal: Maximum domain value.
		"""
		self._D = 2					# Number of dimensions: supported 2 only.
		self._M = gridPointsPerDim	# Number of grid points per dimension.
		self._minVal = minVal
		self._maxVal = maxVal
		self._h = (self._maxVal - self._minVal) / (self._M - 1)  	# Domain spacing (assuming a 1:1 ratio).
		self._TOL = 1e-16			# Tolerance for convergence.

		# Initialize uniform grid with large initial values in its nodes.
		largeNumber = 10. * ( self._maxVal - self._minVal )
		self._U: np.ndarray = np.full( (self._M, self._M), largeNumber )

		# Preset the interface as a map of discrete coordinates to fixed \phi values (0 or close to it).
		self._Gamma: Dict[Tuple[int, int], float] = dict()


	def definePointAtOriginInterface( self ):
		"""
		Create an interface consisting of a single point at the origin.
		"""
		c = int( self._M / 2 )
		self._Gamma[(c, c)] = 0.0		# The interface: coordinates are discrete (i,j), i for x, j for y.
										# In the discrete grid, x are rows, and y are columns.
		for location in self._Gamma:
			self._U[location] = self._Gamma[location]  			# Fix known values for interface on grid.


	def _update( self, coord: Tuple[int, int] ) -> Tuple[Tuple[int, int], float]:
		"""
		Update the solution to the Eikonal equation value at U[i,j] using a Godunov finite difference scheme.
		:param coord: Tuple with the discrete (i,j) coordinates.
		:return New value at U(i,j).
		"""
		if self._Gamma.get( coord ) is not None:	# Do not modify the solution at (or adjacent to) the interface.
			return coord, self._Gamma.get( coord )

		i, j = coord[0], coord[1]

		# Finite difference along x.
		if i == 0:									# Left domain border?
			uh_xmin = self._U[1, j]
		elif i == self._M - 1:						# Right domain border?
			uh_xmin = self._U[self._M - 2, j]
		else:										# Interior point.
			uh_xmin = min( self._U[i-1, j], self._U[i+1, j] )

		# Finite difference along y.
		if j == 0:									# Bottom domain border?
			uh_ymin = self._U[i, 1]
		elif j == self._M - 1:						# Top domain border?
			uh_ymin = self._U[i, self._M - 2]
		else:										# Interior point.
			uh_ymin = min( self._U[i, j-1], self._U[i, j+1] )

		# Solving the Godunov finite difference equation.
		a, b = uh_xmin, uh_ymin
		if abs( a - b ) >= self._h:
			uBar = min( a, b ) + self._h
		else:
			uBar = ( a + b + np.sqrt( 2. * self._h ** 2 - ( a - b ) ** 2 ) ) / 2.

		# Update u_ij to be the smaller between computed u and u_ij^old.
		return coord, min( uBar, self._U[i, j] )


	def goSerial( self ):
		"""
		Execute serial version of FSM.
		"""
		print( "Serial Fast Sweeping Method began..." )
		rootTime = time.time()
		if not len( self._Gamma ):
			raise Exception( "Uninitialized interface!" )

		# Indices for traversing the 2D field.
		I: np.ndarray = np.array( range( self._M ) )
		J: np.ndarray = np.array( range( self._M ) )

		# Use the max of L1 norm to check for convergence.
		errorNorm = 1
		while errorNorm > self._TOL:
			U_old: np.ndarray = np.array( self._U )

			# 2^D Gauss-Seidel iterations.
			for ordering in range( 2 ** self._D ):
				startTime = time.time()
				print( "  Ordering", ordering, end="..." )
				for i in I:								# Along x-axis.
					for j in J:							# Along y-axis.
						_, u = self._update( (i, j) )
						self._U[i, j] = u

				print( " {} seconds".format( time.time() - startTime ) )

				# Efficient axis rotation by fliping coordinate indices.
				if not ( ordering % 2 ):
					I = I[::-1]
				else:
					J = J[::-1]

			# New L1 error norm.
			errorNorm = np.max( np.mean( np.abs( self._U - U_old ), axis=0 ) )

		print( "Done after {} seconds".format( time.time() - rootTime ) )


	def goParallel( self, processes: int=4 ):
		"""
		Execute parallel version of FSM.
		:param processes: Number of threads to spawn.
		"""
		print( "Parallel Fast Sweeping Method began..." )
		rootTime = time.time()
		if not len( self._Gamma ):
			raise Exception( "Uninitialized interface!")

		# Indices for traversing the 2D field.
		I: np.ndarray = np.array( range( self._M ) )
		J: np.ndarray = np.array( range( self._M ) )

		# Use the max of L1 norm to check for convergence.
		errorNorm = 1
		while errorNorm > self._TOL:
			U_old: np.ndarray = np.array( self._U )

			# 2^D Gauss-Sidel iterations.
			for ordering in range( 2 ** self._D ):
				startTime = time.time()
				print( "  Ordering", ordering, end="..." )
				for level in range( 2 * self._M - 1 ):		# 0 : I + J - 2.
					I1 = max( 0, level - self._M + 1 )		# Lower bound for discrete x coords.
					I2 = min( self._M - 1, level )			# Upper bound for discrete x coords (inclusive).

					params: List[Tuple[FSM, Tuple[int, int]]] = []	# A list of references to this class and tuples of discrete coords along the current level
																	# interleaved: [(self, (i0, j0)), (self, (i1, j1)), ...]
					for i in range( I1, I2 + 1 ):					# Gather the coords we'll process parallely.
						j = level - i
						params.append( (self, (I[i], J[j])) )		# Recall: accessing the "rotated axes" coordinates.

					pool = Pool( processes=processes )
					updates = pool.starmap( FSM._update, params )  	# Each coordinate pair in its own thread.
					pool.close()
					pool.join()

					for update in updates:							# Write data into discrete solution grid.
						self._U[update[0]] = update[1]

				print( " {} seconds".format( time.time() - startTime ) )

				# Efficient axis rotation by fliping coordinate indices.
				if not (ordering % 2):
					I = I[::-1]
				else:
					J = J[::-1]

			# New L1 error norm.
			errorNorm = np.max( np.mean( np.abs( self._U - U_old ), axis=0 ) )

		print( "Done after {} seconds".format( time.time() - rootTime ) )


	def plotSurface( self ):
		"""
		Plot the surface discretized in U.
		"""
		x = np.linspace( self._minVal, self._maxVal, self._M )				# Grid node discretizations along each cartesian direction.
		y = np.linspace( self._minVal, self._maxVal, self._M )
		X, Y = np.meshgrid( x, y )
		fig = plt.figure( dpi=150 )
		ax: Axes3D = fig.add_subplot( 111, projection="3d" )
		plt.title( r"Approximated distance function for $\phi(x,y)$" )
		surf = ax.plot_surface( X, Y, self._U.transpose(), cmap=cm.coolwarm, linewidth=0 )		# Note the transpose operation.
		ax.set_xlabel( r"$x$" )
		ax.set_ylabel( r"$y$" )
		ax.set_zlabel( r"$\phi(x,y)$" )
		fig.colorbar( surf, shrink=0.5, aspect=15 )
		plt.show()