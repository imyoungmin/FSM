import numpy as np
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import multiprocessing
from multiprocessing import shared_memory
import time
import abc

"""
Implementation of the Fast Sweeping Method in a uniform grid and in a shared memory environment.
"""
class _FSM( abc.ABC ):

	def __init__( self, gridPointsPerDim, minVal, maxVal ):
		"""
		Constructor.
		:param gridPointsPerDim: Number of grid points per dimension.
		:param minVal: Minimum domain value.
		:param maxVal: Maximum domain value.
		"""
		self._D = 2													# Number of dimensions: supported 2 only.
		self._M = gridPointsPerDim									# Number of grid points per dimension.
		self._minVal = minVal
		self._maxVal = maxVal
		self._h = (self._maxVal - self._minVal) / (self._M - 1)  	# Domain spacing (assuming a 1:1 ratio).
		self._TOL = 1e-16			# Tolerance for convergence.

		# Initialize uniform grid with large initial values in its nodes.
		self._largeNumber = 10. * ( self._maxVal - self._minVal )
		self._U: np.ndarray = np.full( (self._M, self._M), self._largeNumber )

		# Preset the interface as a map of discrete coordinates to fixed \phi values (0 or close to it).
		self._Gamma: Dict[Tuple[int, int], float] = dict()


	def definePointAtOriginInterface( self ):
		"""
		Create an interface consisting of a single point at the origin.
		"""
		self._Gamma = dict()			# Reset interface map.
		c = int( self._M / 2 )
		self._Gamma[(c, c)] = 0.0		# The interface: coordinates are discrete (i,j), i for x, j for y.
										# In the discrete grid, x are rows, and y are columns.


	@abc.abstractmethod
	def go( self ):
		"""
		Execute FSM.
		"""
		pass


	@staticmethod
	def _update( U: np.ndarray, M: int, h: float, coord: Tuple[int, int] ) -> Tuple[Tuple[int, int], float]:
		"""
		Compute the update value for solution to the Eikonal equation value at [i,j] using a Godunov finite difference scheme.
		:param U: Solution grid.
		:param M: Number of grid points along each Cartesian direction.
		:param h: Grid spacing.
		:param coord: Tuple with the discrete (i,j) coordinates.
		:return New value at U(i,j).
		"""
		i, j = coord[0], coord[1]

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
		return coord, min( uBar, U[i, j] )


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

################################################ Serial version of FSM #################################################

class Serial( _FSM ):
	"""
	Serial version of Fast Sweeping Method.
	"""

	def __init__( self, gridPointsPerDim, minVal=-0.5, maxVal=+0.5 ):
		"""
		Constructor.
		:param gridPointsPerDim: Number of grid points per dimension.
		:param minVal: Minimum domain value.
		:param maxVal: Maximum domain value.
		"""
		super( Serial, self ).__init__( gridPointsPerDim, minVal, maxVal )


	def go( self ):
		"""
		Execute serial version of FSM.
		"""
		if not len( self._Gamma ):
			raise Exception( "Uninitialized interface!" )

		print( "Serial Fast Sweeping Method began..." )
		rootTime = time.time()

		# Reset U.
		self._U: np.ndarray = np.full( (self._M, self._M), self._largeNumber )
		for location in self._Gamma:
			self._U[location] = self._Gamma[location]  			# Fix known values for interface on grid.

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
				for i in I:											# Along x-axis.
					for j in J:										# Along y-axis.
						if self._Gamma.get( (i, j) ) is not None:	# Do not modify the solution at (or adjacent to) the interface.
							continue
						_, u = self._update( self._U, self._M, self._h, (i, j) )
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


################################################ Parallel version of FSM ###############################################


class Parallel( _FSM ):
	"""
	Parallel version of Fast Sweeping Method on a uniform grid in a shared-memory environment.
	Based on the model at https://docs.python.org/3/library/multiprocessing.shared_memory.html
	"""

	def __init__( self, gridPointsPerDim, minVal=-0.5, maxVal=+0.5 ):
		"""
		Constructor.
		:param gridPointsPerDim: Number of grid points per dimension.
		:param minVal: Minimum domain value.
		:param maxVal: Maximum domain value.
		"""
		super( Parallel, self ).__init__( gridPointsPerDim, minVal, maxVal )
		self._shm = None


	def go( self, processes: int=4 ):
		"""
		Execute parallel version of FSM.
		:param processes: Number of threads to spawn.
		"""
		if processes <= 1 or processes > multiprocessing.cpu_count() - 2:		# Invalid number of requested processes.
			raise Exception( "Invalid number of processes.  Choose between 2 and", ( multiprocessing.cpu_count() - 2 ) )

		if not len( self._Gamma ):												# No interface?
			raise Exception( "Uninitialized interface!")

		print( "Parallel Fast Sweeping Method began..." )
		rootTime = time.time()

		# Create and initialize a shared memory block.
		U = np.full( (self._M, self._M), self._largeNumber )
		for location in self._Gamma:
			U[location] = self._Gamma[location]  			# Fix known values for interface on grid.

		self._shm = shared_memory.SharedMemory( create=True, size=self._U.nbytes )
		self._U = np.ndarray( (self._M, self._M), dtype=np.float, buffer=self._shm.buf )  	# An array backed by shared memory.
		self._U[::] = U[::]

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

					coords: List[Tuple[int, int]] = []		# A list of grid point coordinates to update along current level.
					for i in range( I1, I2 + 1 ):			# Gather the coords we'll (probably) process parallely.
						j = level - i
						coord = (I[i], J[j])						# Recall: accessing the "rotated axes" coordinates.

						if self._Gamma.get( coord ) is None:		# Skip interface/fixed nodes.
							coords.append( coord )

					self._loadBalancing( coords, processes )

				print( " {} seconds".format( time.time() - startTime ) )

				# Efficient axis rotation by fliping coordinate indices.
				if not (ordering % 2):
					I = I[::-1]
				else:
					J = J[::-1]

			# New L1 error norm.
			errorNorm = np.max( np.mean( np.abs( self._U - U_old ), axis=0 ) )

		# Free and release the shared memory block at the very end.
		#self._shm.close()
		self._shm.unlink()
		print( "Done after {} seconds".format( time.time() - rootTime ) )


	def _loadBalancing( self, coords: List[Tuple[int, int]], processes: int ):
		"""
		Compute new values for solution by load-balancing calculations among processes (if necessary).
		:param coords: A list of discrete coordinates to update.
		:param processes: A valid number of processes to consider.
		"""
		# Determine if we need to use multiprocessing at all.
		nodeCount = len( coords )
		if not len( coords ):
			return

		if nodeCount < 3 * processes:		# If there are less than 3 nodes per processor proceed serially.
			for coord in coords:
				_, u = self._update( self._U, self._M, self._h, coord )
				self._U[coord] = u
		else:								# Share load among requested processes.
			requests: List[Tuple[str, int, float, List[Tuple[int, int]]]] = []		# List of the form [(uName, M, h, [(i0,j0),(i1,j1),...]), (uName, M, h, [(i0,j0),(i1,j1),...]), ...]
			points: List[Tuple[int, int]] = []
			nodesPerProcess = nodeCount // processes
			includingInLastGroup = nodesPerProcess * processes	# Don't create a new group when reaching this node index in the list of coords.
			i = 0
			while i < len( coords ):
				points.append( coords[i] )
				i += 1
				if i % nodesPerProcess == 0 and i != includingInLastGroup:			# Create a new group, except in the last subset of coords.
					requests.append( ( self._shm.name, self._M, self._h, points ) )
					points = []

			requests.append( ( self._shm.name, self._M, self._h, points ) )			# Don't forget to append last group.

			# Process each group of coordinates in its own thread.
			pool = multiprocessing.Pool( processes=processes )
			updateChunks: List[List[Tuple[Tuple[int, int], float]]] = pool.starmap( Parallel._processRequests, requests )
			pool.close()
			pool.join()

			# Process results.
			for updateList in updateChunks:  					# Write data into discrete solution grid.
				for updateTuple in updateList:
					self._U[updateTuple[0]] = updateTuple[1]


	@staticmethod
	def _processRequests( shmName: str, M: int, h: float, coords: List[Tuple[int, int]] ) -> List[Tuple[Tuple[int, int], float]]:
		"""
		Process requests in parallel.
		:param shmName: Name of shared memory block.
		:param M: Number of nodes along each cartesian direction.
		:param h: Grid spacing.
		:param coords: List of coordinates.
		:return: A list of update tuples: [((i0,j0), u_new), ...].
		"""
		result: List[Tuple[Tuple[int, int], float]] = []
		shm = shared_memory.SharedMemory( name=shmName )						# Attach to the existing shared memory block.
		U: np.ndarray = np.ndarray( (M, M), dtype=np.float, buffer=shm.buf )	# Access shared memory.

		for coord in coords:
			result.append( Parallel._update( U, M, h, coord ) )

		shm.close()			# Cleaning up.

		return result