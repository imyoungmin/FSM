"""
Numerical methods.
"""
import math
import numpy as np
from typing import List, Tuple


def minmod( alpha: float, beta: float ) -> float:
	"""
	The MinMod operator as defined in [4].
	:param alpha: First operand.
	:param beta: Second operand.
	:return: 0 or first or second operand.
	"""
	absAlpha = math.fabs( alpha )
	absBeta = math.fabs( beta )
	if alpha * beta > 0:
		if absAlpha <= absBeta:
			return alpha
		else:
			return beta
	else:
		return 0


def D0_dd( phi: List[float], s: List[float] ) -> float:
	"""
	Compute the central difference approximation of the second order derivative of phi along the same dimension, as in [3].
	:param phi: List of the 3 upwind-ordered phi values (i.e. [\phi_{i-1}, \phi_i, \phi_{i+1}]).
	:param s: List of the 2 distances along the d dimension between the upwind-ordered phi values (i.e. [s_{i-1}, s_{i+1}]).
	:return: D^0_{dd}\phi_i.
	"""
	return 2.0 / sum( s[:2] ) * ( ( phi[2] - phi[1] ) / s[1] - ( phi[1] - phi[0] ) / s[0] )


def approximateNodeDistanceToInterface( phi: List[float], s: List[float], direction: bool, eps: float=1e-6 ) -> float:
	"""
	Approximate the distance to the interface from a reference vertex, having the latter as part of a segment (horizontal
	or vertical) currently being cut out by the interface, as in [3].  For example:
	                                       |
		i-2          i-1           i       |   i+1          i+2
		 o------------o------------o- s_I -x----o------------o
		                                   |
		                                   | <-- Interface
	:param phi: A 5-element list of level-set function values at grid nodes, ordered in the upwinding direction (i.e. [\phi_{i-2}, \phi_{i-1}, \phi_i, \phi_{i+1}, \phi_{i+2}]).
	:param s: A 4-element list of distances between reference point and other grid nodes in phi (i.e. [s_{i-2}, s_{i-1}, s_{i+1}, s_{i+2}]).
	:param direction: True if interface lies between i and {i+1}, false if interface lies between {i-1} and i.
	:param eps: A small number to avoid division by zero.
	:return: Distance between center point and the interface (lying in cut-out segment).
	"""
	assert len( phi ) == 5
	assert len( s ) == 4

	# Determining the coefficients for \phi^0(x) = c_2 x^2 + c_1 x + c_0.
	if direction:					# Right (up) direction: Interface cuts the segment between i and {i+1}.
		assert phi[2] * phi[3] < 0	# Verify the interface is indeed cutting the line between i and {i+1}.

		c2 = 0.5 * minmod( D0_dd( phi[1:4], s[1:3] ), D0_dd( phi[2:5], s[2:4] ) )
		c1 = ( phi[3] - phi[2] ) / s[2]
		c0 = ( phi[3] + phi[2] ) / 2 - c2 * ( s[2] ** 2 ) / 4

		sI = s[2] / 2
	else:							# Down (left) direction: Interface cuts the segment between i and {i+1}.
		assert phi[1] * phi[2] < 0	# Verify the interface is indeed cutting the line between {i-1} and i.

		c2 = 0.5 * minmod( D0_dd( phi[0:3], s[0:2] ), D0_dd( phi[1:4], s[1:3] ) )
		c1 = (phi[1] - phi[2]) / s[1]
		c0 = (phi[1] + phi[2]) / 2 - c2 * (s[1] ** 2) / 4

		sI = s[1] / 2

	# Calculating the distance between center node and interface along a single dimension (x, y, or z).
	if math.fabs( c2 ) < eps:
		sI -= c0 / c1
	else:
		radical = math.sqrt( c1 ** 2 - 4 * c2 * c0 )
		if phi[2] < 0:
			sI += (-c1 + radical) / (2 * c2)
		else:
			sI += (-c1 - radical) / (2 * c2)

	return sI


def distanceToLineSegment( p: np.ndarray, v1: np.ndarray, v2: np.ndarray ) -> float:
	"""
	Compute the shortest distance between a point and a line segment.
	:param p: Query point.
	:param v1: Line vertex 1.
	:param v2: Line vertex 2.
	:return: Normal distance.
	"""
	v: np.ndarray = v2 - v1					# From point to vectors having a common source.
	u: np.ndarray = p - v1
	projUv = u.dot( v ) / v.dot( v ) * v	# Projection of u onto v.
	projUvPerp: np.ndarray = projUv - u		# Negative projection of u onto v perp.
	return math.sqrt( projUvPerp.dot( projUvPerp ) )

# Testing.
if __name__ == "__main__":

	# Distance from point to a line segment.
	p1 = np.array( [1, 0] )
	p2 = np.array( [1, 1] )
	q = np.array( [1, 1] )
	print( "From", q, "to line through", p1, "and", p2, ":", distanceToLineSegment( q, p1, p2 ) )

	# Distance to interface from grid node.
	phiValues = [-1.3, -1.2, -0.5, 0.1, 0.8, 1.3]
	sValues = [1, 1, 1, 1, 1]
	print( approximateNodeDistanceToInterface( phiValues[:5], sValues[:4], True ) )
	print( approximateNodeDistanceToInterface( phiValues[1:], sValues[1:], False ) )