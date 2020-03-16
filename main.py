import fsm
import importlib

importlib.reload( fsm )

#################### Signed distance function to origin ####################

if __name__ == '__main__':
	# Serial.
	# fsmSerial = fsm.Serial( 11 )
	# fsmSerial.definePointAtOriginInterface()
	# fsmSerial.go()
	# fsmSerial.plotSurface()

	# Parallel.
	fsmParallel = fsm.Parallel( 101 )
	fsmParallel.definePointAtOriginInterface()
	fsmParallel.go( processes=4 )
	fsmParallel.plotSurface()