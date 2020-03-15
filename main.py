import fsm
import importlib

importlib.reload( fsm )

#################### Signed distance function to origin ####################

fsmSerial = fsm.FSM( 101 )
fsmSerial.definePointAtOriginInterface()
fsmSerial.goSerial()
fsmSerial.plotSurface()

fsmParallel = fsm.FSM( 101 )
fsmParallel.definePointAtOriginInterface()
fsmParallel.goParallel()
fsmParallel.plotSurface()