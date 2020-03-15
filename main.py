import fsm
import importlib

importlib.reload( fsm )

#################### Signed distance function to origin ####################

fsmObj = fsm.FSM( 11 )
fsmObj.definePointAtOriginInterface()
fsmObj.goParallel()
fsmObj.plotSurface()