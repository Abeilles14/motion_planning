# STATE MACHINE FOR 3D PICK AND PLACE SIMULATION
import numpy as np
from numpy.linalg import norm
from math import *
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from random import random
from scipy.spatial import ConvexHull
from matplotlib import path
import time
from mpl_toolkits import mplot3d
from enum import Enum

from utils import init_fonts
from path_shortening import shorten_path
from obstacles import Parallelepiped
from arm import Arm

### CONSTANTS ###
ARM_HOME_POS = [0, 0, 0]    #TODO

class ArmState(Enum):
    APPROACH_OBJECT = 1,
    GRAB_OBJECT = 2,
    APPROACH_DEST = 3,
    DROP_OBJECT = 4,
    HOME = 5,
    DONE = 6

class ArmStateMachine:
    def __init__(self, arm, object, log_verbose=True):
        self.state = ArmState.APPROACH_OBJECT
        self.arm = arm
        self.object = object

        if self.log_verbose:
            loginfo("ArmStateMachine:__init__")
            loginfo("arm: {}, object: {}".format(
                    arm.name(), world_to_psm_tf, object))
            loginfo("home_when_done: {}".format(self.home_when_done))

        self.state_functions = {
            ArmState.APPROACH_OBJECT : self._approach_object,
            ArmState.GRAB_OBJECT : self._grab_object,
            ArmState.APPROACH_DEST : self._approach_dest,
            ArmState.DROP_OBJECT : self._drop_object,
            ArmState.HOME : self._home,
        }
        self.next_functions = {
            ArmState.APPROACH_OBJECT : self._approach_object_next,
            ArmState.GRAB_OBJECT : self._grab_object_next,
            ArmState.APPROACH_DEST : self._approach_dest_next,
            ArmState.DROP_OBJECT : self._drop_object_next,
            ArmState.HOME : self._home_next
        }

    ### STATE FUNCTIONS ###
    def _approach_object(self):
        if self.log_verbose:
            loginfo("Picking Object {}".format(self.object.pos))
        self._set_arm_dest(self._obj_pos())

    def _approach_object_next(self):
        if self.psm._arm__goal_reached and \
            vector_eps_eq(self.psm.get_current_position().p, self._obj_pos()):
            return PickAndPlaceState.GRAB_OBJECT
        else:
            return PickAndPlaceState.APPROACH_OBJECT
    
    def _grab_object(self):
        self._set_arm_dest(self._obj_pos() + self._approach_vec())

    def _grab_object_next(self):
        if self.psm._arm__goal_reached and \
            vector_eps_eq(self.psm.get_current_position().p, self._obj_pos() + self._approach_vec()):
            return PickAndPlaceState.CLOSE_JAW
        else:
            return PickAndPlaceState.GRAB_OBJECT

     def _approach_dest(self):
        self._set_arm_dest(self._obj_dest())

    def _approach_dest_next(self):
        if self.psm._arm__goal_reached and \
            vector_eps_eq(self.psm.get_current_position().p, self._obj_dest()):
            return PickAndPlaceState.DROP_OBJECT
        else:
            return PickAndPlaceState.APPROACH_DEST 

    def _drop_object(self):
        if self.psm.get_desired_jaw_position() < math.pi / 3:
            self.psm.open_jaw(blocking=False)

    def _drop_object_next(self):
        # open_jaw() sets jaw to 80 deg, we check if we're open past 60 deg
        if self.psm.get_current_jaw_position() > math.pi / 3:
            # early out if this is being controlled by the parent state machine
            if not self.closed_loop:
                if self.home_when_done:
                    return PickAndPlaceState.HOME
                else:
                    return PickAndPlaceState.DONE

            elif len(self.world.objects) > 0:
                # there are objects left, find one and go to APPROACH_OBJECT
                closest_object = None
                if self.pick_closest_to_base_frame:
                    # closest object to base frame
                    closest_object = min(self.world.objects,
                                        key=lambda obj : (self.world_to_psm_tf * obj.pos).Norm())
                else:
                    # closest object to current position, only if we're running 
                    closest_object = min(self.world.objects,
                                        key=lambda obj : (self.world_to_psm_tf * obj.pos \
                                                        - self.psm.get_current_position().p).Norm())
                self.object = closest_object
                return PickAndPlaceState.APPROACH_OBJECT
            else:
                return PickAndPlaceState.HOME
        else:
            return PickAndPlaceState.DROP_OBJECT

    def _home(self):
        self._set_arm_dest(PSM_HOME_POS)

    def _home_next(self):
        # the home state is used for arm state machines that are completely 
        # finished executing as determined by the parent state machine
        return PickAndPlaceState.HOME 

    ### END STATE FUNCTIONS ###

    def _obj_pos(self):
        return self.world_to_psm_tf * self.object.pos

    def _approach_vec(self):
        return self.world_to_psm_tf.M * self.approach_vec

    def _obj_dest(self):
        return self.world_to_psm_tf * self.obj_dest

    def _set_arm_dest(self, dest):
        if self.log_verbose:
            loginfo("Setting {} dest to {}".format(self.psm.name(), dest))
        if self.psm.get_desired_position().p != dest:
            self.psm.move(PyKDL.Frame(DOWN_JAW_ORIENTATION, dest), blocking=False)

    def run_once(self):
        if self.log_verbose:
            loginfo("Running state {}".format(self.state))
        if self.is_done():
            return
        # execute the current state
        self.state_functions[self.state]()

        self.state = self.next_functions[self.state]()

