# for motion switcher
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
# for loco client
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
import time

# MotionSwitcher used to switch mode between debug mode and ai mode
class MotionSwitcher:
    def __init__(self):
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(1.0)
        self.msc.Init()

    def Enter_Debug_Mode(self):
        try:
            status, result = self.msc.CheckMode()
            while result['name']:
                self.msc.ReleaseMode()
                status, result = self.msc.CheckMode()
                time.sleep(1)
            return status, result
        except Exception as e:
            return None, None
    
    def Exit_Debug_Mode(self):
        try:
            status, result = self.msc.SelectMode(nameOrAlias='ai')
            return status, result
        except Exception as e:
            return None, None

class LocoClientWrapper:
    def __init__(self):
        self.client = LocoClient()
        self.client.SetTimeout(0.0001)
        self.client.Init()

    def Enter_Damp_Mode(self):
        self.client.Damp()

    def Damp(self):
        # Alias for Enter_Damp_Mode: teleop_hand_and_arm.py's double-thumbstick
        # soft-e-stop path calls loco_wrapper.Damp() directly, which this class
        # never defined (only Enter_Damp_Mode existed) -- that call would raise
        # AttributeError instead of actually damping the robot. Local patch,
        # 2026-07-21, ahead of first walking-mode use.
        self.client.Damp()

    # ---------------------------- DIMENSO ADDITION ----------------------------
    def StopMove(self):
        """Stop walking and KEEP BALANCING. This is what the double-thumbstick
        combo calls now; `Damp()` above is no longer bound to any controller
        input. See teleop_hand_and_arm.py's double-thumbstick block for why.

        `StopMove()` zeroes the commanded velocity and leaves the balance
        controller running, so the robot stays upright on its own feet. In FSM
        501 with balance mode 0 the FSM starts and stops the gait itself, which
        is why zeroing velocity is sufficient to end a walk -- measured on
        Hercules 2026-09-02, and the reason balance mode 1 was abandoned there
        (continuous gait marches in place forever and StopMove cannot end it).
        """
        self.client.StopMove()
    # -------------------------- END DIMENSO ADDITION --------------------------

    def Move(self, vx, vy, vyaw):
        self.client.Move(vx, vy, vyaw, continous_move=False)

if __name__ == '__main__':
    ChannelFactoryInitialize(1) # 0 for real robot, 1 for simulation
    ms = MotionSwitcher()
    status, result = ms.Enter_Debug_Mode()
    print("Enter debug mode:", status, result)
    time.sleep(5)
    status, result = ms.Exit_Debug_Mode()
    print("Exit debug mode:", status, result)
    time.sleep(2)
