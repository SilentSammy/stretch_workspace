import sys
import os
# Add hybrid_control directory to path (assumes it's a sibling directory)
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'hybrid_control'))
import state_control as sc
from cmd_src import CommandSource
import hybrid_control as hc
from normalized_velocity_control import NormalizedVelocityControl, zero_vel

class Interceptor(CommandSource):
    def __init__(self, robot):
        self.robot = robot
        self.state_controller = sc.StateControl(robot, sc.stowed_state)
        self.toy_visible = False
        self.visual_servoing_cmd = zero_vel.copy()
    
    def reset_iteration(self):
        self.toy_visible = False
        self.visual_servoing_cmd = zero_vel.copy()
    
    def get_cmd(self):
        wrist_yaw_cmd = {
            "wrist_yaw_counterclockwise": self.visual_servoing_cmd.get("wrist_yaw_counterclockwise", 0.0)
        }
        stow_cmd = self.state_controller.get_command()

        print("Visual servoing cmd:", self.visual_servoing_cmd)
        print("Wrist yaw cmd:", wrist_yaw_cmd)
        print("Stow cmd:", stow_cmd)

        # If toy is visible, concede wrist yaw to visual servoing, so it follows the toy, but while stowed
        if self.toy_visible:
            if "wrist_yaw_counterclockwise" in stow_cmd:
                del stow_cmd["wrist_yaw_counterclockwise"]

        final_cmd = hc.merge_override(stow_cmd, wrist_yaw_cmd)
        # return final_cmd
        return self.visual_servoing_cmd
