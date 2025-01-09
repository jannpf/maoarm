from dataclasses import dataclass, field
from arm import AngleControl


@dataclass
class PID:
    """
    A PID controller for smooth robotic arm control.

    Attributes:
        control (AngleControl): Interface for controlling the robotic arm's movement.
        dt (float): Time step between control updates.
        kp, ki, kd (float): PID gains.
        min_output, max_output (int): limits for the speed controller output.
    """

    control: AngleControl
    dt: float

    kpx: float = 80.0
    kpy: float = 16.0
    kix: float = 0.1
    kiy: float = 0.1
    kdx: float = 0.5
    kdy: float = 0.1

    min_output_x: int = 10
    max_output_x: int = 30
    min_output_y: int = 2
    max_output_y: int = 20

    # Internal variables
    error_sum_x: float = field(default=0.0, init=False)
    error_sum_y: float = field(default=0.0, init=False)
    last_error_x: float = field(default=0.0, init=False)
    last_error_y: float = field(default=0.0, init=False)

    def move_control(
        self,
        target_x: float,
        target_y: float,
        width: float,
        height: float,
    ):
        """
        Moves the robotic arm in the direction of the specified coordinates
        (e.g. the central point of the detected face).
        Adjusts movement speed proportionally based on distance from the target
        to ensure smooth control.

        Args:
            control (AngleControl): The control interface for the robotic arm.
            target_x: Target x-coordinate relative to the frame's center.
            target_y: Target y-coordinate relative to the frame's center.
            width: Width of the frame/image.
            height: Height of the frame/image.
        """
        error_x = abs(target_x / (width / 2))
        error_y = abs(target_y / (height / 2))

        # Proportional term
        p_x = self.kpx * error_x
        p_y = self.kpy * error_y

        # Integral term
        self.error_sum_x += error_x * self.dt
        self.error_sum_x += error_x * self.dt
        i_x = self.error_sum_x * self.kix
        i_y = self.error_sum_y * self.kiy

        # Derivative term
        d_x = self.kdx * (error_x - self.last_error_x) / self.dt
        d_y = self.kdy * (error_y - self.last_error_y) / self.dt
        self.last_error_x = error_x
        self.last_error_y = error_x

        # PID output
        control_x = p_x + i_x + d_x
        control_y = p_y + i_y + d_y

        # np.clip from min to max
        spdx = min(max(int(control_x), self.min_output_x), self.max_output_x)
        spdy = min(max(int(control_y), self.min_output_y), self.max_output_y)

        if target_x > width / 20:
            self.control.base_cw(spdx)
        elif target_x < - width / 20:
            self.control.base_ccw(spdx)
        else:
            self.control.base_stop()
        if target_y > height / 10:
            self.control.elbow_up(spdy)
        elif target_y < - height / 10:
            self.control.elbow_down(spdy)
        else:
            self.control.elbow_stop()

        # reset shoulder as it tends to move
        self.control.shoulder_to(0, spd=2, acc=2)
