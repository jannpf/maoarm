import json
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from pathlib import Path
from time import time

from arm import AngleControl

VISUALIZATION_FILE = Path("interim_values.json")


@dataclass
class PID:
    """
    A PID controller for smooth robotic arm control. As a side effect,
    records values of current control params over time into a json file.

    Attributes:
        control (AngleControl): Interface for controlling the robotic arm's movement.
        record_visualization (bool): Whether to record current parameter values.
        dt (float): Time step between control updates.
        kp, ki, kd (float): PID gains.
        min_output, max_output (int): limits for the speed controller output.
    """

    control: AngleControl
    record_visualization: bool = True

    # could use more testing in different conditions
    # (lighting, crowds, distances)
    # Also, added weight to the arm changes the behaviour more than initially 
    # expected, especially due to more physical momentum of the "head"
    kpx: float = 80.0
    kpy: float = 12.0
    kix: float = 2
    kiy: float = 0.8
    kdx: float = 2.8
    kdy: float = 1

    min_output_x: int = 5
    max_output_x: int = 50
    min_output_y: int = 2
    max_output_y: int = 20

    # Internal variables
    error_sum_x: float = field(default=0.0, init=False)
    error_sum_y: float = field(default=0.0, init=False)
    last_error_x: float = field(default=0.0, init=False)
    last_error_y: float = field(default=0.0, init=False)

    last_control_x: float = field(default=0.0, init=False)
    last_control_y: float = field(default=0.0, init=False)

    momentum: float = 0.2

    previous_time: float = 0

    def __post_init__(self):
        # Clear the content of the visualization file if it exists
        if self.record_visualization:
            VISUALIZATION_FILE.touch(exist_ok=True)  # Ensure the file exists
            VISUALIZATION_FILE.write_text('[]')  # Empty the file

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
        current_time = time()
        self.dt: float = current_time - self.previous_time
        self.previous_time = current_time

        # for first iteration, dt is not usable
        if self.dt == current_time:
            return


        error_x = target_x / (width / 2)
        error_y = target_y / (height / 2)

        # Proportional term
        p_x = self.kpx * error_x
        p_y = self.kpy * error_y

        # Integral term
        self.error_sum_x += error_x * self.dt
        self.error_sum_y += error_y * self.dt
        i_x = self.error_sum_x * self.kix
        i_y = self.error_sum_y * self.kiy

        # Derivative term
        d_x = self.kdx * (error_x - self.last_error_x) / self.dt
        d_y = self.kdy * (error_y - self.last_error_y) / self.dt
        self.last_error_x = error_x
        self.last_error_y = error_y

        # PID output
        control_x = p_x + i_x + d_x
        control_y = p_y + i_y + d_y


        # momentum term (inspired by SGD) to prevent too sudden change
        # only one previous value considered, jumps in I and D can cause faltering behaviour
        control_x = self.last_control_x * self.momentum + (1 - self.momentum) * control_x
        control_y = self.last_control_y * self.momentum + (1 - self.momentum) * control_y
        
        self.last_control_x = control_x
        self.last_control_y = control_y

        # since we are controlling the speed, take the abs output
        # and clip from min to max
        spdx = round(min(max(int(abs(control_x)), self.min_output_x), self.max_output_x), 3)
        spdy = round(min(max(int(abs(control_y)), self.min_output_y), self.max_output_y), 3)

        # decreased the ratios to get a swifter response after resting
        # could use more testing in different conditions
        # (lighting, crowds, distances)
        # Also, added weight to the arm changes the behaviour more than initially 
        # expected, especially due to more physical momentum of the "head"
        if target_x > width / 30:
            self.control.base_cw(spdx)
        elif target_x < - width / 30:
            self.control.base_ccw(spdx)
        else:
            self.control.base_stop()
        if target_y > height / 15:
            self.control.elbow_up(spdy)
        elif target_y < - height / 15:
            self.control.elbow_down(spdy)
        else:
            self.control.elbow_stop()

        if self.record_visualization:
            entry = {
                "time": round(current_time, 3),
                "target_x": target_x,
                "target_y": target_y,
                "error_x": error_x,
                "error_y": error_y,
                "p_x": p_x,
                "p_y": p_y,
                "i_x": i_x,
                "i_y": i_y,
                "d_x": d_x,
                "d_y": d_y,
                "control_x": control_x,
                "control_y": control_y,
                "spdx": spdx,
                "spdy": spdy
            }
            self._append_to_json_file(entry)


    def _append_to_json_file(self, entry: dict):
        import json
        try:
            with open(VISUALIZATION_FILE, 'r+') as f:
                data = json.load(f)
                data.append(entry)
                f.seek(0)
                json.dump(data, f, indent=4)
        except FileNotFoundError:  # just in case
            with open(VISUALIZATION_FILE, 'w') as f:
                json.dump([entry], f, indent=4)


def visualize(file=VISUALIZATION_FILE):
    # Load data from file
    with open(file, 'r') as f:
        data = json.load(f)

    # Extracting times and corresponding values
    fields = [
        "target_x", "control_x",
        "target_x", "spdx",
        "target_y", "control_y",
        "target_y", "spdy",
        "p_x", "p_y",
        "i_x", "i_y",
        "d_x", "d_y",
    ]

    times = [entry["time"] for entry in data]
    values = {field: [entry[field] for entry in data] for field in fields}

    # Create subplots
    num_fields = len(fields) // 2  # Each pair of x, y counts as one plot
    fig, axes = plt.subplots(num_fields, 1, figsize=(12, 3 * num_fields), sharex=True)

    # Plot each pair of fields
    for i in range(num_fields):
        field_x = fields[2 * i]
        field_y = fields[2 * i + 1]
        ax = axes[i]
        ax2 = ax.twinx()

        ax.plot(times, values[field_x], label=f"{field_x}", linestyle="-", marker=None)
        ax2.plot(times, values[field_y], label=f"{field_y}", linestyle="-", color='orange', marker=None)
        ax.set_ylabel("Values")
        ax.legend(loc=0)
        ax2.legend(loc=1)
        ax.grid(True)

        # Add dashed vertical lines at 5-second intervals
        max_time = max(times)
        min_time = min(times)
        step = 5
        for t in range(int(min_time), int(max_time) + step, step):
            ax.axvline(x=t, color='gray', linestyle='--', linewidth=0.8)

    # Final plot adjustments
    axes[-1].set_xlabel("Time (seconds)")
    plt.suptitle("PID Response")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
