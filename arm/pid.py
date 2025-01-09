from dataclasses import dataclass, field
from arm import AngleControl
from time import time


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

    interim_values: list = field(default_factory=list, init=False)
    start_time = time()

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
        current_time = time() - self.start_time
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


    def _append_to_json_file(self, entry):
        import json
        try:
            with open("interim_values.json", 'r+') as f:
                data = json.load(f)
                data.append(entry)
                f.seek(0)
                json.dump(data, f, indent=4)
        except FileNotFoundError:
            with open("interim_values.json", 'w') as f:
                json.dump([entry], f, indent=4)


def visualize(file="interim_values.json")
    import matplotlib.pyplot as plt
    import json

    # Load data from file
    with open(file, 'r') as f:
        data = json.load(f)

    # Extracting times and corresponding values
    fields = [
        "target_x", "target_y", "error_x", "error_y",
        "p_x", "p_y", "i_x", "i_y",
        "d_x", "d_y", "control_x", "control_y",
        "spdx", "spdy"
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

        ax.plot(times, values[field_x], label=f"{field_x}", linestyle="-", marker=None)
        ax.plot(times, values[field_y], label=f"{field_y}", linestyle="-", marker=None)
        ax.set_ylabel("Values")
        ax.legend()
        ax.grid(True)

        # Add dashed vertical lines at 5-second intervals
        max_time = max(times)
        min_time = min(times)
        step = 5
        for t in range(int(min_time), int(max_time) + step, step):
            ax.axvline(x=t, color='gray', linestyle='--', linewidth=0.8)

    # Final plot adjustments
    axes[-1].set_xlabel("Time (seconds)")
    plt.suptitle("Comparison of All Values Over Time")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
