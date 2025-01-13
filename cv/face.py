from typing import Optional


class Face:
    """
    A class defining a custom representation for a detected bounding box.

    Attributes:
        frame_width (float): Width of the frame.
        frame_height (float): Height of the frame.
        x (Optional[float]): Center x-coordinate of the bounding box (relative to frame center).
        y (Optional[float]): Center y-coordinate of the bounding box (relative to frame center, inverted).
        w (Optional[float]): Width of the bounding box.
        h (Optional[float]): Height of the bounding box.
    """

    def __init__(
        self,
        lx: Optional[float],
        ly: Optional[float],
        rx: Optional[float],
        ry: Optional[float],
        width: float,
        height: float,
    ):
        """
        Initialize a Face object.

        Args:
            lx (Optional[float]): Top-left x-coordinate of the bounding box.
            ly (Optional[float]): Top-left y-coordinate of the bounding box.
            rx (Optional[float]): Bottom-right x-coordinate of the bounding box.
            ry (Optional[float]): Bottom-right y-coordinate of the bounding box.
            width (float): Width of the frame.
            height (float): Height of the frame.
        """

        self.frame_width = width
        self.frame_height = height

        if all(value is not None for value in (lx, ly, rx, ry)):
            self.x, self.y, self.w, self.h = self._to_center_coord(lx, ly, rx, ry)
        else:
            self.x, self.y, self.w, self.h = None, None, None, None

    def __repr__(self):
        """Calling face should return centered coordinates."""
        return f"Face center at: {self.x, self.y}; frame dim is {self.frame_width}x{self.frame_height}"

    def _to_center_coord(self, lx: float, ly: float, rx: float, ry: float):
        """
        Convert bounding box coordinates to centered coordinates.

        Returns:
            tuple[float, float, float, float, float, float]: A tuple containing:
                - center_x (float): Center x-coordinate with respect to the frame's center.
                - center_y (float): Center y-coordinate with respect to the frame's center (inverted).
                - box_w (float): Width of the bounding box.
                - box_h (float): Height of the bounding box.
        """
        box_w = rx - lx
        box_h = ry - ly

        # Convert the coordinates so that (0, 0) is at the center of the frame
        adjusted_lx = lx - self.frame_width / 2
        adjusted_ly = ly - self.frame_height / 2

        # Calculate the center of the bounding box
        center_x = adjusted_lx + box_w / 2
        center_y = adjusted_ly + box_h / 2

        return (center_x, -center_y, box_w, box_h)

    def is_detected(self) -> bool:
        return all(value is not None for value in (self.x, self.y, self.w, self.h))

    @classmethod
    def empty(cls, width: float = 0.0, height: float = 0.0) -> 'Face':
        """
        Create an empty Face object with no coordinates.

        Args:
            width (float): Width of the frame.
            height (float): Height of the frame.

        Returns:
            Face: An instance of Face with all coordinates set to None.
        """
        return cls(None, None, None, None, width, height)
