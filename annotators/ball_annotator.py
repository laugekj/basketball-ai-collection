"""
A module for annotating tracked balls on individual video frames.

This module provides a lightweight wrapper around the existing drawing utilities so
single-frame ball annotations can be reused in notebooks and scripts.
"""

from drawers.utils import draw_triangle


class BallAnnotator:
    """
    Annotate tracked balls on a single frame.

    Attributes:
        color (tuple): BGR color used to draw the ball marker.
    """

    def __init__(self, color=(0, 255, 0)):
        """
        Initialize the annotator with the drawing color for balls.

        Args:
            color (tuple, optional): BGR color for ball annotation.
                Defaults to (0, 255, 0).
        """
        self.color = color

    def annotate(self, frame, ball_track):
        """
        Draw tracked balls on a frame.

        Args:
            frame (numpy.ndarray): Frame to annotate.
            ball_track (dict): Dictionary containing ball tracking information.
                Each value is expected to contain a "bbox" entry.

        Returns:
            numpy.ndarray: Annotated frame.
        """
        annotated_frame = frame.copy()

        for _, ball in ball_track.items():
            if ball["bbox"] is None:
                continue

            annotated_frame = draw_triangle(
                annotated_frame,
                ball["bbox"],
                self.color
            )

        return annotated_frame
