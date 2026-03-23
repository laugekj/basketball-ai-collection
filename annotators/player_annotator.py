"""
A module for annotating tracked players on individual video frames.

This module provides a lightweight wrapper around the existing drawing utilities so
single-frame player annotations can be reused in notebooks and scripts.
"""

from drawers.utils import draw_ellipse


class PlayerAnnotator:
	"""
	Annotate tracked players on a single frame.

	Attributes:
		color (list): BGR color used to draw the player ellipse and label box.
	"""

	def __init__(self, color=[255, 245, 238]):
		"""
		Initialize the annotator with the drawing color for players.

		Args:
			color (list, optional): BGR color for player annotation.
				Defaults to [255, 245, 238].
		"""
		self.color = color

	def annotate(self, frame, player_tracks):
		"""
		Draw tracked players on a frame.

		Args:
			frame (numpy.ndarray): Frame to annotate.
			player_tracks (dict): Dictionary mapping track IDs to player tracking
				information. Each value is expected to contain a "bbox" entry.

		Returns:
			numpy.ndarray: Annotated frame.
		"""
		annotated_frame = frame.copy()

		for track_id, player in player_tracks.items():
			annotated_frame = draw_ellipse(
				annotated_frame,
				player["bbox"],
				self.color,
				track_id
			)

		return annotated_frame
