"""
A module for annotating tracked players on individual video frames.

This module provides a lightweight wrapper around the existing drawing utilities so
single-frame player annotations can be reused in notebooks and scripts.
"""

from drawers.utils import draw_ellipse, draw_triangle


class PlayerAnnotator:
	"""
	Annotate tracked players on a single frame.

	Attributes:
		team_colors (dict): Mapping from team or class id to BGR annotation color.
	"""

	def __init__(self, team_colors=None):
		"""
		Initialize the annotator with team-specific drawing colors.

		Args:
			team_colors (dict, optional): Mapping from class or team id to BGR color.
				Defaults to colors for ids 0 and 1.
		"""
		self.team_colors = team_colors or {
			0: [255, 245, 238],
			1: [128, 0, 0],
		}

	def annotate(self, frame, player_tracks, ball_aquisition=None):
		"""
		Draw tracked players on a frame.

		Args:
			frame (numpy.ndarray): Frame to annotate.
			player_tracks (dict): Dictionary mapping track IDs to player tracking
				information. Each value is expected to contain a "bbox" entry.
			ball_aquisition (int, optional): Track ID of the player currently in
				possession of the ball. If None, no possession marker is drawn.

		Returns:
			numpy.ndarray: Annotated frame.
		"""
		annotated_frame = frame.copy()

		for track_id, player in player_tracks.items():
			player_color = self.team_colors.get(
				player.get("class_id"),
				self.team_colors[0]
			)
			annotated_frame = draw_ellipse(
				annotated_frame,
				player["bbox"],
				player_color,
				track_id
			)

			if ball_aquisition is not None and track_id == ball_aquisition:
				annotated_frame = draw_triangle(
					annotated_frame,
					player["bbox"],
					(0, 0, 255)
				)

		return annotated_frame
