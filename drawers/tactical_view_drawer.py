import cv2
import numpy as np

from configs.basketball import BasketballCourtConfiguration

class TacticalViewDrawer:
    def __init__(self, team_1_color=[255, 245, 238], team_2_color=[128, 0, 0]):
        self.start_x = 20
        self.start_y = 40
        self.team_1_color = team_1_color
        self.team_2_color = team_2_color

    def _to_pixel_point(self, point):
        """Convert a 2D point-like value to integer pixel coordinates for OpenCV."""
        if point is None or len(point) < 2:
            return None

        try:
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
        except (TypeError, ValueError):
            return None

        return x, y

    def _scale_point(self, point, scale_x, scale_y):
        """Scale tactical coordinates into display coordinates."""
        pixel_point = self._to_pixel_point(point)
        if pixel_point is None:
            return None

        x, y = pixel_point
        return int(round(x * scale_x)), int(round(y * scale_y))

    def _build_court_from_config(self, court_config, width, height):
        """Build a tactical court image using configuration vertices and edges."""
        court = np.full((height, width, 3), (132, 164, 196), dtype=np.uint8)

        scale_x = width / float(court_config.length)
        scale_y = height / float(court_config.width)

        def to_canvas(point):
            x = int(round(point[0] * scale_x))
            y = int(round(point[1] * scale_y))
            x = min(max(x, 0), width - 1)
            y = min(max(y, 0), height - 1)
            return x, y

        line_color = (255, 255, 255)
        line_thickness = 2

        for start, end in court_config.edges:
            p1 = to_canvas(court_config.vertices[start - 1])
            p2 = to_canvas(court_config.vertices[end - 1])
            cv2.line(court, p1, p2, line_color, line_thickness)

        left_hoop = to_canvas(court_config.vertices[6])
        right_hoop = to_canvas(court_config.vertices[26])
        hoop_radius = max(3, int(round(min(width, height) * 0.01)))
        cv2.circle(court, left_hoop, hoop_radius, line_color, line_thickness)
        cv2.circle(court, right_hoop, hoop_radius, line_color, line_thickness)

        center = to_canvas(court_config.vertices[16])
        cv2.circle(court, center, max(2, hoop_radius - 1), line_color, -1)

        return court

    def draw(self, 
             video_frames, 
             court_image_path, 
             width,
             height,
             tactical_court_keypoints,
             tactical_player_positions=None,
             player_assignment=None,
             ball_acquisition=None,
             tactical_source_width=None,
             tactical_source_height=None):
        """
        Draw tactical view with court keypoints and player positions.
        
        Args:
            video_frames (list): List of video frames to draw on.
            court_image_path (str): Path to the court image.
            width (int): Width of the tactical view.
            height (int): Height of the tactical view.
            tactical_court_keypoints (list): List of court keypoints in tactical view.
            tactical_player_positions (list, optional): List of dictionaries mapping player IDs to 
                their positions in tactical view coordinates.
            player_assignment (list, optional): List of dictionaries mapping player IDs to team assignments.
            ball_acquisition (list, optional): List indicating which player has the ball in each frame.
            tactical_source_width (int, optional): Width of tactical coordinate system used by points.
                If not provided, width is used (no scaling).
            tactical_source_height (int, optional): Height of tactical coordinate system used by points.
                If not provided, height is used (no scaling).
            
        Returns:
            list: List of frames with tactical view drawn on them.
        """
        source_width = tactical_source_width if tactical_source_width else width
        source_height = tactical_source_height if tactical_source_height else height
        scale_x = width / source_width
        scale_y = height / source_height

        court_image = cv2.imread(court_image_path)
        court_image = cv2.resize(court_image, (width, height))

        output_video_frames = []
        for frame_idx, frame in enumerate(video_frames):
            frame = frame.copy()

            y1 = self.start_y
            y2 = self.start_y+height
            x1 = self.start_x
            x2 = self.start_x+width
            
            alpha = 0.6  # Transparency factor
            overlay = frame[y1:y2, x1:x2].copy()
            cv2.addWeighted(court_image, alpha, overlay, 1 - alpha, 0, frame[y1:y2, x1:x2])
            
            # Draw court keypoints
            for keypoint_index, keypoint in enumerate(tactical_court_keypoints):
                pixel_point = self._scale_point(keypoint, scale_x, scale_y)
                if pixel_point is None:
                    continue

                x, y = pixel_point
                x += self.start_x
                y += self.start_y
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(frame, str(keypoint_index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw player positions in tactical view if available
            if tactical_player_positions and player_assignment and frame_idx < len(tactical_player_positions):
                frame_positions = tactical_player_positions[frame_idx]
                frame_assignments = player_assignment[frame_idx] if frame_idx < len(player_assignment) else {}
                player_with_ball = ball_acquisition[frame_idx] if ball_acquisition and frame_idx < len(ball_acquisition) else -1
                
                for player_id, position in frame_positions.items():
                    # Get player's team
                    team_id = frame_assignments.get(player_id, 1)  # Default to team 1 if not assigned
                    
                    # Set color based on team
                    color = self.team_1_color if team_id == 1 else self.team_2_color
                    
                    # Adjust position to overlay coordinates
                    pixel_point = self._scale_point(position, scale_x, scale_y)
                    if pixel_point is None:
                        continue
                    x, y = pixel_point[0] + self.start_x, pixel_point[1] + self.start_y
                    
                    # Draw player circle
                    player_radius = 8
                    cv2.circle(frame, (x, y), player_radius, color, -1)
                    
                    # Add player ID
                    #cv2.putText(frame, str(player_id), (x-4, y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                    
                    # Highlight player with ball
                    if player_id == player_with_ball:
                        cv2.circle(frame, (x, y), player_radius+3, (0, 0, 255), 2)
            
            output_video_frames.append(frame)

        return output_video_frames

    def draw_on_config(
            self,
            video_frames,
            court_config,
            width,
            height,
            tactical_court_keypoints,
            tactical_player_positions=None,
            player_assignment=None,
            ball_acquisition=None,
            tactical_source_width=None,
            tactical_source_height=None):
        """
        Draw tactical view using BasketballCourtConfiguration instead of a court image file.

        Args:
            video_frames (list): List of video frames to draw on.
            court_config (BasketballCourtConfiguration): Court geometry configuration.
            width (int): Width of the tactical view overlay.
            height (int): Height of the tactical view overlay.
            tactical_court_keypoints (list): List of court keypoints in tactical view coordinates.
            tactical_player_positions (list, optional): List of dictionaries mapping player IDs to positions.
            player_assignment (list, optional): List of dictionaries mapping player IDs to team assignments.
            ball_acquisition (list, optional): List indicating which player has the ball in each frame.
            tactical_source_width (int, optional): Width of tactical coordinate system used by points.
            tactical_source_height (int, optional): Height of tactical coordinate system used by points.

        Returns:
            list: List of frames with tactical view drawn on them.
        """
        if court_config is None:
            court_config = BasketballCourtConfiguration()

        source_width = tactical_source_width if tactical_source_width else width
        source_height = tactical_source_height if tactical_source_height else height
        scale_x = width / source_width
        scale_y = height / source_height

        court_image = self._build_court_from_config(court_config, width, height)

        output_video_frames = []
        for frame_idx, frame in enumerate(video_frames):
            frame = frame.copy()

            y1 = self.start_y
            y2 = self.start_y + height
            x1 = self.start_x
            x2 = self.start_x + width

            alpha = 0.6
            overlay = frame[y1:y2, x1:x2].copy()
            cv2.addWeighted(court_image, alpha, overlay, 1 - alpha, 0, frame[y1:y2, x1:x2])

            for keypoint_index, keypoint in enumerate(tactical_court_keypoints):
                pixel_point = self._scale_point(keypoint, scale_x, scale_y)
                if pixel_point is None:
                    continue

                x, y = pixel_point
                x += self.start_x
                y += self.start_y
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(frame, str(keypoint_index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if tactical_player_positions and player_assignment and frame_idx < len(tactical_player_positions):
                frame_positions = tactical_player_positions[frame_idx]
                frame_assignments = player_assignment[frame_idx] if frame_idx < len(player_assignment) else {}
                player_with_ball = ball_acquisition[frame_idx] if ball_acquisition and frame_idx < len(ball_acquisition) else -1

                for player_id, position in frame_positions.items():
                    team_id = frame_assignments.get(player_id, 1)
                    color = self.team_1_color if team_id == 1 else self.team_2_color

                    pixel_point = self._scale_point(position, scale_x, scale_y)
                    if pixel_point is None:
                        continue

                    x = pixel_point[0] + self.start_x
                    y = pixel_point[1] + self.start_y
                    player_radius = 8
                    cv2.circle(frame, (x, y), player_radius, color, -1)

                    if player_id == player_with_ball:
                        cv2.circle(frame, (x, y), player_radius + 3, (0, 0, 255), 2)

            output_video_frames.append(frame)

        return output_video_frames
