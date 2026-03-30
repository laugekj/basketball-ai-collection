import os
import sys
import pathlib
import numpy as np
import cv2
from copy import deepcopy
from .homography import Homography

folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path,"../"))
from utils import get_foot_position,measure_distance

class TacticalViewConverter:
    width: int = 50   
    length: int = 94  
    
    key_length: int = 19
    key_width: int = 16

    three_point_distance: int = 28 # from baseline to top of the arc
    three_point_margin: int = 3 # distance from the three point line to the court outline
    three_point_line_length: int = 14 # length of the straight part of the three point line on the sides
    
    hoop_distance: int = 5.3
    
    def __init__(self, court_image_path):
        self.court_image_path = court_image_path
        # self.width = 300
        # self.height= 161
        self.width = 94
        self.height = 50

        self.actual_width_in_meters=28
        self.actual_height_in_meters=15 

        w = self.__class__.width
        l = self.__class__.length
        kw = self.key_width
        kl = self.key_length
        tpd = self.three_point_distance
        tpm = self.three_point_margin
        hd = self.hoop_distance
        tpll = self.three_point_line_length

        self.key_points = [
            # See my diagram for the numbering of vertices. 
            # But it goes from top left then down. Go right once, then down.

            # Left baseline.
            (0, 0),                # 1 top left.
            (0, tpm),              # 2 top left arc.
            (0, (w - kw) / 2),     # 3 top left key.                
            (0, (w + kw) / 2),     # 4 bottom left key.
            (0, w - tpm),          # 5 bottom left arc.
            (0, w),                # 6 bottom left

            # Left hoop
            (hd, w / 2),           # 7 left hoop.

            # Left 3pt arc
            (tpll, tpm),          # 8 top left arc.
            (tpll, w - tpm),      # 9 bottom left arc.

            # Left ft. line
            (kl, (w - kw) / 2),   # 10 top right corner of the key
            (kl, w / 2),          # 11 ft line center of the key
            (kl, (w + kw) / 2),   # 12 bottom right corner of the key

            # Left court sideline and 3pt top center
            (tpd, 0),             # 13 halfway between baseline and middle top side
            (tpd, w / 2),         # 14 top of the 3pt arc center
            (tpd, w),             # 15 halfway between baseline and middle bottom side

            # Middle of the court
            (l / 2, 0),            # 16 Top middle of court
            (l / 2, w / 2),        # 17 center of the court
            (l / 2, w),            # 18 Bottom middle of court
            
            # Right court sideline and 3pt top center
            (l - tpd, 0),          # 19 halfway between baseline and middle top side
            (l - tpd, w / 2),      # 20 top of the 3pt arc center
            (l - tpd, w),          # 21 halfway between baseline and middle bottom side
            
            # Right ft. line
            (l - kl, (w - kw) / 2),     # 22 top left corner of the key
            (l - kl, w / 2),            # 23 ft line center of the key
            (l - kl, (w + kw) / 2),     # 24 bottom left corner of the key
            
            # Right 3pt arc
            (l - tpll, tpm),      # 25 top corner of the arc away from baseline and sideline
            (l - tpll, w - tpm),  # 26 bottom corner of the arc away from baseline and sideline RIGHT

            # Right hoop
            (l - hd, w / 2),      # 27 right hoop

            # Right baseline
            (l, 0),                # 28 Right top corner of the court
            (l, tpm),              # 29 top corner of the arc sideline
            (l, (w - kw) / 2),     # 30 top right corner of the key
            (l, (w + kw) / 2),     # 31 bottom right corner of the key
            (l, w - tpm),          # 32 bottom corner of the arc sideline 
            (l, w),                # 33 right bottom corner of the court
        ]

    def validate_keypoints(self, keypoints_list):
        """
        Validates detected keypoints by comparing their proportional distances
        to the tactical view keypoints.
        
        Args:
            keypoints_list (List[List[Tuple[float, float]]]): A list containing keypoints for each frame.
                Each outer list represents a frame.
                Each inner list contains keypoints as (x, y) tuples.
                A keypoint of (0, 0) indicates that the keypoint is not detected for that frame.
        
        Returns:
            List[bool]: A list indicating whether each frame's keypoints are valid.
        """

        keypoints_list = deepcopy(keypoints_list)

        for frame_idx, frame_keypoints in enumerate(keypoints_list):
            frame_keypoints = frame_keypoints.xy.tolist()[0]
            
            # Get indices of detected keypoints (not (0, 0))
            detected_indices = [i for i, kp in enumerate(frame_keypoints) if kp[0] >0 and kp[1]>0]
            
            # Need at least 3 detected keypoints to validate proportions
            if len(detected_indices) < 3:
                continue
            
            invalid_keypoints = []
            # Validate each detected keypoint
            for i in detected_indices:
                # Skip if this is (0, 0)
                if frame_keypoints[i][0] == 0 and frame_keypoints[i][1] == 0:
                    continue

                # Choose two other random detected keypoints
                other_indices = [idx for idx in detected_indices if idx != i and idx not in invalid_keypoints]
                if len(other_indices) < 2:
                    continue

                # Take first two other indices for simplicity
                j, k = other_indices[0], other_indices[1]

                # Calculate distances between detected keypoints
                d_ij = measure_distance(frame_keypoints[i], frame_keypoints[j])
                d_ik = measure_distance(frame_keypoints[i], frame_keypoints[k])
                
                # Calculate distances between corresponding tactical keypoints
                t_ij = measure_distance(self.key_points[i], self.key_points[j])
                t_ik = measure_distance(self.key_points[i], self.key_points[k])

                # Calculate and compare proportions with 50% error margin
                if t_ij > 0 and t_ik > 0:
                    prop_detected = d_ij / d_ik if d_ik > 0 else float('inf')
                    prop_tactical = t_ij / t_ik if t_ik > 0 else float('inf')

                    error = (prop_detected - prop_tactical) / prop_tactical
                    error = abs(error)

                    if error >0.8:  # 80% error margin                        
                        keypoints_list[frame_idx].xy[0][i] *= 0
                        keypoints_list[frame_idx].xyn[0][i] *= 0
                        invalid_keypoints.append(i)
            
        return keypoints_list

    def transform_players_to_tactical_view(self, keypoints_list, player_tracks):
        """
        Transform player positions from video frame coordinates to tactical view coordinates.
        
        Args:
            keypoints_list (list): List of detected court keypoints for each frame.
            player_tracks (list): List of dictionaries containing player tracking information for each frame,
                where each dictionary maps player IDs to their bounding box coordinates.
        
        Returns:
            list: List of dictionaries where each dictionary maps player IDs to their (x, y) positions
                in the tactical view coordinate system. The list index corresponds to the frame number.
        """
        tactical_player_positions = []
        
        for frame_idx, (frame_keypoints, frame_tracks) in enumerate(zip(keypoints_list, player_tracks)):
            # Initialize empty dictionary for this frame
            tactical_positions = {}

            frame_keypoints = frame_keypoints.xy.tolist()[0]

            # Skip frames with insufficient keypoints
            if frame_keypoints is None or len(frame_keypoints) == 0:
                tactical_player_positions.append(tactical_positions)
                continue
            
            # Get detected keypoints for this frame
            detected_keypoints = frame_keypoints
            
            # Filter out undetected keypoints (those with coordinates (0,0))
            valid_indices = [i for i, kp in enumerate(detected_keypoints) if kp[0] > 0 and kp[1] > 0]
            
            # Need at least 4 points for a reliable homography
            if len(valid_indices) < 4:
                tactical_player_positions.append(tactical_positions)
                continue
            
            # Create source and target point arrays for homography
            source_points = np.array([detected_keypoints[i] for i in valid_indices], dtype=np.float32)
            target_points = np.array([self.key_points[i] for i in valid_indices], dtype=np.float32)
            
            try:
                # Create homography transformer
                homography = Homography(source_points, target_points)
                
                # Transform each player's position
                for player_id, player_data in frame_tracks.items():
                    bbox = player_data["bbox"]
                    # Use bottom center of bounding box as player position
                    player_position = np.array([get_foot_position(bbox)])
                    # Transform to tactical view coordinates
                    tactical_position = homography.transform_points(player_position)


                    x, y = tactical_position[0]

                    if not (0 <= x <= self.__class__.length and 0 <= y <= self.__class__.width):
                        continue
                    # If tactical position is not in the tactical view, skip
                    #if tactical_position[0][0] < 0 or tactical_position[0][0] > self.width or tactical_position[0][1] < 0 or tactical_position[0][1] > self.height:
                    #    continue

                    tactical_positions[player_id] = tactical_position[0].tolist()
                    
            except (ValueError, cv2.error) as e:
                # If homography fails, continue with empty dictionary
                pass
            
            tactical_player_positions.append(tactical_positions)
        
        return tactical_player_positions
