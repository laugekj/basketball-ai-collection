from typing import Optional, List

import cv2
import supervision as sv
import numpy as np

from configs.basketball import BasketballCourtConfiguration


def draw_court(
    config: BasketballCourtConfiguration,
    background_color: sv.Color = sv.Color(196, 164, 132),  # hardwood
    line_color: sv.Color = sv.Color.WHITE,
    padding: int = 50,
    line_thickness: int = 4,
    hoop_radius: int = 6,
    point_radius: int = 5,
    scale: float = 10.0,
    show_labels: bool = False
) -> np.ndarray:
    """
    Draw basketball court using BasketballCourtConfiguration.
    """

    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)

    court = np.ones(
        (scaled_width + 2 * padding,
         scaled_length + 2 * padding, 3),
        dtype=np.uint8
    ) * np.array(background_color.as_bgr(), dtype=np.uint8)

    # Draw edges
    for start, end in config.edges:

        p1 = (
            int(config.vertices[start - 1][0] * scale) + padding,
            int(config.vertices[start - 1][1] * scale) + padding
        )

        p2 = (
            int(config.vertices[end - 1][0] * scale) + padding,
            int(config.vertices[end - 1][1] * scale) + padding
        )

        cv2.line(
            court,
            p1,
            p2,
            line_color.as_bgr(),
            line_thickness
        )

    # Draw hoops
    left_hoop = (
        int(config.vertices[6][0] * scale) + padding,
        int(config.vertices[6][1] * scale) + padding
    )

    right_hoop = (
        int(config.vertices[26][0] * scale) + padding,
        int(config.vertices[26][1] * scale) + padding
    )

    cv2.circle(
        court,
        left_hoop,
        hoop_radius,
        line_color.as_bgr(),
        line_thickness
    )

    cv2.circle(
        court,
        right_hoop,
        hoop_radius,
        line_color.as_bgr(),
        line_thickness
    )

    # Center point
    center = (
        int(config.vertices[16][0] * scale) + padding,
        int(config.vertices[16][1] * scale) + padding
    )

    cv2.circle(
        court,
        center,
        point_radius,
        line_color.as_bgr(),
        -1
    )

    return court


def draw_points_on_court(
    config: BasketballCourtConfiguration,
    xy: np.ndarray,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 10,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 10,
    court: Optional[np.ndarray] = None
) -> np.ndarray:
    if court is None:
        court = draw_court(
            config=config,
            padding=padding,
            scale=scale
        )

    for point in xy:
        scaled_point = (
            int(point[0] * scale) + padding,
            int(point[1] * scale) + padding
        )
        cv2.circle(court, scaled_point, radius, face_color.as_bgr(), -1)
        cv2.circle(court, scaled_point, radius, edge_color.as_bgr(), thickness)

    return court

def draw_paths_on_court(
    config: BasketballCourtConfiguration,
    paths: List[np.ndarray],
    color: sv.Color = sv.Color.WHITE,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 10,
    court: Optional[np.ndarray] = None
) -> np.ndarray:
    if court is None:
        court = draw_court(
            config=config,
            padding=padding,
            scale=scale
        )

    for path in paths:
        scaled_path = [
            (
                int(p[0] * scale) + padding,
                int(p[1] * scale) + padding
            )
            for p in path if p.size > 0
        ]

        for i in range(len(scaled_path) - 1):
            cv2.line(
                court,
                scaled_path[i],
                scaled_path[i + 1],
                color.as_bgr(),
                thickness
            )

    return court


def draw_court_voronoi_diagram(
    config: BasketballCourtConfiguration,
    team_1_xy: np.ndarray,
    team_2_xy: np.ndarray,
    team_1_color: sv.Color = sv.Color.RED,
    team_2_color: sv.Color = sv.Color.BLUE,
    opacity: float = 0.5,
    padding: int = 50,
    scale: float = 10,
    court: Optional[np.ndarray] = None
) -> np.ndarray:
    if court is None:
        court = draw_court(
            config=config,
            padding=padding,
            scale=scale
        )

    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)

    voronoi = np.zeros_like(court, dtype=np.uint8)

    y_coords, x_coords = np.indices((
        scaled_width + 2 * padding,
        scaled_length + 2 * padding
    ))

    y_coords -= padding
    x_coords -= padding

    def distances(xy):
        return np.sqrt(
            (xy[:, 0][:, None, None] * scale - x_coords) ** 2 +
            (xy[:, 1][:, None, None] * scale - y_coords) ** 2
        )

    d1 = np.min(distances(team_1_xy), axis=0)
    d2 = np.min(distances(team_2_xy), axis=0)

    mask = d1 < d2
    voronoi[mask] = team_1_color.as_bgr()
    voronoi[~mask] = team_2_color.as_bgr()

    return cv2.addWeighted(voronoi, opacity, court, 1 - opacity, 0)
