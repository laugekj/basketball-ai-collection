from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class BasketballCourtConfiguration:
    width: int = 50   
    length: int = 94  
    
    key_length: int = 19
    key_width: int = 16

    three_point_distance: int = 28 # from baseline to top of the arc
    three_point_margin: int = 3 # distance from the three point line to the court outline
    three_point_line_length: int = 14 # length of the straight part of the three point line on the sides
    
    hoop_distance: int = 5.3
    

    @property
    def vertices(self) -> List[Tuple[int, int]]:
        w = self.width
        l = self.length
        kw = self.key_width
        kl = self.key_length
        tpd = self.three_point_distance
        tpm = self.three_point_margin
        hd = self.hoop_distance
        tpll = self.three_point_line_length

        return [
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

    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), # Left baseline
        (1, 13), (13, 16), (16, 19), (19, 28), # Top sideline
        (6, 15), (15, 18), (18, 21), (21, 33), # Bottom sideline
        (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), # Right baseline
        (16, 17), (17, 18), # Middle line
        
        (2, 8), (8, 14), (14, 9), (9, 5), # Left 3pt arc
        (3, 10), (10, 11), (11, 12), (12, 4), # Left key
        
        (29, 25), (25, 20), (20, 26), (26, 32), # Right 3pt arc
        (30, 22), (22, 23), (23, 24), (24, 31) # Right key
    ])

    labels: List[str] = field(default_factory=lambda: [
        "1", "2", "3", "4", "5",
        "6", "7", "8", "9", "10",
        "11",
        "12", "13", "14", "15", "16",
        "17", "18", "19", "20", "21",
        "22", "23", "24", "25", "26",
        "27", "28", "29", "30", "31",
        "32", "33"
    ])

    colors: List[str] = field(default_factory=lambda: [
        "#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF",
        "#00BFFF", "#00BFFF",
        "#FFD700", "#FFD700",
        "#FF6347", "#FF6347", "#FF6347", "#FF6347",
        "#32CD32", "#32CD32", "#32CD32", "#32CD32",
        "#FF1493", "#FF1493"
    ])
