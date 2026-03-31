import sys
import pathlib
from typing import Dict, Generator, Iterable, List, TypeVar

import numpy as np
import supervision as sv
import torch
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from utils import read_stub, save_stub

V = TypeVar("V")

SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:
    """
    Generate batches from a sequence with a specified batch size.

    Args:
        sequence (Iterable[V]): The input sequence to be batched.
        batch_size (int): The size of each batch.

    Yields:
        Generator[List[V], None, None]: A generator yielding batches of the input
            sequence.
    """
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    """
    A classifier that uses a pre-trained SiglipVisionModel for feature extraction,
    UMAP for dimensionality reduction, and KMeans for clustering.
    """
    def __init__(self, device: str = 'cpu', batch_size: int = 32):
        """
       Initialize the TeamClassifier with device and batch size.

       Args:
           device (str): The device to run the model on ('cpu' or 'cuda').
           batch_size (int): The batch size for processing images.
       """
        self.device = device
        self.batch_size = batch_size
        self.features_model = SiglipVisionModel.from_pretrained(
            SIGLIP_MODEL_PATH).to(device)
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2)

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a list of image crops using the pre-trained
            SiglipVisionModel.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Extracted features as a numpy array.
        """
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        data = []
        with torch.no_grad():
            for batch in tqdm(batches, desc='Embedding extraction'):
                inputs = self.processor(
                    images=batch, return_tensors="pt").to(self.device)
                outputs = self.features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fit the classifier model on a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.
        """
        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict the cluster labels for a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Predicted cluster labels.
        """
        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        return self.cluster_model.predict(projections)

    @staticmethod
    def _crop_player(frame: np.ndarray, bbox) -> np.ndarray:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        return frame[y1:y2, x1:x2]

    def get_player_teams_across_frames(
        self,
        video_frames: List[np.ndarray],
        player_tracks: List[Dict],
        read_from_stub: bool = False,
        stub_path: str = None,
    ) -> List[Dict]:
        """
        Fit the classifier on all player crops collected from every frame, then
        predict a team (1 or 2) for each player in each frame.

        The first fitting step gathers one crop per unique player across all frames
        so that KMeans sees a representative sample of both teams before prediction.

        Args:
            video_frames (List[np.ndarray]): Video frames in BGR format.
            player_tracks (List[Dict]): Per-frame dicts mapping player_id -> {"bbox": ...}.
            read_from_stub (bool): Load cached result from disk if available.
            stub_path (str): Path to the pickle cache file.

        Returns:
            List[Dict]: Per-frame dicts mapping player_id -> team_id (1 or 2).
        """
        player_assignment = read_stub(read_from_stub, stub_path)
        if player_assignment is not None and len(player_assignment) == len(video_frames):
            return player_assignment

        # --- Collect one representative crop per player for fitting ---
        player_crops: Dict[int, np.ndarray] = {}
        for frame_idx, frame_tracks in enumerate(player_tracks):
            for player_id, track in frame_tracks.items():
                if player_id not in player_crops:
                    crop = self._crop_player(video_frames[frame_idx], track["bbox"])
                    if crop.size > 0:
                        player_crops[player_id] = crop

        fit_ids = list(player_crops.keys())
        fit_crops = [player_crops[pid] for pid in fit_ids]

        print(f"Fitting TeamClassifier on {len(fit_crops)} unique player crops...")
        self.fit(fit_crops)

        # Cluster label (0 or 1) → team_id (1 or 2).
        # Anchor: assign the cluster whose centroid has the smallest index to team 1.
        fit_labels = self.predict(fit_crops)
        cluster_to_team: Dict[int, int] = {
            int(fit_labels[0]): 1,
            int(1 - fit_labels[0]): 2,
        }

        # --- Predict per-frame ---
        player_assignment = []
        for frame_idx, frame_tracks in enumerate(player_tracks):
            frame_result: Dict = {}

            crops, ids = [], []
            for player_id, track in frame_tracks.items():
                crop = self._crop_player(video_frames[frame_idx], track["bbox"])
                if crop.size > 0:
                    crops.append(crop)
                    ids.append(player_id)

            if crops:
                labels = self.predict(crops)
                for player_id, label in zip(ids, labels):
                    frame_result[player_id] = cluster_to_team.get(int(label), 1)

            player_assignment.append(frame_result)

        save_stub(stub_path, player_assignment)
        return player_assignment
