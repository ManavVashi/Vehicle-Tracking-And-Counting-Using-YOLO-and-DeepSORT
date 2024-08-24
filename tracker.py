from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
import numpy as np

class Tracker:
    """
    Tracker class that encapsulates the Deep SORT tracking algorithm.
    This class is responsible for initializing the tracker, encoding detections, 
    and updating the tracked objects across video frames.
    """
    tracker = None
    encoder = None
    tracks = None

    def __init__(self):
        """
        Initialize the Tracker object with a pre-trained encoder and set up the tracking parameters.
        """
        # Maximum cosine distance metric for the nearest neighbor matching
        max_cosine_distance = 0.4
        nn_budget = None  # Optional budget for the number of features to retain

        # Path to the pre-trained model used for generating features from detections
        encoder_model_filename = '/home/manav/Documents/Tracker_Version_Problem/model_data/mars-small128.pb'

        # Initialize the Nearest Neighbor Distance Metric using cosine similarity
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        
        # Initialize the Deep SORT tracker with the specified metric
        self.tracker = DeepSortTracker(metric)
        
        # Initialize the encoder for generating features from bounding boxes
        self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)

    def update(self, frame, detections):
        """
        Update the tracker with new detections for the current frame.

        Args:
            frame (numpy.ndarray): The current video frame.
            detections (list): List of detections, where each detection is represented as 
                               [x1, y1, x2, y2, score].
        """
        if len(detections) == 0:
            # If no detections, predict the new positions of tracked objects
            self.tracker.predict()
            self.tracker.update([])  
            self.update_tracks()
            return

        # Convert detections to bounding box format [x, y, width, height]
        bboxes = np.asarray([d[:-1] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]  # Convert from [x1, y1, x2, y2] to [x, y, width, height]
        scores = [d[-1] for d in detections]  # Extract the confidence scores from detections

        # Generate feature embeddings for each bounding box
        features = self.encoder(frame, bboxes)

        # Create Detection objects for each detection in the frame
        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            dets.append(Detection(bbox, scores[bbox_id], features[bbox_id]))

        # Predict the new positions of tracked objects
        self.tracker.predict()
        
        # Update the tracker with the current frame's detections
        self.tracker.update(dets)
        
        # Update the list of active tracks
        self.update_tracks()

    def update_tracks(self):
        """
        Update the list of active tracks with confirmed tracks that have been recently updated.
        """
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue  # Skip tracks that are not confirmed or haven't been updated recently
            
            # Convert the bounding box format to [x1, y1, x2, y2]
            bbox = track.to_tlbr()
            id = track.track_id  # Get the unique track ID
            
            # Append the confirmed and updated track to the active tracks list
            tracks.append(Track(id, bbox))
        
        # Update the tracks attribute with the active tracks
        self.tracks = tracks

class Track:
    """
    Track class to represent a single tracked object.
    """
    track_id = None
    bbox = None

    def __init__(self, id, bbox):
        """
        Initialize the Track object with a unique ID and bounding box.

        Args:
            id (int): Unique track ID.
            bbox (numpy.ndarray): Bounding box coordinates [x1, y1, x2, y2].
        """
        self.track_id = id
        self.bbox = bbox
