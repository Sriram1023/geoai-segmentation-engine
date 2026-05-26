import numpy as np
from typing import Dict, Any

class GeospatialStitcher:
    """Manages the reconstruction and stitching of isolated tile predictions 
    back into a continuous global array layout tracking original coordinates.
    """
    def __init__(self, full_width: int, full_height: int):
        self.full_width = full_width
        self.full_height = full_height
        # Allocate a single continuous matrix canvas for the final unified mask output
        self.global_mask = np.zeros((full_height, full_width), dtype=np.uint8)

    def stitch_tile_mask(self, local_mask: np.ndarray, x_offset: int, y_offset: int) -> None:
        """Places a local chunk prediction matrix back into its exact position on the global canvas.

        Args:
            local_mask: Boolean/Binary matrix slice from model output.
            x_offset: Origin X pixel column position of the tile in the master asset.
            y_offset: Origin Y pixel row position of the tile in the master asset.
        """
        h, w = local_mask.shape
        
        # Determine strict placement limits to prevent array index clipping at boundaries
        target_h = min(h, self.full_height - y_offset)
        target_w = min(w, self.full_width - x_offset)

        # Merge local predictions to the global array canvas using a bitwise OR operation
        self.global_mask[y_offset : y_offset + target_h, x_offset : x_offset + target_w] |= local_mask[:target_h, :target_w].astype(np.uint8)

    def get_final_mask(self) -> np.ndarray:
        """Returns the completely assembled global reconstruction array."""
        return self.global_mask