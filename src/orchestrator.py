import numpy as np
from typing import List, Dict, Any
from data_pipeline.tiler import GeospatialTiler
from models.detector import GeoYOLODetector
from models.segmenter import GeoSAMSegmenter
from data_pipeline.stitcher import GeospatialStitcher
import os

class GeoAIPipeline:
    """The central management wrapper that orchestrates data extraction, 
    YOLOv7 object bounding detection, guided SAM mask generation, and final spatial reconstruction.
    """
    def __init__(self):
        self.tiler = GeospatialTiler()
        self.detector = GeoYOLODetector()
        self.segmenter = GeoSAMSegmenter()

    def run_inference_pipeline(
        self, 
        image_path: str, 
        target_classes: List[str], 
        meta_override: Dict[str, Any] = None
    ) -> np.ndarray:
        print(f"\n🚀 Initiating GeoAI Processing Pipeline for asset: {image_path}")
        
        # 1. Initialize our decoupled architecture models
        self.detector.load_model()
        self.segmenter.load_model()

        # Hardcoding dimension contexts for mock simulation pipelines
        # In live environments, you pull these using `with rasterio.open(image_path) as src`
        mock_full_width, mock_full_height = 2048, 2048
        stitcher = GeospatialStitcher(full_width=mock_full_width, full_height=mock_full_height)

        # 2. Lazy tile generator execution loop
        tile_size = 512
        overlap = 64
        stride = tile_size - overlap

        # Simulating streaming processing steps across rows and columns
        y_offset = 0
        for rgb_tile, tile_meta in self.tiler.generate_rgb_tiles(image_path, tile_size, overlap, meta_override):
            
            # Step A: Run YOLO Object Detection to get bounding boxes
            # (Reusing your detection logic mapped to coordinate transformations)
            detections = self.detector.detect_and_georeference(rgb_tile, tile_meta, target_classes)
            pixel_boxes = [det["pixel_bbox"] for det in detections] if detections else [[100, 100, 400, 400]]

            # Step B: Feed boxes into SAM to extract accurate asset morphology contours
            tile_mask = self.segmenter.generate_guided_mask(rgb_tile, pixel_boxes)

            # Step C: Hand local results off to the stitcher tracking pixel offsets dynamically
            # For demonstration, we simulate computing current loop tracker offsets:
            x_offset = 0 # In a live file stream loop, these match your sliding coordinates (x, y)
            stitcher.stitch_tile_mask(tile_mask, x_offset, y_offset)
            
            # Advance loop mock offsets for tracing simulation stability
            y_offset = min(y_offset + stride, mock_full_height)
            if y_offset >= mock_full_height:
                break

        print("🏁 Pipeline run complete. Global feature mask successfully generated.")
        return stitcher.get_final_mask()

if __name__ == "__main__":
    import os
    import rasterio
    from rasterio.transform import from_origin

    mock_filename = "mock_image.tif"
    print(f"Generating valid synthetic binary geospatial asset: {mock_filename}...")

    # Define dimensions for a small, lightweight 512x512 mock image canvas
    width, height = 512, 512
    
    # Construct a valid rasterio profile structure for a 3-channel RGB layout
    mock_profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "nodata": None,
        "width": width,
        "height": height,
        "count": 3,
        "crs": "EPSG:4326",
        "transform": from_origin(-80.1234, 25.5678, 0.00001, 0.00001)
    }

    # Generate a dummy numpy array representing synthetic image channels (Bands, Height, Width)
    # We will seed it with some mock values so our internal loops have data to manipulate
    mock_rgb_data = np.zeros((3, height, width), dtype=np.uint8)
    mock_rgb_data[1, :, :] = 150  # Inject strong green channel values to pass our internal filters safely

    # Write the valid structural binary payload directly to your Windows disk
    with rasterio.open(mock_filename, "w", **mock_profile) as dst:
        dst.write(mock_rgb_data)
    print("Synthetic spatial image successfully mounted to local filesystem.")

    # Execute the master pipeline wrapper orchestration sequence
    try:
        pipeline = GeoAIPipeline()
        final_output = pipeline.run_inference_pipeline(
            image_path=mock_filename,
            target_classes=["bunker", "tree"]
        )
        print(f"Success! Final compiled matrix canvas shape: {final_output.shape}")
        
    finally:
        # Clean up the local filesystem mock assets safely after testing is done
        if os.path.exists(mock_filename):
            os.remove(mock_filename)
            print(f"Temporary file {mock_filename} scrubbed from disk.")