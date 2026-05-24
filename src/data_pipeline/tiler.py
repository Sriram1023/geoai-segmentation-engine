import os
from typing import Generator, Tuple, Dict, Any, Optional
import numpy as np
import rasterio
from rasterio.windows import Window

class GeospatialTiler:
    """Handles memory-efficient tiling of standard RGB imagery while preserving 
    geospatial coordinate systems and reformatting data for Computer Vision models.
    """
    def __init__(self):
        pass

    def generate_rgb_tiles(
        self, 
        src_path: str, 
        tile_size: int = 512, 
        overlap: int = 64,
        meta_override: Optional[Dict[str, Any]] = None
    ) -> Generator[Tuple[np.ndarray, Dict[str, Any]], None, None]:
        """Lazy generator that yields sliced image chunks as standard CV-ready RGB arrays 
        (Height, Width, 3) alongside their updated geographical metadata profiles.
        """
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Source file not found at: {src_path}")

        stride = tile_size - overlap
        if stride <= 0:
            raise ValueError("Overlap metric cannot be equal to or greater than the tile size.")

        # Determine if file is a native geospatial raster
        is_tiff = src_path.lower().endswith(('.tif', '.tiff'))

        with rasterio.open(src_path) as src:
            width = src.width
            height = src.height

            # 1. Establish the correct base spatial metadata and root transform matrix
            if is_tiff:
                base_meta = src.meta.copy()
                root_transform = src.transform
            else:
                if not meta_override or 'crs' not in meta_override or 'bbox' not in meta_override:
                    raise ValueError("For non-TIFF sources, meta_override must include 'crs' and 'bbox' information.")
                
                # FIX: Calculate the transform using full image dimensions, not tile_size
                root_transform = rasterio.transform.from_bounds(
                    *meta_override['bbox'], 
                    width=width, 
                    height=height
                )
                base_meta = {
                    "crs": meta_override['crs'],
                    "transform": root_transform,
                    "dtype": src.dtypes[0] if src.dtypes else 'uint8'
                }

            # Explicitly lock down standard 3-channel configuration profiles
            base_meta.update({
                "count": 3,
                "driver": "GTiff"
            })

            # Slide our window bounds across the primary dimensions
            for y in range(0, height, stride):
                for x in range(0, width, stride):
                    
                    w_width = min(tile_size, width - x)
                    w_height = min(tile_size, height - y)
                    
                    window = Window(x, y, w_width, w_height)
                    
                    # 2. Defensive handling for channel index extraction
                    if src.count >= 3:
                        tile_data = src.read(indexes=(1, 2, 3), window=window)
                    else:
                        # Fallback for single channel or anomaly assets: broadcast to 3 channels
                        raw_read = src.read(window=window)
                        tile_data = np.repeat(raw_read[:1, :, :], 3, axis=0)
                    
                    # Shift axes from (Bands, H, W) -> (H, W, Bands) for computer vision
                    rgb_cv_ready = np.transpose(tile_data, (1, 2, 0))
                    
                    # 3. FIX: Calculate shifting transform utilizing the true root geospatial transform
                    tile_transform = rasterio.windows.transform(window, root_transform)
                    
                    # Clone and package metadata context for this specific chip
                    tile_profile = base_meta.copy()
                    tile_profile.update({
                        "height": w_height,
                        "width": w_width,
                        "transform": tile_transform
                    })
                    
                    yield rgb_cv_ready, tile_profile

if __name__ == "__main__":
    print("GeospatialTiler optimized for standard RGB pipelines initialized.")