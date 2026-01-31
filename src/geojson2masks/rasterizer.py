"""
Polygon rasterization for creating label mask images.

Uses OpenCV for efficient polygon filling.
"""

import numpy as np
import cv2
from typing import Iterator, Optional
from .parser import CellGeometry


def polygon_to_pixel_coords(
    polygon: list[tuple[float, float]]
) -> np.ndarray:
    """
    Convert polygon coordinates to integer pixel coordinates for OpenCV.

    Args:
        polygon: List of (x, y) coordinate tuples

    Returns:
        numpy array of shape (N, 1, 2) with int32 dtype for cv2.fillPoly
    """
    # OpenCV expects coordinates as (x, y) which matches our format
    coords = np.array(polygon, dtype=np.float32)
    # Round to nearest pixel and convert to int32
    coords = np.round(coords).astype(np.int32)
    # Reshape to (N, 1, 2) for cv2.fillPoly
    return coords.reshape((-1, 1, 2))


def create_label_masks(
    cell_geometries: Iterator[CellGeometry],
    width: int,
    height: int,
    total_cells: Optional[int] = None,
    progress_callback: Optional[callable] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create cell and nucleus label masks from cell geometries.

    Args:
        cell_geometries: Iterator of CellGeometry objects
        width: Width of the output mask image
        height: Height of the output mask image
        total_cells: Optional total count for progress reporting
        progress_callback: Optional callback(current, total) for progress

    Returns:
        Tuple of (cell_mask, nucleus_mask) as numpy arrays
        - cell_mask: Label mask for whole cell segmentations
        - nucleus_mask: Label mask for nuclear segmentations
        Both use int32 dtype (OpenCV requirement), supporting up to ~2 billion cells
    """
    # Use int32 for OpenCV compatibility (supports up to ~2 billion unique cell IDs)
    # Note: OpenCV fillPoly requires signed int32, not uint32
    cell_mask = np.zeros((height, width), dtype=np.int32)
    nucleus_mask = np.zeros((height, width), dtype=np.int32)

    processed = 0
    for cell in cell_geometries:
        # Rasterize cell polygon
        cell_coords = polygon_to_pixel_coords(cell.cell_polygon)
        cv2.fillPoly(cell_mask, [cell_coords], color=int(cell.cell_id))

        # Rasterize nucleus polygon if present
        if cell.nucleus_polygon:
            nucleus_coords = polygon_to_pixel_coords(cell.nucleus_polygon)
            cv2.fillPoly(nucleus_mask, [nucleus_coords], color=int(cell.cell_id))

        processed += 1
        if progress_callback and total_cells:
            progress_callback(processed, total_cells)

    return cell_mask, nucleus_mask


def determine_optimal_dtype(max_cell_id: int) -> np.dtype:
    """
    Determine the optimal dtype for storing label masks.

    Args:
        max_cell_id: Maximum cell ID value

    Returns:
        numpy dtype that can accommodate all cell IDs
    """
    if max_cell_id <= 255:
        return np.uint8
    elif max_cell_id <= 65535:
        return np.uint16
    else:
        return np.uint32


def convert_mask_dtype(mask: np.ndarray, max_cell_id: int) -> np.ndarray:
    """
    Convert mask to optimal dtype based on maximum cell ID.

    Args:
        mask: Label mask array
        max_cell_id: Maximum cell ID in the mask

    Returns:
        Mask converted to optimal dtype
    """
    optimal_dtype = determine_optimal_dtype(max_cell_id)
    if mask.dtype != optimal_dtype:
        return mask.astype(optimal_dtype)
    return mask
