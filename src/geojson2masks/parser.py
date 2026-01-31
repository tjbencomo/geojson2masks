"""
Streaming GeoJSON parser for QuPath segmentation exports.

Uses ijson for memory-efficient parsing of large GeoJSON files.
"""

import ijson
from typing import Iterator, Optional
from dataclasses import dataclass


@dataclass
class CellGeometry:
    """Container for a single cell's geometry data."""
    cell_id: int
    cell_polygon: list[tuple[float, float]]
    nucleus_polygon: Optional[list[tuple[float, float]]]


def parse_polygon_coordinates(coords: list) -> list[tuple[float, float]]:
    """
    Parse GeoJSON polygon coordinates to list of (x, y) tuples.

    GeoJSON polygons have structure: [[[x1,y1], [x2,y2], ...]]
    The outer list is for multi-polygons, we take the first ring.
    """
    if not coords or not coords[0]:
        return []

    # Take the exterior ring (first element)
    exterior_ring = coords[0]
    return [(float(pt[0]), float(pt[1])) for pt in exterior_ring]


def stream_cell_geometries(geojson_path: str) -> Iterator[CellGeometry]:
    """
    Stream cell geometries from a QuPath GeoJSON export file.

    This function uses ijson to parse the file incrementally,
    avoiding loading the entire file into memory.

    Args:
        geojson_path: Path to the GeoJSON file

    Yields:
        CellGeometry objects for each cell in the file
    """
    cell_id = 1  # Start cell IDs at 1 (0 is background in label masks)

    with open(geojson_path, 'rb') as f:
        # Use ijson to iterate over features array
        # The prefix 'features.item' matches each element in the features array
        parser = ijson.items(f, 'features.item')

        for feature in parser:
            # Skip non-cell objects
            properties = feature.get('properties', {})
            if properties.get('objectType') != 'cell':
                continue

            # Parse cell (whole cell) geometry
            geometry = feature.get('geometry', {})
            cell_coords = geometry.get('coordinates', [])
            cell_polygon = parse_polygon_coordinates(cell_coords)

            if not cell_polygon:
                continue

            # Parse nucleus geometry (may not exist)
            nucleus_geometry = feature.get('nucleusGeometry', {})
            nucleus_coords = nucleus_geometry.get('coordinates', [])
            nucleus_polygon = parse_polygon_coordinates(nucleus_coords) if nucleus_coords else None

            yield CellGeometry(
                cell_id=cell_id,
                cell_polygon=cell_polygon,
                nucleus_polygon=nucleus_polygon
            )

            cell_id += 1


def count_cells(geojson_path: str) -> int:
    """
    Count the number of cells in a GeoJSON file.

    Uses streaming to avoid loading entire file into memory.

    Args:
        geojson_path: Path to the GeoJSON file

    Returns:
        Number of cell objects in the file
    """
    count = 0
    with open(geojson_path, 'rb') as f:
        parser = ijson.items(f, 'features.item.properties')
        for props in parser:
            if props.get('objectType') == 'cell':
                count += 1
    return count
