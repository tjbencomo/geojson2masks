"""Unit tests for the parser module."""

import json
import tempfile
from pathlib import Path

import pytest

from segmentation_mask_maker.parser import (
    parse_polygon_coordinates,
    stream_cell_geometries,
    count_cells,
    CellGeometry,
)


class TestParsePolygonCoordinates:
    """Tests for parse_polygon_coordinates function."""

    def test_simple_square(self):
        """Test parsing a simple square polygon."""
        coords = [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]
        result = parse_polygon_coordinates(coords)

        assert len(result) == 5
        assert result[0] == (0.0, 0.0)
        assert result[1] == (10.0, 0.0)
        assert result[2] == (10.0, 10.0)
        assert result[3] == (0.0, 10.0)
        assert result[4] == (0.0, 0.0)

    def test_floating_point_coordinates(self):
        """Test parsing coordinates with floating point values."""
        coords = [[[1.5, 2.5], [3.7, 4.8], [1.5, 2.5]]]
        result = parse_polygon_coordinates(coords)

        assert len(result) == 3
        assert result[0] == (1.5, 2.5)
        assert result[1] == (3.7, 4.8)

    def test_empty_coordinates(self):
        """Test parsing empty coordinates."""
        assert parse_polygon_coordinates([]) == []
        assert parse_polygon_coordinates([[]]) == []

    def test_multipolygon_takes_first_ring(self):
        """Test that only the exterior ring (first element) is used."""
        # Polygon with hole - should only return exterior ring
        coords = [
            [[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]],  # exterior
            [[20, 20], [80, 20], [80, 80], [20, 80], [20, 20]]  # hole (ignored)
        ]
        result = parse_polygon_coordinates(coords)

        assert len(result) == 5
        assert result[0] == (0.0, 0.0)


class TestStreamCellGeometries:
    """Tests for stream_cell_geometries function."""

    def test_stream_simple_cells(self, temp_geojson_file):
        """Test streaming cells from a simple GeoJSON file."""
        cells = list(stream_cell_geometries(str(temp_geojson_file)))

        assert len(cells) == 2
        assert all(isinstance(c, CellGeometry) for c in cells)

    def test_cell_ids_start_at_one(self, temp_geojson_file):
        """Test that cell IDs start at 1 (0 is background)."""
        cells = list(stream_cell_geometries(str(temp_geojson_file)))

        assert cells[0].cell_id == 1
        assert cells[1].cell_id == 2

    def test_cell_ids_are_sequential(self, large_geojson_file):
        """Test that cell IDs are sequential."""
        cells = list(stream_cell_geometries(str(large_geojson_file)))

        ids = [c.cell_id for c in cells]
        assert ids == list(range(1, 101))

    def test_cell_polygon_parsed(self, temp_geojson_file):
        """Test that cell polygons are correctly parsed."""
        cells = list(stream_cell_geometries(str(temp_geojson_file)))

        # First cell should have polygon at (10,10) to (20,20)
        cell = cells[0]
        assert len(cell.cell_polygon) == 5
        assert cell.cell_polygon[0] == (10.0, 10.0)

    def test_nucleus_polygon_parsed(self, temp_geojson_file):
        """Test that nucleus polygons are correctly parsed."""
        cells = list(stream_cell_geometries(str(temp_geojson_file)))

        # First cell has nucleus
        assert cells[0].nucleus_polygon is not None
        assert len(cells[0].nucleus_polygon) == 5
        assert cells[0].nucleus_polygon[0] == (12.0, 12.0)

    def test_cell_without_nucleus(self, temp_geojson_file):
        """Test that cells without nucleus geometry have None."""
        cells = list(stream_cell_geometries(str(temp_geojson_file)))

        # Second cell has no nucleus
        assert cells[1].nucleus_polygon is None

    def test_skips_non_cell_features(self, temp_mixed_geojson_file):
        """Test that non-cell features are skipped."""
        cells = list(stream_cell_geometries(str(temp_mixed_geojson_file)))

        # Should only have 2 cells, not the annotation
        assert len(cells) == 2

    def test_empty_geojson(self):
        """Test handling of empty GeoJSON."""
        geojson = {"type": "FeatureCollection", "features": []}

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.geojson', delete=False
        ) as f:
            json.dump(geojson, f)
            temp_path = f.name

        try:
            cells = list(stream_cell_geometries(temp_path))
            assert len(cells) == 0
        finally:
            Path(temp_path).unlink()


class TestCountCells:
    """Tests for count_cells function."""

    def test_count_simple(self, temp_geojson_file):
        """Test counting cells in a simple file."""
        count = count_cells(str(temp_geojson_file))
        assert count == 2

    def test_count_large(self, large_geojson_file):
        """Test counting cells in a larger file."""
        count = count_cells(str(large_geojson_file))
        assert count == 100

    def test_count_skips_non_cells(self, temp_mixed_geojson_file):
        """Test that count_cells skips non-cell features."""
        count = count_cells(str(temp_mixed_geojson_file))
        assert count == 2

    def test_count_empty(self):
        """Test counting cells in empty GeoJSON."""
        geojson = {"type": "FeatureCollection", "features": []}

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.geojson', delete=False
        ) as f:
            json.dump(geojson, f)
            temp_path = f.name

        try:
            count = count_cells(temp_path)
            assert count == 0
        finally:
            Path(temp_path).unlink()


class TestRealDataFile:
    """Tests using the real test data file."""

    def test_stream_real_data(self, real_test_file):
        """Test streaming from the real test data file."""
        # Just verify we can stream without errors
        cells = list(stream_cell_geometries(str(real_test_file)))
        assert len(cells) == 6446

    def test_count_real_data(self, real_test_file):
        """Test counting cells in the real test data file."""
        count = count_cells(str(real_test_file))
        assert count == 6446

    def test_all_cells_have_geometry(self, real_test_file):
        """Test that all cells have valid geometry."""
        for cell in stream_cell_geometries(str(real_test_file)):
            assert cell.cell_polygon is not None
            assert len(cell.cell_polygon) >= 3

    def test_all_cells_have_nucleus(self, real_test_file):
        """Test that all cells in real data have nucleus geometry."""
        for cell in stream_cell_geometries(str(real_test_file)):
            assert cell.nucleus_polygon is not None
            assert len(cell.nucleus_polygon) >= 3
