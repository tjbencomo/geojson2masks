"""Unit tests for the rasterizer module."""

import numpy as np
import pytest

from segmentation_mask_maker.parser import CellGeometry
from segmentation_mask_maker.rasterizer import (
    polygon_to_pixel_coords,
    create_label_masks,
    determine_optimal_dtype,
    convert_mask_dtype,
)


class TestPolygonToPixelCoords:
    """Tests for polygon_to_pixel_coords function."""

    def test_integer_coordinates(self):
        """Test conversion of integer coordinates."""
        polygon = [(10, 20), (30, 20), (30, 40), (10, 40)]
        result = polygon_to_pixel_coords(polygon)

        assert result.dtype == np.int32
        assert result.shape == (4, 1, 2)
        np.testing.assert_array_equal(result[0, 0], [10, 20])
        np.testing.assert_array_equal(result[1, 0], [30, 20])

    def test_floating_point_rounds(self):
        """Test that floating point coordinates are rounded."""
        polygon = [(10.4, 20.6), (30.5, 20.4)]
        result = polygon_to_pixel_coords(polygon)

        # 10.4 -> 10, 20.6 -> 21, 30.5 -> 30 (or 31), 20.4 -> 20
        assert result.dtype == np.int32
        np.testing.assert_array_equal(result[0, 0], [10, 21])
        np.testing.assert_array_equal(result[1, 0], [30, 20])

    def test_shape_for_opencv(self):
        """Test that output shape is correct for cv2.fillPoly."""
        polygon = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        result = polygon_to_pixel_coords(polygon)

        # OpenCV expects (N, 1, 2) for fillPoly
        assert result.shape == (5, 1, 2)


class TestCreateLabelMasks:
    """Tests for create_label_masks function."""

    def test_single_cell(self):
        """Test creating masks with a single cell."""
        cells = [
            CellGeometry(
                cell_id=1,
                cell_polygon=[(10, 10), (20, 10), (20, 20), (10, 20), (10, 10)],
                nucleus_polygon=[(12, 12), (18, 12), (18, 18), (12, 18), (12, 12)]
            )
        ]

        cell_mask, nucleus_mask = create_label_masks(iter(cells), width=50, height=50)

        assert cell_mask.shape == (50, 50)
        assert nucleus_mask.shape == (50, 50)
        assert cell_mask.max() == 1
        assert nucleus_mask.max() == 1

    def test_cell_ids_in_mask(self):
        """Test that cell IDs are correctly written to masks."""
        cells = [
            CellGeometry(
                cell_id=1,
                cell_polygon=[(5, 5), (15, 5), (15, 15), (5, 15), (5, 5)],
                nucleus_polygon=[(7, 7), (13, 7), (13, 13), (7, 13), (7, 7)]
            ),
            CellGeometry(
                cell_id=2,
                cell_polygon=[(25, 25), (35, 25), (35, 35), (25, 35), (25, 25)],
                nucleus_polygon=[(27, 27), (33, 27), (33, 33), (27, 33), (27, 27)]
            )
        ]

        cell_mask, nucleus_mask = create_label_masks(iter(cells), width=50, height=50)

        # Check cell 1 region
        assert cell_mask[10, 10] == 1
        assert nucleus_mask[10, 10] == 1

        # Check cell 2 region
        assert cell_mask[30, 30] == 2
        assert nucleus_mask[30, 30] == 2

        # Check background
        assert cell_mask[0, 0] == 0
        assert nucleus_mask[0, 0] == 0

    def test_cell_without_nucleus(self):
        """Test cell without nucleus geometry."""
        cells = [
            CellGeometry(
                cell_id=1,
                cell_polygon=[(10, 10), (20, 10), (20, 20), (10, 20), (10, 10)],
                nucleus_polygon=None
            )
        ]

        cell_mask, nucleus_mask = create_label_masks(iter(cells), width=50, height=50)

        # Cell mask should have the cell
        assert cell_mask[15, 15] == 1

        # Nucleus mask should be empty
        assert nucleus_mask[15, 15] == 0
        assert nucleus_mask.max() == 0

    def test_progress_callback(self):
        """Test that progress callback is called."""
        cells = [
            CellGeometry(
                cell_id=i,
                cell_polygon=[(0, 0), (5, 0), (5, 5), (0, 5), (0, 0)],
                nucleus_polygon=None
            )
            for i in range(1, 11)
        ]

        progress_values = []

        def callback(current, total):
            progress_values.append((current, total))

        create_label_masks(
            iter(cells),
            width=50,
            height=50,
            total_cells=10,
            progress_callback=callback
        )

        assert len(progress_values) == 10
        assert progress_values[-1] == (10, 10)

    def test_mask_dtype_is_int32(self):
        """Test that masks use int32 dtype (OpenCV requirement)."""
        cells = [
            CellGeometry(
                cell_id=1,
                cell_polygon=[(10, 10), (20, 10), (20, 20), (10, 20), (10, 10)],
                nucleus_polygon=[(12, 12), (18, 12), (18, 18), (12, 18), (12, 12)]
            )
        ]

        cell_mask, nucleus_mask = create_label_masks(iter(cells), width=50, height=50)

        assert cell_mask.dtype == np.int32
        assert nucleus_mask.dtype == np.int32

    def test_large_cell_ids(self):
        """Test handling of large cell IDs."""
        cells = [
            CellGeometry(
                cell_id=100000,
                cell_polygon=[(10, 10), (20, 10), (20, 20), (10, 20), (10, 10)],
                nucleus_polygon=[(12, 12), (18, 12), (18, 18), (12, 18), (12, 12)]
            )
        ]

        cell_mask, nucleus_mask = create_label_masks(iter(cells), width=50, height=50)

        assert cell_mask[15, 15] == 100000
        assert nucleus_mask[15, 15] == 100000

    def test_overlapping_cells(self):
        """Test that later cells overwrite earlier ones in overlapping regions."""
        cells = [
            CellGeometry(
                cell_id=1,
                cell_polygon=[(10, 10), (30, 10), (30, 30), (10, 30), (10, 10)],
                nucleus_polygon=None
            ),
            CellGeometry(
                cell_id=2,
                cell_polygon=[(20, 20), (40, 20), (40, 40), (20, 40), (20, 40)],
                nucleus_polygon=None
            )
        ]

        cell_mask, _ = create_label_masks(iter(cells), width=50, height=50)

        # Non-overlapping region of cell 1
        assert cell_mask[15, 15] == 1

        # Overlapping region - cell 2 should overwrite
        assert cell_mask[25, 25] == 2

        # Cell 2 only region
        assert cell_mask[35, 35] == 2


class TestDetermineOptimalDtype:
    """Tests for determine_optimal_dtype function."""

    def test_uint8_range(self):
        """Test that small values use uint8."""
        assert determine_optimal_dtype(0) == np.uint8
        assert determine_optimal_dtype(1) == np.uint8
        assert determine_optimal_dtype(255) == np.uint8

    def test_uint16_range(self):
        """Test that medium values use uint16."""
        assert determine_optimal_dtype(256) == np.uint16
        assert determine_optimal_dtype(1000) == np.uint16
        assert determine_optimal_dtype(65535) == np.uint16

    def test_uint32_range(self):
        """Test that large values use uint32."""
        assert determine_optimal_dtype(65536) == np.uint32
        assert determine_optimal_dtype(100000) == np.uint32
        assert determine_optimal_dtype(1000000) == np.uint32


class TestConvertMaskDtype:
    """Tests for convert_mask_dtype function."""

    def test_convert_to_uint8(self):
        """Test conversion to uint8."""
        mask = np.array([[1, 2], [3, 4]], dtype=np.int32)
        result = convert_mask_dtype(mask, max_cell_id=4)

        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, mask)

    def test_convert_to_uint16(self):
        """Test conversion to uint16."""
        mask = np.array([[1, 256], [1000, 0]], dtype=np.int32)
        result = convert_mask_dtype(mask, max_cell_id=1000)

        assert result.dtype == np.uint16
        np.testing.assert_array_equal(result, mask)

    def test_convert_to_uint32(self):
        """Test conversion to uint32."""
        mask = np.array([[1, 100000], [0, 0]], dtype=np.int32)
        result = convert_mask_dtype(mask, max_cell_id=100000)

        assert result.dtype == np.uint32
        np.testing.assert_array_equal(result, mask)

    def test_no_conversion_needed(self):
        """Test that no conversion happens when dtype matches."""
        mask = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        result = convert_mask_dtype(mask, max_cell_id=4)

        assert result.dtype == np.uint8
        assert result is mask  # Same object, no copy
