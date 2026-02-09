"""Unit tests for the ID mapping functionality."""

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

from geojson2masks.parser import (
    stream_cell_geometries,
    build_id_mapping,
    CellGeometry,
)
from geojson2masks.cli import main


class TestGeojsonIdParsing:
    """Tests for GeoJSON feature ID extraction in stream_cell_geometries."""

    def test_geojson_id_extracted(self, temp_geojson_file):
        """Test that geojson_id is extracted from features."""
        cells = list(stream_cell_geometries(str(temp_geojson_file)))

        assert cells[0].geojson_id == "test-cell-001"
        assert cells[1].geojson_id == "test-cell-002"

    def test_geojson_id_none_when_missing(self):
        """Test that geojson_id is None when feature has no id field."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[10, 10], [20, 10], [20, 20], [10, 20], [10, 10]]]
                    },
                    "properties": {"objectType": "cell"}
                }
            ]
        }

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.geojson', delete=False
        ) as f:
            json.dump(geojson, f)
            temp_path = f.name

        try:
            cells = list(stream_cell_geometries(temp_path))
            assert len(cells) == 1
            assert cells[0].geojson_id is None
        finally:
            Path(temp_path).unlink()

    def test_geojson_id_with_uuid_format(self):
        """Test extraction of UUID-formatted IDs (typical QuPath format)."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": "3c16dc89-fff3-4c18-bcd8-22597c736198",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[10, 10], [20, 10], [20, 20], [10, 20], [10, 10]]]
                    },
                    "properties": {"objectType": "cell"}
                }
            ]
        }

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.geojson', delete=False
        ) as f:
            json.dump(geojson, f)
            temp_path = f.name

        try:
            cells = list(stream_cell_geometries(temp_path))
            assert cells[0].geojson_id == "3c16dc89-fff3-4c18-bcd8-22597c736198"
        finally:
            Path(temp_path).unlink()

    def test_geojson_id_skipped_for_non_cells(self, temp_mixed_geojson_file):
        """Test that non-cell feature IDs are not included."""
        cells = list(stream_cell_geometries(str(temp_mixed_geojson_file)))

        # Only 2 cells, non-cell annotation is skipped
        assert len(cells) == 2
        geojson_ids = [c.geojson_id for c in cells]
        assert "test-annotation-001" not in geojson_ids

    def test_geojson_id_in_real_data(self, real_test_file):
        """Test that GeoJSON IDs are extracted from real data."""
        cells = list(stream_cell_geometries(str(real_test_file)))
        # Real data should have UUID-format IDs
        assert cells[0].geojson_id == "3c16dc89-fff3-4c18-bcd8-22597c736198"
        # All cells should have non-None IDs
        assert all(c.geojson_id is not None for c in cells)


class TestBuildIdMapping:
    """Tests for build_id_mapping function."""

    def test_basic_mapping(self):
        """Test basic ID mapping construction."""
        cells = [
            CellGeometry(
                cell_id=1,
                geojson_id="uuid-001",
                cell_polygon=[(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
                nucleus_polygon=None
            ),
            CellGeometry(
                cell_id=2,
                geojson_id="uuid-002",
                cell_polygon=[(20, 20), (30, 20), (30, 30), (20, 30), (20, 20)],
                nucleus_polygon=None
            ),
        ]

        mapping = build_id_mapping(iter(cells))

        assert len(mapping) == 2
        assert mapping[0] == (1, "uuid-001")
        assert mapping[1] == (2, "uuid-002")

    def test_mapping_with_none_geojson_id(self):
        """Test that None geojson_id becomes empty string in mapping."""
        cells = [
            CellGeometry(
                cell_id=1,
                geojson_id=None,
                cell_polygon=[(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
                nucleus_polygon=None
            ),
        ]

        mapping = build_id_mapping(iter(cells))

        assert mapping[0] == (1, "")

    def test_mapping_empty_iterator(self):
        """Test mapping from empty iterator."""
        mapping = build_id_mapping(iter([]))
        assert mapping == []

    def test_mapping_preserves_order(self):
        """Test that mapping preserves cell order."""
        cells = [
            CellGeometry(
                cell_id=i,
                geojson_id=f"id-{i}",
                cell_polygon=[(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
                nucleus_polygon=None
            )
            for i in range(1, 6)
        ]

        mapping = build_id_mapping(iter(cells))

        assert len(mapping) == 5
        for i, (mask_id, geojson_id) in enumerate(mapping, start=1):
            assert mask_id == i
            assert geojson_id == f"id-{i}"

    def test_mapping_from_file(self, temp_geojson_file):
        """Test building mapping from a parsed GeoJSON file."""
        cells = stream_cell_geometries(str(temp_geojson_file))
        mapping = build_id_mapping(cells)

        assert len(mapping) == 2
        assert mapping[0] == (1, "test-cell-001")
        assert mapping[1] == (2, "test-cell-002")


class TestIdMappingCsvOutput:
    """Tests for ID mapping CSV file output via CLI."""

    def test_csv_file_created(self, temp_geojson_file, temp_output_dir):
        """Test that ID mapping CSV file is created."""
        result = main([
            str(temp_geojson_file),
            '-W', '50',
            '-H', '50',
            '-o', str(temp_output_dir),
            '-q'
        ])

        assert result == 0

        base_name = temp_geojson_file.stem
        id_mapping_path = temp_output_dir / f"{base_name}_id_mapping.csv"
        assert id_mapping_path.exists()

    def test_csv_header(self, temp_geojson_file, temp_output_dir):
        """Test that CSV has correct header row."""
        main([
            str(temp_geojson_file),
            '-W', '50',
            '-H', '50',
            '-o', str(temp_output_dir),
            '-q'
        ])

        base_name = temp_geojson_file.stem
        id_mapping_path = temp_output_dir / f"{base_name}_id_mapping.csv"

        with open(id_mapping_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)

        assert header == ['mask_id', 'geojson_id']

    def test_csv_content(self, temp_geojson_file, temp_output_dir):
        """Test that CSV contains correct mapping data."""
        main([
            str(temp_geojson_file),
            '-W', '50',
            '-H', '50',
            '-o', str(temp_output_dir),
            '-q'
        ])

        base_name = temp_geojson_file.stem
        id_mapping_path = temp_output_dir / f"{base_name}_id_mapping.csv"

        with open(id_mapping_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0] == ['1', 'test-cell-001']
        assert rows[1] == ['2', 'test-cell-002']

    def test_csv_row_count_matches_cells(self, large_geojson_file, temp_output_dir):
        """Test that CSV has one row per cell."""
        main([
            str(large_geojson_file),
            '-W', '600',
            '-H', '600',
            '-o', str(temp_output_dir),
            '-q'
        ])

        base_name = large_geojson_file.stem
        id_mapping_path = temp_output_dir / f"{base_name}_id_mapping.csv"

        with open(id_mapping_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            rows = list(reader)

        assert len(rows) == 100

    def test_csv_mask_ids_match_mask(self, temp_geojson_file, temp_output_dir):
        """Test that mask IDs in CSV match those in the actual mask."""
        main([
            str(temp_geojson_file),
            '-W', '50',
            '-H', '50',
            '-o', str(temp_output_dir),
            '-q'
        ])

        base_name = temp_geojson_file.stem
        cell_mask = tifffile.imread(temp_output_dir / f"{base_name}_cell_mask.tif")
        id_mapping_path = temp_output_dir / f"{base_name}_id_mapping.csv"

        with open(id_mapping_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            csv_mask_ids = {int(row[0]) for row in reader}

        mask_ids = set(np.unique(cell_mask)) - {0}
        assert csv_mask_ids == mask_ids

    def test_csv_skips_non_cells(self, temp_mixed_geojson_file, temp_output_dir):
        """Test that CSV only contains cell entries, not annotations."""
        main([
            str(temp_mixed_geojson_file),
            '-W', '50',
            '-H', '50',
            '-o', str(temp_output_dir),
            '-q'
        ])

        base_name = temp_mixed_geojson_file.stem
        id_mapping_path = temp_output_dir / f"{base_name}_id_mapping.csv"

        with open(id_mapping_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            rows = list(reader)

        # Only 2 cells, not the annotation
        assert len(rows) == 2
        geojson_ids = [row[1] for row in rows]
        assert "test-annotation-001" not in geojson_ids

    def test_csv_with_missing_geojson_ids(self, temp_output_dir):
        """Test CSV output when features have no id field."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[10, 10], [20, 10], [20, 20], [10, 20], [10, 10]]]
                    },
                    "properties": {"objectType": "cell"}
                }
            ]
        }

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.geojson', delete=False
        ) as f:
            json.dump(geojson, f)
            temp_path = Path(f.name)

        try:
            main([
                str(temp_path),
                '-W', '50',
                '-H', '50',
                '-o', str(temp_output_dir),
                '-q'
            ])

            id_mapping_path = temp_output_dir / f"{temp_path.stem}_id_mapping.csv"

            with open(id_mapping_path, 'r') as csvf:
                reader = csv.reader(csvf)
                next(reader)  # skip header
                rows = list(reader)

            assert len(rows) == 1
            assert rows[0] == ['1', '']
        finally:
            temp_path.unlink()

    def test_csv_output_in_default_dir(self, temp_geojson_file):
        """Test that CSV is created in default output directory."""
        result = main([
            str(temp_geojson_file),
            '-W', '50',
            '-H', '50',
            '-q'
        ])

        assert result == 0

        base_name = temp_geojson_file.stem
        parent_dir = temp_geojson_file.parent
        id_mapping_path = parent_dir / f"{base_name}_id_mapping.csv"

        assert id_mapping_path.exists()

        # Cleanup
        id_mapping_path.unlink()
        (parent_dir / f"{base_name}_cell_mask.tif").unlink()
        (parent_dir / f"{base_name}_nucleus_mask.tif").unlink()


class TestIdMappingRealData:
    """Integration tests for ID mapping with real test data."""

    def test_real_data_csv_created(self, real_test_file, temp_output_dir):
        """Test that CSV is created with real data."""
        main([
            str(real_test_file),
            '-W', '15000',
            '-H', '3000',
            '-o', str(temp_output_dir),
            '-q'
        ])

        base_name = real_test_file.stem
        id_mapping_path = temp_output_dir / f"{base_name}_id_mapping.csv"
        assert id_mapping_path.exists()

    def test_real_data_csv_row_count(self, real_test_file, temp_output_dir):
        """Test that CSV has correct number of rows for real data."""
        main([
            str(real_test_file),
            '-W', '15000',
            '-H', '3000',
            '-o', str(temp_output_dir),
            '-q'
        ])

        base_name = real_test_file.stem
        id_mapping_path = temp_output_dir / f"{base_name}_id_mapping.csv"

        with open(id_mapping_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            rows = list(reader)

        assert len(rows) == 6446

    def test_real_data_csv_ids_are_sequential(self, real_test_file, temp_output_dir):
        """Test that mask IDs in CSV are sequential starting from 1."""
        main([
            str(real_test_file),
            '-W', '15000',
            '-H', '3000',
            '-o', str(temp_output_dir),
            '-q'
        ])

        base_name = real_test_file.stem
        id_mapping_path = temp_output_dir / f"{base_name}_id_mapping.csv"

        with open(id_mapping_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            mask_ids = [int(row[0]) for row in reader]

        assert mask_ids == list(range(1, 6447))

    def test_real_data_csv_has_uuid_ids(self, real_test_file, temp_output_dir):
        """Test that GeoJSON IDs in CSV are non-empty UUIDs."""
        main([
            str(real_test_file),
            '-W', '15000',
            '-H', '3000',
            '-o', str(temp_output_dir),
            '-q'
        ])

        base_name = real_test_file.stem
        id_mapping_path = temp_output_dir / f"{base_name}_id_mapping.csv"

        with open(id_mapping_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            geojson_ids = [row[1] for row in reader]

        # All should be non-empty
        assert all(gid != '' for gid in geojson_ids)
        # All should be unique
        assert len(set(geojson_ids)) == 6446
