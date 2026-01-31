"""Pytest fixtures for segmentation-mask-maker tests."""

import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def sample_cell_feature():
    """A single cell feature with both cell and nucleus geometry."""
    return {
        "type": "Feature",
        "id": "test-cell-001",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[10, 10], [20, 10], [20, 20], [10, 20], [10, 10]]]
        },
        "nucleusGeometry": {
            "type": "Polygon",
            "coordinates": [[[12, 12], [18, 12], [18, 18], [12, 18], [12, 12]]]
        },
        "properties": {
            "objectType": "cell"
        }
    }


@pytest.fixture
def sample_cell_no_nucleus():
    """A cell feature without nucleus geometry."""
    return {
        "type": "Feature",
        "id": "test-cell-002",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[30, 30], [40, 30], [40, 40], [30, 40], [30, 30]]]
        },
        "properties": {
            "objectType": "cell"
        }
    }


@pytest.fixture
def sample_non_cell_feature():
    """A non-cell feature (e.g., annotation)."""
    return {
        "type": "Feature",
        "id": "test-annotation-001",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]]]
        },
        "properties": {
            "objectType": "annotation"
        }
    }


@pytest.fixture
def simple_geojson(sample_cell_feature, sample_cell_no_nucleus):
    """A simple GeoJSON FeatureCollection with two cells."""
    return {
        "type": "FeatureCollection",
        "features": [sample_cell_feature, sample_cell_no_nucleus]
    }


@pytest.fixture
def mixed_geojson(sample_cell_feature, sample_cell_no_nucleus, sample_non_cell_feature):
    """A GeoJSON with cells and non-cell features."""
    return {
        "type": "FeatureCollection",
        "features": [
            sample_cell_feature,
            sample_non_cell_feature,
            sample_cell_no_nucleus
        ]
    }


@pytest.fixture
def temp_geojson_file(simple_geojson):
    """Create a temporary GeoJSON file."""
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.geojson', delete=False
    ) as f:
        json.dump(simple_geojson, f)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_mixed_geojson_file(mixed_geojson):
    """Create a temporary GeoJSON file with mixed features."""
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.geojson', delete=False
    ) as f:
        json.dump(mixed_geojson, f)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def real_test_file():
    """Path to the real test data file if it exists."""
    test_file = Path(__file__).parent.parent / "data" / "test_export_data.geojson"
    if test_file.exists():
        return test_file
    pytest.skip("Real test data file not available")


def create_geojson_with_cells(num_cells: int, include_nuclei: bool = True) -> dict:
    """Helper to create a GeoJSON with a specified number of cells."""
    features = []
    for i in range(num_cells):
        x_offset = (i % 10) * 50
        y_offset = (i // 10) * 50

        feature = {
            "type": "Feature",
            "id": f"cell-{i:04d}",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [x_offset + 10, y_offset + 10],
                    [x_offset + 40, y_offset + 10],
                    [x_offset + 40, y_offset + 40],
                    [x_offset + 10, y_offset + 40],
                    [x_offset + 10, y_offset + 10]
                ]]
            },
            "properties": {
                "objectType": "cell"
            }
        }

        if include_nuclei:
            feature["nucleusGeometry"] = {
                "type": "Polygon",
                "coordinates": [[
                    [x_offset + 15, y_offset + 15],
                    [x_offset + 35, y_offset + 15],
                    [x_offset + 35, y_offset + 35],
                    [x_offset + 15, y_offset + 35],
                    [x_offset + 15, y_offset + 15]
                ]]
            }

        features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features
    }


@pytest.fixture
def large_geojson_file():
    """Create a temporary GeoJSON file with 100 cells."""
    geojson = create_geojson_with_cells(100)

    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.geojson', delete=False
    ) as f:
        json.dump(geojson, f)
        temp_path = Path(f.name)

    yield temp_path

    if temp_path.exists():
        temp_path.unlink()
