"""Integration tests for the CLI module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

from segmentation_mask_maker.cli import main, parse_args


class TestParseArgs:
    """Tests for argument parsing."""

    def test_required_args(self):
        """Test that required arguments are enforced."""
        with pytest.raises(SystemExit):
            parse_args(['input.geojson'])  # Missing width and height

        with pytest.raises(SystemExit):
            parse_args(['input.geojson', '-W', '100'])  # Missing height

        with pytest.raises(SystemExit):
            parse_args(['input.geojson', '-H', '100'])  # Missing width

    def test_minimal_args(self):
        """Test parsing with minimal required arguments."""
        args = parse_args(['input.geojson', '-W', '100', '-H', '200'])

        assert args.input == Path('input.geojson')
        assert args.width == 100
        assert args.height == 200
        assert args.output_dir is None
        assert args.cell_suffix == '_cell_mask'
        assert args.nucleus_suffix == '_nucleus_mask'
        assert args.no_compress is False
        assert args.quiet is False

    def test_all_args(self):
        """Test parsing with all arguments."""
        args = parse_args([
            'input.geojson',
            '-W', '1000',
            '-H', '2000',
            '-o', '/output/dir',
            '--cell-suffix', '_cells',
            '--nucleus-suffix', '_nuclei',
            '--no-compress',
            '-q'
        ])

        assert args.input == Path('input.geojson')
        assert args.width == 1000
        assert args.height == 2000
        assert args.output_dir == Path('/output/dir')
        assert args.cell_suffix == '_cells'
        assert args.nucleus_suffix == '_nuclei'
        assert args.no_compress is True
        assert args.quiet is True

    def test_long_form_args(self):
        """Test parsing with long-form arguments."""
        args = parse_args([
            'input.geojson',
            '--width', '100',
            '--height', '200'
        ])

        assert args.width == 100
        assert args.height == 200


class TestMain:
    """Integration tests for main function."""

    def test_file_not_found(self, temp_output_dir):
        """Test error handling for missing input file."""
        result = main([
            'nonexistent.geojson',
            '-W', '100',
            '-H', '100',
            '-o', str(temp_output_dir),
            '-q'
        ])

        assert result == 1

    def test_basic_conversion(self, temp_geojson_file, temp_output_dir):
        """Test basic conversion creates output files."""
        result = main([
            str(temp_geojson_file),
            '-W', '50',
            '-H', '50',
            '-o', str(temp_output_dir),
            '-q'
        ])

        assert result == 0

        # Check output files exist
        base_name = temp_geojson_file.stem
        cell_mask_path = temp_output_dir / f"{base_name}_cell_mask.tif"
        nucleus_mask_path = temp_output_dir / f"{base_name}_nucleus_mask.tif"

        assert cell_mask_path.exists()
        assert nucleus_mask_path.exists()

    def test_output_dimensions(self, temp_geojson_file, temp_output_dir):
        """Test that output files have correct dimensions."""
        main([
            str(temp_geojson_file),
            '-W', '100',
            '-H', '75',
            '-o', str(temp_output_dir),
            '-q'
        ])

        base_name = temp_geojson_file.stem
        cell_mask = tifffile.imread(temp_output_dir / f"{base_name}_cell_mask.tif")
        nucleus_mask = tifffile.imread(temp_output_dir / f"{base_name}_nucleus_mask.tif")

        assert cell_mask.shape == (75, 100)
        assert nucleus_mask.shape == (75, 100)

    def test_custom_suffixes(self, temp_geojson_file, temp_output_dir):
        """Test custom output file suffixes."""
        main([
            str(temp_geojson_file),
            '-W', '50',
            '-H', '50',
            '-o', str(temp_output_dir),
            '--cell-suffix', '_whole_cell',
            '--nucleus-suffix', '_nuc',
            '-q'
        ])

        base_name = temp_geojson_file.stem
        assert (temp_output_dir / f"{base_name}_whole_cell.tif").exists()
        assert (temp_output_dir / f"{base_name}_nuc.tif").exists()

    def test_cell_ids_consistent(self, temp_geojson_file, temp_output_dir):
        """Test that cell IDs are consistent between masks."""
        main([
            str(temp_geojson_file),
            '-W', '50',
            '-H', '50',
            '-o', str(temp_output_dir),
            '-q'
        ])

        base_name = temp_geojson_file.stem
        cell_mask = tifffile.imread(temp_output_dir / f"{base_name}_cell_mask.tif")
        nucleus_mask = tifffile.imread(temp_output_dir / f"{base_name}_nucleus_mask.tif")

        # Get unique IDs (excluding background)
        cell_ids = set(np.unique(cell_mask)) - {0}
        nucleus_ids = set(np.unique(nucleus_mask)) - {0}

        # Nucleus IDs should be subset of cell IDs
        assert nucleus_ids.issubset(cell_ids)

    def test_correct_cell_count(self, large_geojson_file, temp_output_dir):
        """Test that all cells are present in output."""
        main([
            str(large_geojson_file),
            '-W', '600',
            '-H', '600',
            '-o', str(temp_output_dir),
            '-q'
        ])

        base_name = large_geojson_file.stem
        cell_mask = tifffile.imread(temp_output_dir / f"{base_name}_cell_mask.tif")

        # Should have 100 cells + background
        unique_ids = np.unique(cell_mask)
        assert len(unique_ids) == 101  # 0 (background) + 100 cells

    def test_no_compress_option(self, temp_geojson_file, temp_output_dir):
        """Test that no-compress option works."""
        # With compression
        main([
            str(temp_geojson_file),
            '-W', '100',
            '-H', '100',
            '-o', str(temp_output_dir),
            '-q'
        ])

        base_name = temp_geojson_file.stem
        compressed_size = (temp_output_dir / f"{base_name}_cell_mask.tif").stat().st_size

        # Without compression
        temp_output_dir2 = Path(tempfile.mkdtemp())
        try:
            main([
                str(temp_geojson_file),
                '-W', '100',
                '-H', '100',
                '-o', str(temp_output_dir2),
                '--no-compress',
                '-q'
            ])

            uncompressed_size = (temp_output_dir2 / f"{base_name}_cell_mask.tif").stat().st_size

            # Uncompressed should generally be larger (for sparse data)
            # Note: for very small files, this might not always hold
            assert uncompressed_size >= compressed_size
        finally:
            import shutil
            shutil.rmtree(temp_output_dir2)

    def test_default_output_dir(self, temp_geojson_file):
        """Test that default output directory is same as input file."""
        result = main([
            str(temp_geojson_file),
            '-W', '50',
            '-H', '50',
            '-q'
        ])

        assert result == 0

        # Output should be in same directory as input
        base_name = temp_geojson_file.stem
        parent_dir = temp_geojson_file.parent
        cell_mask_path = parent_dir / f"{base_name}_cell_mask.tif"
        nucleus_mask_path = parent_dir / f"{base_name}_nucleus_mask.tif"

        assert cell_mask_path.exists()
        assert nucleus_mask_path.exists()

        # Cleanup
        cell_mask_path.unlink()
        nucleus_mask_path.unlink()

    def test_optimal_dtype_selection(self, temp_output_dir):
        """Test that optimal dtype is selected based on cell count."""
        # Create geojson with exactly 100 cells
        from tests.conftest import create_geojson_with_cells

        geojson = create_geojson_with_cells(100)

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.geojson', delete=False
        ) as f:
            json.dump(geojson, f)
            temp_path = Path(f.name)

        try:
            main([
                str(temp_path),
                '-W', '600',
                '-H', '600',
                '-o', str(temp_output_dir),
                '-q'
            ])

            base_name = temp_path.stem
            cell_mask = tifffile.imread(temp_output_dir / f"{base_name}_cell_mask.tif")

            # 100 cells should fit in uint8
            assert cell_mask.dtype == np.uint8
        finally:
            temp_path.unlink()


class TestRealDataIntegration:
    """Integration tests using real test data."""

    def test_full_pipeline(self, real_test_file, temp_output_dir):
        """Test full pipeline with real data."""
        result = main([
            str(real_test_file),
            '-W', '15000',
            '-H', '3000',
            '-o', str(temp_output_dir),
            '-q'
        ])

        assert result == 0

        # Check files exist
        base_name = real_test_file.stem
        cell_mask_path = temp_output_dir / f"{base_name}_cell_mask.tif"
        nucleus_mask_path = temp_output_dir / f"{base_name}_nucleus_mask.tif"

        assert cell_mask_path.exists()
        assert nucleus_mask_path.exists()

        # Load and verify
        cell_mask = tifffile.imread(cell_mask_path)
        nucleus_mask = tifffile.imread(nucleus_mask_path)

        assert cell_mask.shape == (3000, 15000)
        assert nucleus_mask.shape == (3000, 15000)

        # Should have 6446 cells + background
        cell_ids = set(np.unique(cell_mask)) - {0}
        assert len(cell_ids) == 6446

        # All nucleus IDs should be in cell IDs
        nucleus_ids = set(np.unique(nucleus_mask)) - {0}
        assert nucleus_ids.issubset(cell_ids)

    def test_real_data_dtype_optimization(self, real_test_file, temp_output_dir):
        """Test that real data uses uint16 (6446 > 255)."""
        main([
            str(real_test_file),
            '-W', '15000',
            '-H', '3000',
            '-o', str(temp_output_dir),
            '-q'
        ])

        base_name = real_test_file.stem
        cell_mask = tifffile.imread(temp_output_dir / f"{base_name}_cell_mask.tif")

        # 6446 cells requires uint16
        assert cell_mask.dtype == np.uint16
