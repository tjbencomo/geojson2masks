"""
Command-line interface for geojson2masks.

Converts QuPath GeoJSON segmentation exports to label mask images.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile
from tqdm import tqdm

from .parser import stream_cell_geometries, count_cells
from .rasterizer import create_label_masks, convert_mask_dtype


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog='geojson2masks',
        description='Convert QuPath GeoJSON segmentation exports to label mask images.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  geojson2masks input.geojson --width 20000 --height 20000
  geojson2masks input.geojson -W 20000 -H 20000 -o output_dir/
  geojson2masks input.geojson -W 20000 -H 20000 --cell-suffix _cells --nucleus-suffix _nuclei
        """
    )

    parser.add_argument(
        'input',
        type=Path,
        help='Input GeoJSON file from QuPath export'
    )

    parser.add_argument(
        '-W', '--width',
        type=int,
        required=True,
        help='Width of the output mask image in pixels'
    )

    parser.add_argument(
        '-H', '--height',
        type=int,
        required=True,
        help='Height of the output mask image in pixels'
    )

    parser.add_argument(
        '-o', '--output-dir',
        type=Path,
        default=None,
        help='Output directory for mask files (default: same as input file)'
    )

    parser.add_argument(
        '--cell-suffix',
        type=str,
        default='_cell_mask',
        help='Suffix for cell mask filename (default: _cell_mask)'
    )

    parser.add_argument(
        '--nucleus-suffix',
        type=str,
        default='_nucleus_mask',
        help='Suffix for nucleus mask filename (default: _nucleus_mask)'
    )

    parser.add_argument(
        '--no-compress',
        action='store_true',
        help='Disable TIFF compression (faster but larger files)'
    )

    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parsed = parse_args(args)

    # Validate input file
    if not parsed.input.exists():
        print(f"Error: Input file not found: {parsed.input}", file=sys.stderr)
        return 1

    if not parsed.input.suffix.lower() == '.geojson':
        print(f"Warning: Input file does not have .geojson extension", file=sys.stderr)

    # Set up output directory
    output_dir = parsed.output_dir or parsed.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filenames
    base_name = parsed.input.stem
    cell_mask_path = output_dir / f"{base_name}{parsed.cell_suffix}.tif"
    nucleus_mask_path = output_dir / f"{base_name}{parsed.nucleus_suffix}.tif"

    if not parsed.quiet:
        print(f"Input: {parsed.input}")
        print(f"Output dimensions: {parsed.width} x {parsed.height}")
        print(f"Cell mask: {cell_mask_path}")
        print(f"Nucleus mask: {nucleus_mask_path}")
        print()

    # Count cells for progress bar
    if not parsed.quiet:
        print("Counting cells...")
        total_cells = count_cells(str(parsed.input))
        print(f"Found {total_cells} cells")
        print()
    else:
        total_cells = None

    # Create progress bar
    if not parsed.quiet and total_cells:
        pbar = tqdm(total=total_cells, desc="Rasterizing", unit="cells")

        def progress_callback(current, total):
            pbar.n = current
            pbar.refresh()
    else:
        pbar = None
        progress_callback = None

    # Stream and rasterize
    if not parsed.quiet:
        print("Creating label masks...")

    cell_geometries = stream_cell_geometries(str(parsed.input))
    cell_mask, nucleus_mask = create_label_masks(
        cell_geometries,
        width=parsed.width,
        height=parsed.height,
        total_cells=total_cells,
        progress_callback=progress_callback
    )

    if pbar:
        pbar.close()

    # Get max cell ID for dtype optimization
    max_cell_id = max(cell_mask.max(), nucleus_mask.max())
    if not parsed.quiet:
        print(f"Max cell ID: {max_cell_id}")

    # Convert to optimal dtype
    cell_mask = convert_mask_dtype(cell_mask, max_cell_id)
    nucleus_mask = convert_mask_dtype(nucleus_mask, max_cell_id)
    if not parsed.quiet:
        print(f"Using dtype: {cell_mask.dtype}")

    # Save masks
    compression = None if parsed.no_compress else 'zlib'

    if not parsed.quiet:
        print(f"Saving cell mask to {cell_mask_path}...")
    tifffile.imwrite(
        cell_mask_path,
        cell_mask,
        compression=compression,
        photometric='minisblack'
    )

    if not parsed.quiet:
        print(f"Saving nucleus mask to {nucleus_mask_path}...")
    tifffile.imwrite(
        nucleus_mask_path,
        nucleus_mask,
        compression=compression,
        photometric='minisblack'
    )

    if not parsed.quiet:
        print()
        print("Done!")
        print(f"  Cell mask: {cell_mask_path} ({cell_mask_path.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"  Nucleus mask: {nucleus_mask_path} ({nucleus_mask_path.stat().st_size / 1024 / 1024:.1f} MB)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
