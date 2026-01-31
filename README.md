# geojson2masks

Convert QuPath GeoJSON segmentation exports to label mask images.

## Problem Statement

Multiplex immunofluorescence (mIF) imaging workflows often use [QuPath](https://qupath.github.io/) for cell and nucleus segmentation. QuPath can export segmentation results as GeoJSON files containing polygon coordinates for each detected cell and its nucleus.

However, many downstream analysis tools and deep learning frameworks expect segmentation data as **label mask images** (where each pixel is assigned an integer cell ID) rather than vector polygons. Converting large GeoJSON exports (often 15-20GB for whole slide images) to label masks is challenging due to memory constraints.

**geojson2masks** solves this by:
- Streaming the GeoJSON file to handle arbitrarily large files without loading them entirely into memory
- Generating paired cell and nucleus label masks with consistent cell IDs
- Automatically optimizing output file size with appropriate data types and compression

## Installation

Requires Python 3.9+.

```bash
# Clone the repository
git clone https://github.com/tjbencomo/geojson2masks.git
cd geojson2masks

# Create a virtual environment and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

### Basic Usage

```bash
geojson2masks input.geojson --width 20000 --height 20000
```

This creates two TIFF files in the same directory as the input:
- `input_cell_mask.tif` - Label mask for whole cell segmentations
- `input_nucleus_mask.tif` - Label mask for nuclear segmentations

### Options

```
geojson2masks [-h] -W WIDTH -H HEIGHT [-o OUTPUT_DIR]
              [--cell-suffix CELL_SUFFIX] [--nucleus-suffix NUCLEUS_SUFFIX]
              [--no-compress] [-q] input

Arguments:
  input                 Input GeoJSON file from QuPath export

Required:
  -W, --width           Width of the output mask image in pixels
  -H, --height          Height of the output mask image in pixels

Optional:
  -o, --output-dir      Output directory (default: same as input file)
  --cell-suffix         Suffix for cell mask filename (default: _cell_mask)
  --nucleus-suffix      Suffix for nucleus mask filename (default: _nucleus_mask)
  --no-compress         Disable TIFF compression (faster but larger files)
  -q, --quiet           Suppress progress output
```

### Examples

```bash
# Specify output directory
geojson2masks input.geojson -W 20000 -H 20000 -o /path/to/output/

# Custom output suffixes
geojson2masks input.geojson -W 20000 -H 20000 --cell-suffix _cells --nucleus-suffix _nuclei

# Faster processing (no compression)
geojson2masks input.geojson -W 20000 -H 20000 --no-compress
```

## Output Format

- **File format**: TIFF with zlib compression (unless `--no-compress` is used)
- **Data type**: Automatically selected based on cell count:
  - uint8 for ≤255 cells
  - uint16 for ≤65,535 cells
  - uint32 for larger datasets
- **Cell IDs**: 1-indexed (0 = background), consistent between cell and nucleus masks

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

## License

MIT
