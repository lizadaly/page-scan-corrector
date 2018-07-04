# Page Scan Corrector
Utility for using opencv to detect and reformat page scans, as for OCR.

The program will attempt to crop an image (PNG, JPG) to only major text sections. Preserves color.

## Installation

```
pipenv install
setup.py install --editable .
```

## Running from the command-line
```
process_image input.png outdir
```

This will write `outdir/input.png` with the result.