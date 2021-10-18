# BLINK

BLINK (Binned Large Interval Network Kernel) is a Python library for efficiently computing all-by-all cosine scores and number of matching ions between many fragmentation mass spectra.

## Installation

Use the package manager [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html) to install environment-base.yml for minimum requirements.

```bash
conda env create -f environment-base.yml
```

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/biorack/blink/HEAD)

## Python dependencies
- python3
- numpy
- scipy
- pandas
- pyteomics

## Usage

```bash
usage: blink.py [-h] (-d DISCRETIZE [DISCRETIZE ...] | -t TRIM [TRIM ...] | -s SCORE [SCORE ...]) [--bin_width BIN_WIDTH]
                [--intensity_power I] [--tolerance TOLERANCE] [--min_score S] [--min_matches M] [-f] [--out_dir OUT_DIR]

BLINK discretizes mass spectra (given .mgf inputs), and scores discretized spectra (given .npz inputs)

optional arguments:
  -h, --help            show this help message and exit
  -d DISCRETIZE [DISCRETIZE ...], --discretize DISCRETIZE [DISCRETIZE ...]
                        convert input file(s) into CSR matrix format (.npz)
  -t TRIM [TRIM ...], --trim TRIM [TRIM ...]
                        remove empty spectra from CSR matrix file(s)
  -s SCORE [SCORE ...], --score SCORE [SCORE ...]
                        score CSR matrix file with itself or another

  --bin_width BIN_WIDTH
                        width of bins in mz
  --intensity_power I   power to raise intensites to in when scoring
  --tolerance TOLERANCE
                        maximum tolerance in mz for fragment ions to match
  --min_score S         minimum scores to include in output
  --min_matches M       minimum matches to include in output

  -f, --force           force file(s) to be remade if they exist
  --out_dir OUT_DIR     change output location for output file(s)

# Discretize fragmentation mass spectra to sparse matrix format (.npz)
# Output filename takes form: [file]_[bin_width]_[intensity_scale].npz
# small = 1e2 spectra, medium = 1e4 spectra
>> blink.py -d ./example/*.mgf
small_0p001_0p5.npz medium_0p001_0p5.npz

# Compute all-by-all cosine scores and # matching ions for each fragmentation mass spectrum
# Output filename takes form: [file]__[tolerance]_[min_score]_[min_matches].npz
>> blink.py -s ./example/small_0p001_0p5.npz
small_0p001_0p5__0p01_0p4_3.npz

# Compute A-vs-B cosine scores and # matching ions for each fragmentation mass spectrum
# Output filename takes form: [file1]_[file2]_[tolerance]_[min_score]_[min_matches].npz

>> blink.py -s ./example/small_0p001_0p5.npz ./example/medium_0p001_0p5.npz
small_0p001_0p5_medium_0p001_0p5_0p01_0p4_3.npz
```

## Contributing
Pull requests are welcome.

For major changes, please open an issue first to discuss what you would like to change.
