# BLINK

BLINK (Binned Large Interval Neighborhood Kernel) is a Python library for efficiently computing all-by-all cosine scores and number of matching ions between many fragmentation mass spectra.

## Installation

Use the package manager [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html) to install environment-base.yml for minimum requirements.

```bash
conda env create -f environment-base.yml
```

## Python dependencies
- python3
- numpy
- scipy
- pandas
- pyteomics

## Usage

```bash
# Discretize fragmentation mass spectra to sparse matrix format (.npz)
blink.py -d file.mgf

# Compute all-by-all cosine scores and # matching ions for each
# fragmentation mass spectrum (use glob to ignore metadata)
blink.py -s file_sparse.npz

# Compute A-vs-B cosine scores and # matching ions for each
# fragmentation mass spectrum (use glob to ignore metadata)
blink.py -s experiment_file_sparse.npz library_file_sparse.npz
```

## Contributing
Pull requests are welcome.

For major changes, please open an issue first to discuss what you would like to change.
