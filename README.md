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
# Output filename takes form: [file]_[bin_width]_[intensity_scale].npz
>> blink.py -d *.mgf
small_0p001_0p5.npz medium_0p001_0p5.npz

# Compute all-by-all cosine scores and # matching ions for each fragmentation mass spectrum
# Output filename takes form: [file]__[tolerance]_[min_score]_[min_matches].npz
>> blink.py -s small_0p001_0p5.npz
small_0p001_0p5__0p01_0p4_3.npz

# Compute A-vs-B cosine scores and # matching ions for each fragmentation mass spectrum
# Output filename takes form: [file1]_[file2]_[tolerance]_[min_score]_[min_matches].npz

>> blink.py -s small_0p001_0p5.npz medium_0p001_0p5.npz
small_0p001_0p5_medium_0p001_0p5_0p01_0p4_3.npz
```

## Contributing
Pull requests are welcome.

For major changes, please open an issue first to discuss what you would like to change.
