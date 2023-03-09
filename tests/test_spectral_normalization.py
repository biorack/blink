import os
import numpy as np
import pandas as pd
from math import isclose

from blink import spectral_normalization

module_dir = os.path.join(os.path.dirname(__file__), '..')
spectral_data = pd.read_json(os.path.join(module_dir, 'tests', 'parsed_mzml_ms2.json'))
spectral_data.spectrum = spectral_data.spectrum.apply(np.array)

def test_normalize_spectra01():
    spectra = spectral_data.spectrum.tolist()

    normalized_spectra = spectral_normalization._normalize_spectra(spectra, 0.001, 0.5, False, False)

    for spec_id in set(normalized_spectra['spec_ids']):
        spec_id_idxs = normalized_spectra['spec_ids'] == spec_id
        spec_vector_sum = np.sum(normalized_spectra['normalized_intensities'][spec_id_idxs])
        assert isclose(spec_vector_sum, 1)