import pandas as pd
import numpy as np
import os

from blink import msn_io

module_dir = os.path.join(os.path.dirname(__file__), '..')

expected_mzml_df_result = pd.read_json(os.path.join(module_dir, 'tests', 'parsed_mzml_ms2.json'))
mzml_spectra_file = os.path.join(module_dir, 'tests', 'test_mzml_ms2.mzml')

expected_mgf_df_result = pd.read_json(os.path.join(module_dir, 'tests', 'parsed_mgf_ms2.json'))
mgf_spectra_file = os.path.join(module_dir, 'tests', 'test_mgf_ms2.mgf')

def test_read_mzml_from_path01():

    mzml_df = msn_io._read_mzml(mzml_spectra_file)
    assert mzml_df.shape == expected_mzml_df_result.shape

def test_read_mgf_from_path01():

    mgf_df = msn_io._read_mgf(mgf_spectra_file)
    assert mgf_df.shape == expected_mgf_df_result.shape

def test_read_mzml_from_path02():

    mzml_df = msn_io._read_mzml(mzml_spectra_file)

    expected_rounded_spectra = expected_mzml_df_result.spectrum.apply(lambda x: np.round(x, 3))
    mzml_df_rounded_spectra = mzml_df.spectrum.apply(lambda x: np.round(x.astype(float), 3))
    assert expected_rounded_spectra.equals(mzml_df_rounded_spectra)
    
def test_read_mgf_from_path02():

    mgf_df = msn_io._read_mgf(mgf_spectra_file)

    expected_rounded_spectra = expected_mgf_df_result.spectrum.apply(lambda x: np.round(x, 3))
    mgf_df_rounded_spectra = mgf_df.spectrum.apply(lambda x: np.round(x.astype(float), 3))
    assert expected_rounded_spectra.equals(mgf_df_rounded_spectra)

