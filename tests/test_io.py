import pandas as pd

import msn_io

test_mzml_ms2_path = 'test_mzml_ms2.mzml'
test_mgf_ms2_path = 'test_mgf_ms2.mgf'

test_parsed_mzml_ms2_df = pd.read_csv('test_parsed_mzml_ms2.csv')
test_parsed_mgf_ms2_df = pd.read_csv('test_parsed_mgf_ms2.csv')

def test_read_mzml_from_path01():
    mzml_df = msn_io._read_mzml(test_mzml_ms2_path)
    assert mzml_df.equals(test_parsed_mzml_ms2_df)

def test_read_mgf_from_path01():
    mgf_df = msn_io._read_mgf(test_mgf_ms2_path)
    assert mgf_df.equals(test_parsed_mgf_ms2_df)