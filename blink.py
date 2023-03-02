import numpy as np
import pandas as pd

from spectral_normalization import _normalize_spectra
from matrix_manipulation import _build_matrices, _build_matrices_for_network
from scoring import _score_sparse_matrices, _score_mass_diffs, _stack_dense

#########################
# Core Functionality
#########################

def discretize_spectra(s1_df:pd.DataFrame, s2_df:pd.DataFrame, tolerance: float=0.01, bin_width: float=0.001, intensity_power: float=0.5, mass_diffs: list=[0], 
                        network_score: bool=False, associate_metadata: bool=True, trim_empty: bool=False, remove_duplicates: bool=False) -> dict:
    """Normalizes spectral intensities and constructs sparse matrices to be scored
    
    Parameters
    ----------
    s1_df: pd.DataFrame
        The dataframe that contains query spectra and associated metadata
    s2_df: pd.DataFrame
        The dataframe that contains reference spectra and associated metadata
    tolerance : float, optional
        The maximum differences between m/z values (in daltons) to be considered matching
    bin_width: float, optional
        The value (in daltons) used to bin the m/z values. Typically set to the accuracy of MS
    intensity_power: float, optional
        Intensities values are raised to the power of this number
    mass_diffs: list, optional
        This list of mass differences is used to generate the networking scores
    network_score: bool, optional
        If True, construct matrices necessary for calculating the network score 
    associate_metadata: bool, optional
        If True, associate columns in input dataframes with spectral data
    trim_empty: bool, optional
        If True, remove empty spectra from input data
    remove_duplicates: bool, optional
        If True, average m/zs and sum intensities of fragment ion 

    Returns
    -------
    discretized_spectra: dict
        a dict of sparse matrices containing spectral data and associated metadata
    """
    mzis_s1 = s1_df.spectrum.tolist()
    mzis_s2 = s2_df.spectrum.tolist()
    
    n_mzis_s1 = _normalize_spectra(mzis_s1, bin_width, intensity_power, trim_empty, remove_duplicates)
    n_mzis_s2 = _normalize_spectra(mzis_s2, bin_width, intensity_power, trim_empty, remove_duplicates)
    
    if associate_metadata:
        n_mzis_s1['metadata'] = s1_df.drop(columns=['spectrum']).to_dict(orient='records')
        n_mzis_s2['metadata'] = s2_df.drop(columns=['spectrum']).to_dict(orient='records')
        
    if network_score:
         discretized_spectra = _build_matrices_for_network(n_mzis_s1, n_mzis_s2, s1_df, s2_df, tolerance, bin_width, mass_diffs)
    else:
        discretized_spectra = _build_matrices(n_mzis_s1, n_mzis_s2, tolerance, bin_width, mass_diffs)

    return discretized_spectra

def score_sparse_spectra(discretized_spectra: dict) -> dict:
    """Generates scores and matching ion counts for discretized spectra"""
    
    if 'mdi' and 'mdc' in discretized_spectra['s1']:
        scores = _score_mass_diffs(discretized_spectra) 
    else:
        scores = _score_sparse_matrices(discretized_spectra)
        
    return scores

def compute_max_network_score(scores: dict) -> dict:
    """Computes the maximum scores from dense score stack"""

    score_stack, count_stack = _stack_dense(scores)
    
    network_scores = np.max(score_stack, axis=0)
    network_counts = np.max(count_stack, axis=0)

    network_scores = {'mzi_mzi':network_scores, 
                      'mzc_mzc':network_counts}
    
    return network_scores

    
def compute_sum_network_score(scores: dict) -> dict:
    """Computes the sum of scores from dense score stack"""

    score_stack, count_stack = _stack_dense(scores)
    
    network_scores = np.sum(score_stack, axis=0)
    network_counts = np.sum(count_stack, axis=0)

    network_scores = {'mzi_mzi':network_scores, 
                      'mzc_mzc':network_counts}
    
    return network_scores

def filter_hits(scores: dict, min_score: float=0.5, min_matches: int=5, override_matches: int=20) -> dict:
    """Filter scores and counts based on minimum scores, minimum matches, and optionally a number of matches that overrides the minimum scores

    Parameters
    ----------
    good_score: float
        remove hits with less than this score
    min_matches: int, optional
        remove hits with less than this number of matches
    override_matches: int, optional
        keep hits with scores less than the good_score parameter if the number of matches is greater than or equal to this number

    Returns
    ----------
    scores: dict
        a dictionary of filtered Scipy Sparse COO matrices 
    """
    idx = scores['mzi_mzi']>=min_score
    if min_matches is not None:
        idx = idx.multiply(scores['mzc_mzc']>=min_matches)
    if override_matches is not None:
        idx = idx.maximum(scores['mzc_mzc']>=override_matches)
    scores['mzi_mzi'] = scores['mzi_mzi'].multiply(idx).tocoo()
    scores['mzc_mzc'] = scores['mzc_mzc'].multiply(idx).tocoo()
        
    return scores

def reformat_score_matrix(scores: dict) -> np.ndarray:
    """Reformats the score matrix such that it can be conveniently converted to a pandas DataFrame containing non-zero hits"""
    if scores['mzi_mzi'].format != 'coo' and scores['mzc_mzc'].format != 'coo':
        scores['mzi_mzi'] = scores['mzi_mzi'].tocoo()
        scores['mzc_mzc'] = scores['mzc_mzc'].tocoo()
    
    idx = np.ravel_multi_index((scores['mzi_mzi'].row,scores['mzi_mzi'].col),scores['mzi_mzi'].shape)
    r,c = np.unravel_index(idx,scores['mzi_mzi'].shape)

    reformed_score_matrix = np.zeros((len(idx),5))#,dtype='>i4')
    reformed_score_matrix[:,0] = idx
    reformed_score_matrix[:,1] = c #query
    reformed_score_matrix[:,2] = r #reference
    idx = np.in1d(idx, idx).nonzero()
    reformed_score_matrix[idx,3] = scores['mzi_mzi'].data
    reformed_score_matrix[idx,4] = scores['mzc_mzc'].data

    #remove self connections
    reformed_score_matrix = reformed_score_matrix[reformed_score_matrix[:,1]!=reformed_score_matrix[:,2]]

    return reformed_score_matrix

def make_score_df(reformed_score_matrix:dict, discretized_spectra:dict) -> pd.DataFrame:

    df = pd.DataFrame(reformed_score_matrix, columns=['raveled_index','query','ref','score','matches'])
    df = pd.merge(df,pd.DataFrame(discretized_spectra['s1']['metadata']).add_suffix('_query'),left_on='query',right_index=True,how='left')
    df = pd.merge(df,pd.DataFrame(list(discretized_spectra['s2']['metadata'])).add_suffix('_ref'),left_on='ref',right_index=True,how='left')

    return df
