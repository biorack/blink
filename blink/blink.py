import numpy as np
import pandas as pd
import scipy.sparse as sp
import sys
import os
import pickle
import logging
from timeit import default_timer as timer

from .msn_io import _read_mzml, _read_mgf 
from .spectral_normalization import _normalize_spectra
from .data_binning import _generate_full_mass_diffs
from .matrix_manipulation import _build_matrices, _build_matrices_for_network, _flatten_sparse_matrices
from .scoring import _score_sparse_matrices, _score_mass_diffs
from .arg_parser import create_rem_parser

##########################
# Core BLINK Functionality
##########################

def open_msms_file(in_file):
    if '.mgf' in in_file:
        logging.info('Processing {}'.format(os.path.basename(in_file)))
        return _read_mgf(in_file)
    if '.mzml' in in_file.lower():
        logging.info('Processing {}'.format(os.path.basename(in_file)))
        return _read_mzml(in_file)
    else:
        logging.error('Unsupported file type: {}'.format(os.path.splitext(in_file)[-1]))
        raise IOError

def open_sparse_msms_file(in_file):
    if '.npz' in in_file:
        logging.info('Processing {}'.format(os.path.basename(in_file)))
        with np.load(in_file, mmap_mode='w+',allow_pickle=True) as S:
            return dict(S)
    else:
        logging.error('Unsupported file type: {}'.format(os.path.splitext(in_file)[-1]))
        raise IOError

def write_sparse_msms_file(out_file, S):
    np.savez_compressed(out_file, **S)

def discretize_spectra(mzis_s1: list, mzis_s2: list, precursor_mzs_s1: list, precursor_mzs_s2: list, metadata_s1: list=None, metadata_s2: list=None, tolerance: float=0.01, 
                       bin_width: float=0.001, intensity_power: float=0.5, mass_diffs: list=[0], network_score: bool=False, trim_empty: bool=False, remove_duplicates: bool=False) -> dict:
    """Normalizes spectral intensities and constructs sparse matrices to be scored
    
    Parameters
    ----------
    mzis_s1
        List of np.ndarray MS2 spectra. Must match metadata row order if associating metadata.
    mzis_s2
        List of np.ndarray MS2 spectra. Must match metadata row order if associating metadata.
    precursor_mzs_s1
        List of precursor mzs 
    precursor_mzs_s2
        List of precursor mzs   
    metadata_s1: optional
        List of dictionaries that contain metadata associated with lists of spectra in row order. If None, metadata will only contain ion count.
    metadata_s2: optional
        List of dictionaries that contain metadata associated with lists of spectra in row order. If None, metadata will only contain ion count.
    tolerance: optional
        The maximum differences between m/z values (in daltons) to be considered matching
    bin_width: optional
        The value (in daltons) used to bin the m/z values. Typically set to the accuracy of MS
    intensity_power: optional
        Intensities values are raised to the power of this number
    mass_diffs: optional
        This list of mass differences is used to generate the networking scores
    network_score: optional
        If True, construct matrices necessary for calculating the network score 
    trim_empty: optional
        If True, remove empty spectra from input data
    remove_duplicates: optional
        If True, average m/zs and sum intensities of fragment ion 
    
    Returns
    -------
    discretized_spectra: dict
        a dict of sparse matrices containing spectral data and associated metadata
    """
    
    n_mzis_s1 = _normalize_spectra(mzis_s1, bin_width, intensity_power, trim_empty, remove_duplicates)
    n_mzis_s2 = _normalize_spectra(mzis_s2, bin_width, intensity_power, trim_empty, remove_duplicates)
    
    if metadata_s1 is not None:
        n_mzis_s1['metadata'] = metadata_s1
    if metadata_s2 is not None:
        n_mzis_s2['metadata'] = metadata_s2
    else:
        n_mzis_s1['metadata'] = [{} for ele in range(len(mzis_s1))]
        n_mzis_s2['metadata'] = [{} for ele in range(len(mzis_s2))]
        
    for i in range(len(mzis_s1)):
        n_mzis_s1['metadata'][i]['num_ions'] = n_mzis_s1['num_ions'][i]
    for i in range(len(mzis_s2)):
        n_mzis_s2['metadata'][i]['num_ions'] = n_mzis_s2['num_ions'][i]
    
    if network_score:
         discretized_spectra = _build_matrices_for_network(n_mzis_s1, n_mzis_s2, precursor_mzs_s1, precursor_mzs_s2, tolerance, bin_width, mass_diffs)
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
    keep_idx = scores['mzi'].data >= min_score
    
    if min_matches is not None:
        keep_idx = keep_idx * (scores['mzc'].data >= min_matches)
        
    if override_matches is not None:
        keep_idx = np.maximum(keep_idx, scores['mzc'].data >= override_matches)
        
    scores['mzi'].data[~keep_idx] = 0.0
    scores['mzc'].data[~keep_idx] = 0.0
    
    scores['mzi'].eliminate_zeros()
    scores['mzc'].eliminate_zeros()
        
    return scores


def reformat_score_matrix(scores: dict, remove_self_connections=False) -> sp.base.spmatrix:
    """Reformats the score matrix such that it can be conveniently converted to a pandas DataFrame containing non-zero hits
     
    Parameters
    ----------
    scores
        A dictionary of sparse score and count matrices
    remove_self_connections: optional
        If True, scores and counts between the same spectra will be set to zero. Only enable if scoring identical sets of spectra. 

    Returns
    ----------
    nonzero_output_mat
        A sparse matrix output that contains only nonzero rows. Can be easily converted to a pandas DataFrame.
    """
    flat_scores, flat_matches = _flatten_sparse_matrices(scores, scores['mzc'].shape[0], scores['mzc'].shape[1], remove_self_connections)
    
    nonzero_row_idxs = flat_matches.nonzero()[0]
    
    query, ref = np.unravel_index(nonzero_row_idxs, scores['mzc'].shape)
    sparse_query = sp.coo_matrix((query, (nonzero_row_idxs, np.zeros(nonzero_row_idxs.shape))), shape=(flat_scores.shape[0], 1), dtype=float, copy=False)
    sparse_ref = sp.coo_matrix((ref, (nonzero_row_idxs, np.zeros(nonzero_row_idxs.shape))), shape=(flat_scores.shape[0], 1), dtype=float, copy=False)
    
    output_mat = sp.hstack([flat_scores, flat_matches, sparse_query, sparse_ref])
    nonzero_output_mat = output_mat.tocsr()[nonzero_row_idxs, :]
    
    return nonzero_output_mat


def make_output_df(nonzero_output_mat):
    
    cols = ['score', 'matches', 'query', 'ref']
    
    output_df = pd.DataFrame.sparse.from_spmatrix(nonzero_output_mat, columns=cols)
    
    return output_df

#####################
# REM-BLINK Functions
#####################

def stack_network_matrices(scores: dict, remove_self_connections: bool=False, filter_min_score: float=0.2, filter_min_matches: int=3, filter_override_matches: int=5):
    
    md_scores = {'mzi':scores['mdi'], 'mzc':scores['mdc']}
    nl_scores = {'mzi':scores['nli'], 'mzc':scores['nlc']}
    
    depth = scores['massdiff_num'] #number of massdiffs used (in list)
    rows = len(scores['s1_metadata']) #number of query spectra
    cols = len(scores['s2_metadata']) #number of reference spectra
    
    filtered_diff_scores = filter_hits(md_scores, min_score=filter_min_score, min_matches=filter_min_matches, override_matches=filter_override_matches)
    filtered_nl_scores = filter_hits(nl_scores, min_score=filter_min_score, min_matches=filter_min_matches, override_matches=filter_override_matches)
    
    score_list = []
    matches_list = []
    
    for dim in range(depth):
        start_idx = dim * rows
        end_idx = start_idx + rows
    
        diff_scores = {'mzi':filtered_diff_scores['mzi'][start_idx:end_idx, :], 'mzc':filtered_diff_scores['mzc'][start_idx:end_idx, :]}
        
        flat_scores, flat_matches = _flatten_sparse_matrices(diff_scores, rows, cols, remove_self_connections)
        
        score_list.append(flat_scores)
        matches_list.append(flat_matches)

    flat_scores, flat_matches = _flatten_sparse_matrices(filtered_nl_scores, rows, cols, remove_self_connections)
    
    score_list.append(flat_scores)
    matches_list.append(flat_matches)
    
    score_stack = sp.hstack(score_list)
    matches_stack = sp.hstack(matches_list)
        
    return score_stack, matches_stack

def rem_predict(score_stack, scores, regressor, min_predicted_score=0.005):
    
    nonzero_row_idxs = score_stack.nonzero()[0]
    unique_nonzero_rows = np.unique(nonzero_row_idxs)
    nonzero_score_stack = score_stack.tocsr()[unique_nonzero_rows]
    
    predicted_similarity = regressor.predict(nonzero_score_stack)
    sparse_predicted = sp.coo_matrix((predicted_similarity, (unique_nonzero_rows, np.zeros(unique_nonzero_rows.shape))), shape=(score_stack.shape[0], 1), dtype=float, copy=False)
    
    predicted_rows = (sparse_predicted >= min_predicted_score).nonzero()[0]
    predicted_query, predicted_ref = np.unravel_index(predicted_rows, scores['nlc'].shape)
    
    sparse_predicted_query = sp.coo_matrix((predicted_query, (predicted_rows, np.zeros(predicted_rows.shape))), shape=(sparse_predicted.shape[0], 1), dtype=float, copy=False)
    sparse_predicted_ref = sp.coo_matrix((predicted_ref, (predicted_rows, np.zeros(predicted_rows.shape))), shape=(sparse_predicted.shape[0], 1), dtype=float, copy=False)
    
    full_score_stack = sp.hstack([score_stack, sparse_predicted, sparse_predicted_query, sparse_predicted_ref])
    
    return full_score_stack, predicted_rows

def make_rem_df(full_score_stack, matches_stack, predicted_rows, mass_diffs=[0]):
    
    full_diff_list = _generate_full_mass_diffs(mass_diffs)
    
    score_cols = ["{}_score".format(diff) for diff in full_diff_list] + ["neutral_loss_score", "rem_predicted_score", "query", "ref"]
    matches_cols = ["{}_matches".format(diff) for diff in full_diff_list] + ["neutral_loss_matches"]
    
    score_rem_df = pd.DataFrame.sparse.from_spmatrix(full_score_stack.tocsr()[predicted_rows, :], columns=score_cols)
    matches_rem_df = pd.DataFrame.sparse.from_spmatrix(matches_stack.tocsr()[predicted_rows, :], columns=matches_cols)
    
    return score_rem_df, matches_rem_df

#####################
# REM-BLINK Argparser
#####################

def main():
    parser = create_rem_parser()
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    if args.polarity == 'positive':
        with open(args.pos_model_path, 'rb') as out:
            regressor = pickle.load(out)
    else:
        with open(args.neg_model_path, 'rb') as out:
            regressor = pickle.load(out)

    start = timer()
    query_df = open_msms_file(args.query_file)
    reference_df = open_msms_file(args.reference_file)
    end = timer()

    logging.info('Input files read time: {} seconds, {} spectra'.format(end-start, query_df.shape[0]+reference_df.shape[0]))

    query_spectra = query_df.spectrum.tolist()
    reference_spectra = reference_df.spectrum.tolist()

    query_pmzs = query_df.precursor_mz.tolist()
    reference_pmzs = reference_df.precursor_mz.tolist()

    start = timer()
    discretized_spectra = discretize_spectra(query_spectra, reference_spectra, query_pmzs, reference_pmzs, network_score=True, 
                                             tolerance=args.tolerance, bin_width=args.bin_width, intensity_power=args.intensity_power,
                                             trim_empty=args.trim, remove_duplicates=args.dedup, mass_diffs=args.mass_diffs)
    end = timer()
    
    logging.info('Discretization time: {} seconds, {} spectra'.format(end-start, len(query_spectra)+len(reference_spectra)))
    
    start = timer()
    scores = score_sparse_spectra(discretized_spectra)
    stacked_scores, stacked_counts = stack_network_matrices(scores, filter_min_score=args.min_score, 
                                                            filter_min_matches=args.min_matches, filter_override_matches=args.override_matches)
    end = timer()

    logging.info('Scoring time: {} seconds, {} comparisons'.format(end-start, len(query_spectra)*len(reference_spectra)))
    
    start = timer()
    rem_scores, predicted_rows = rem_predict(stacked_scores, scores, regressor, min_predicted_score=args.min_predict)
    end = timer()

    logging.info('Prediction time: {} seconds, {} comparisons'.format(end-start, len(query_spectra)*len(reference_spectra)))

    score_rem_df, matches_rem_df = make_rem_df(rem_scores, stacked_counts, predicted_rows, mass_diffs=args.mass_diffs)

    if args.include_matches:
        output = pd.merge(matches_rem_df, score_rem_df, left_index=True, right_index=True)
    else:
        output = score_rem_df

    output['query_filename'] = args.query_file
    output['ref_filename'] = args.reference_file

    if 'id' in query_df.columns:
        output = pd.merge(output, query_df['id'], left_on='query', right_index=True)
        output.rename(columns={'id':'query_scan_num'}, inplace=True)

    elif 'scans' in query_df.columns:
        output = pd.merge(output, query_df['scans'], left_on='query', right_index=True)
        output.rename(columns={'scans':'query_scan_num'}, inplace=True)
        
    if 'spectrumid' in query_df.columns:
        output = pd.merge(output, query_df['spectrumid'], left_on='query', right_index=True)   
        output.rename(columns={'spectrumid':'query_spectrumid'}, inplace=True)
        
    if 'id' in reference_df.columns:
        output = pd.merge(output, reference_df['id'], left_on='ref', right_index=True)
        output.rename(columns={'id':'ref_scan_num'}, inplace=True)
        
    elif 'scans' in reference_df.columns:
        output = pd.merge(output, reference_df['scans'], left_on='ref', right_index=True)
        output.rename(columns={'scans':'ref_scan_num'}, inplace=True)
        
    if 'spectrumid' in reference_df.columns:
        output = pd.merge(output, reference_df['spectrumid'], left_on='ref', right_index=True)
        output.rename(columns={'spectrumid':'ref_spectrumid'}, inplace=True)

    start = timer()
    output.to_csv(args.output_file)
    end = timer()

    logging.info('Output write time: {} seconds, {} rows'.format(end-start, output.shape[0]))