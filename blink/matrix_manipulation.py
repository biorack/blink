import numpy as np
import scipy.sparse as sp
from .data_binning import _calc_pmzdiff_bins, _calc_massdiff_bins, _calc_massdiff_dims, _network_kernel, _shift_bins, _shift_bins_for_network, _calc_max_mz
  
def _construct_sparse_matrices(d_spec, bin_type, max_mz):
    
    i_data = d_spec['normalized_intensities']
    c_data = d_spec['counts']
    
    m_i = sp.coo_matrix((i_data, (d_spec[bin_type], d_spec['spec_ids'])), shape=(max_mz + 1, np.max(d_spec['spec_ids']) + 1), dtype=float, copy=False)
    m_c = sp.coo_matrix((c_data, (d_spec[bin_type], d_spec['spec_ids'])), shape=(max_mz + 1, np.max(d_spec['spec_ids']) + 1), dtype=float, copy=False)
    
    return m_i, m_c

def _construct_massdiff_sparse_matrices(n_spec, max_mz):

    depth = len(n_spec['massdiff_bins_list'])
    dims = np.concatenate([np.full((massdiff_bins.shape[0],), idx) for idx, massdiff_bins in enumerate(n_spec['massdiff_bins_list'])])
    max_id = np.max(n_spec['spec_ids'])
    
    ids_mod = dims * (max_id + 1)
    ids_tiled = np.tile(n_spec['spec_ids'], depth)
    ids_extended = ids_tiled + ids_mod
    
    i_data_tiled = np.tile(n_spec['normalized_intensities'], depth)
    c_data_tiled = np.tile(n_spec['counts'], depth)
    
    mass_diff_bins = np.concatenate(n_spec['massdiff_bins_list'])
    
    m_i = sp.coo_matrix((i_data_tiled, (mass_diff_bins, ids_extended)), shape=(max_mz + 1, np.max(ids_extended) + 1), dtype=float, copy=False)
    m_c = sp.coo_matrix((c_data_tiled, (mass_diff_bins, ids_extended)), shape=(max_mz + 1, np.max(ids_extended) + 1), dtype=float, copy=False)
    
    return m_i, m_c

def _build_matrices_for_network(n_mzis_s1, n_mzis_s2, precursor_mzs_s1, precursor_mzs_s2, tolerance, bin_width, mass_diffs):
    
    n_mzis_s1['pmzdiff_bins'] = _calc_pmzdiff_bins(n_mzis_s1, precursor_mzs_s1, bin_width)
    n_mzis_s1['massdiff_bins_list'] = _calc_massdiff_bins(n_mzis_s1, mass_diffs, bin_width)
    n_mzis_s1['massdiff_dims'] = _calc_massdiff_dims(n_mzis_s1['massdiff_bins_list'])

    n_mzis_s2['pmzdiff_bins'] = _calc_pmzdiff_bins(n_mzis_s2, precursor_mzs_s2, bin_width)
    _network_kernel(n_mzis_s2, tolerance, bin_width) 

    _shift_bins_for_network(n_mzis_s1, n_mzis_s2)
    max_mz = _calc_max_mz(n_mzis_s1, n_mzis_s2, mass_diffs, bin_width)

    s1_massdiff_m_i, s1_massdiff_m_c = _construct_massdiff_sparse_matrices(n_mzis_s1, max_mz)
    s1_pmzdiff_m_i, s1_pmzdiff_m_c = _construct_sparse_matrices(n_mzis_s1, 'pmzdiff_bins', max_mz)

    s2_mz_m_i, s2_mz_m_c = _construct_sparse_matrices(n_mzis_s2, 'mz_bins', max_mz)
    s2_pmzdiff_m_i, s2_pmzdiff_m_c = _construct_sparse_matrices(n_mzis_s2, 'pmzdiff_bins', max_mz)

    network_sparse_matrices = {'s1':{'mdi':s1_massdiff_m_i, 'mdc':s1_massdiff_m_c, 'nli':s1_pmzdiff_m_i, 'nlc':s1_pmzdiff_m_c, 'metadata':n_mzis_s1['metadata']},
                                's2':{'mzi':s2_mz_m_i, 'mzc':s2_mz_m_c, 'nli':s2_pmzdiff_m_i, 'nlc':s2_pmzdiff_m_c, 'metadata':n_mzis_s2['metadata']},
                                'massdiff_num':len(n_mzis_s1['massdiff_bins_list'])}
    
    return network_sparse_matrices

def _build_matrices(n_mzis_s1, n_mzis_s2, tolerance, bin_width, mass_diffs):
    
    _network_kernel(n_mzis_s1, tolerance, bin_width) 

    _shift_bins(n_mzis_s1, n_mzis_s2)
    max_mz = _calc_max_mz(n_mzis_s1, n_mzis_s2, mass_diffs, bin_width)
    
    s1_mz_m_i, s1_mz_m_c = _construct_sparse_matrices(n_mzis_s1, 'mz_bins', max_mz)
    s2_mz_m_i, s2_mz_m_c = _construct_sparse_matrices(n_mzis_s2, 'mz_bins', max_mz)
    
    sparse_matrices = {'s1':{'mzi':s1_mz_m_i, 'mzc':s1_mz_m_c, 'metadata':n_mzis_s1['metadata']},
                              's2':{'mzi':s2_mz_m_i, 'mzc':s2_mz_m_c, 'metadata':n_mzis_s2['metadata']}}
    
    return sparse_matrices

def _flatten_sparse_matrices(scores, rows, cols, remove_self_connections):
    
    if remove_self_connections:
        scores['mzi'].setdiag(0, k=0)
        scores['mzc'].setdiag(0, k=0)
        
    flat_scores = scores['mzi'].reshape((rows*cols), 1)
    flat_matches = scores['mzc'].reshape((rows*cols), 1)
    
    return flat_scores, flat_matches
