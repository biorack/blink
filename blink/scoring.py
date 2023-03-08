import numpy as np

def _score_mass_diffs(discretized_spectra):
    
    mdi_v1 = discretized_spectra['s1']['mdi'].T
    mdc_v1 = discretized_spectra['s1']['mdc'].T

    nli_v1 = discretized_spectra['s1']['nli'].T
    nlc_v1 = discretized_spectra['s1']['nlc'].T
    
    mzi_v2 = discretized_spectra['s2']['mzi']
    mzc_v2 = discretized_spectra['s2']['mzc']
    
    nli_v2 = discretized_spectra['s2']['nli']
    nlc_v2 = discretized_spectra['s2']['nlc']
    
    mdi_scores = mdi_v1 @ mzi_v2
    mdc_counts = mdc_v1 @ mzc_v2
    
    nli_scores = nli_v1 @ nli_v2
    nlc_counts = nlc_v1 @ nlc_v2
    
    scores = {'mdi':mdi_scores,
              'mdc':mdc_counts,
              'nli':nli_scores,
              'nlc':nlc_counts,
              's1_metadata':discretized_spectra['s1']['metadata'],
              's2_metadata':discretized_spectra['s2']['metadata'],
              'massdiff_num':discretized_spectra['massdiff_num']}

    return scores

def _score_sparse_matrices(discretized_spectra):
    """
    given two sparse matrices, calculate and return their score matrix
    """
    mzi_v1 = discretized_spectra['s1']['mzi'].T
    mzc_v1 = discretized_spectra['s1']['mzc'].T
    
    mzi_v2 = discretized_spectra['s2']['mzi']
    mzc_v2 = discretized_spectra['s2']['mzc']
    
    mzi_scores = mzi_v1 @ mzi_v2
    mzc_counts = mzc_v1 @ mzc_v2

    scores = {'mzi':mzi_scores,
              'mzc':mzc_counts}
    
    return scores

def _stack_dense(scores):
    
    depth = scores['massdiff_num'] #number of massdiffs used (in list)
    rows = len(scores['s1_metadata']['num_ions']) #number of query spectra
    cols = len(scores['s2_metadata']['num_ions']) #number of reference spectra
    
    mdi_score_stack = np.zeros((depth, rows, cols))
    mdc_count_stack = mdi_score_stack.copy()

    for dim in range(depth):
        start_idx = dim * rows
        end_idx = start_idx + rows
    
        mdi_score_stack[dim] = scores['mdi'][start_idx:end_idx, :].todense()
        mdc_count_stack[dim] = scores['mdc'][start_idx:end_idx, :].todense()
        
    score_stack = np.concatenate((scores['nli'].todense(), mdi_score_stack))
    count_stack = np.concatenate((scores['nlc'].todense(), mdc_count_stack))
        
    return score_stack, count_stack
