import numpy as np
import torch
import scipy.sparse as sp

def _convert_to_tensor(vector):
    
    vals = vector.data
    idxs = np.vstack((vector.row, vector.col))

    i = torch.LongTensor(idxs)
    v = torch.FloatTensor(vals)
    shape = vector.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def _convert_to_sparse(tensor):
    
    data = np.array(tensor.values().cpu())
    row, col = np.array(tensor.indices().cpu())
    
    sparse_output = sp.coo_matrix((data, (row, col)), shape=tensor.shape)
    
    return sparse_output

def _score_mass_diffs(discretized_spectra, gpu=False):

    if gpu:
        mdi_v1 = _convert_to_tensor(discretized_spectra['s1']['mdi'].T).to('cuda')
        mdc_v1 = _convert_to_tensor(discretized_spectra['s1']['mdc'].T).to('cuda')

        nli_v1 = _convert_to_tensor(discretized_spectra['s1']['nli'].T).to('cuda')
        nlc_v1 = _convert_to_tensor(discretized_spectra['s1']['nlc'].T).to('cuda')
        
        mzi_v2 = _convert_to_tensor(discretized_spectra['s2']['mzi']).to('cuda')
        mzc_v2 = _convert_to_tensor(discretized_spectra['s2']['mzc']).to('cuda')
        
        nli_v2 = _convert_to_tensor(discretized_spectra['s2']['nli']).to('cuda')
        nlc_v2 = _convert_to_tensor(discretized_spectra['s2']['nlc']).to('cuda')
        
        mdi_scores = _convert_to_sparse(mdi_v1.mm(mzi_v2))
        mdc_counts = _convert_to_sparse(mdc_v1.mm(mzc_v2))
        
        nli_scores = _convert_to_sparse(nli_v1.mm(nli_v2))
        nlc_counts = _convert_to_sparse(nlc_v1.mm(nlc_v2))

    elif not gpu:
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
              'massdiff_num':discretized_spectra['massdiff_num'],
              'mass_diffs':discretized_spectra['mass_diffs']}

    return scores

def _score_sparse_matrices(discretized_spectra, gpu=False):
    """
    given two sparse matrices, calculate and return their score matrix
    """

    if gpu:
        mzi_v1 = _convert_to_tensor(discretized_spectra['s1']['mzi'].T).to("cuda")
        mzc_v1 = _convert_to_tensor(discretized_spectra['s1']['mzc'].T).to("cuda")

        mzi_v2 = _convert_to_tensor(discretized_spectra['s2']['mzi']).to("cuda")
        mzc_v2 = _convert_to_tensor(discretized_spectra['s2']['mzc']).to("cuda")
        
        mzi_scores = _convert_to_sparse(mzi_v1.mm(mzi_v2))
        mzc_counts = _convert_to_sparse(mzc_v1.mm(mzc_v2))
        
    elif not gpu:
        mzi_v1 = discretized_spectra['s1']['mzi'].T
        mzc_v1 = discretized_spectra['s1']['mzc'].T

        mzi_v2 = discretized_spectra['s2']['mzi']
        mzc_v2 = discretized_spectra['s2']['mzc']
        
        mzi_scores = mzi_v1 @ mzi_v2
        mzc_counts = mzc_v1 @ mzc_v2
    

    scores = {'mzi':mzi_scores,
              'mzc':mzc_counts}
    
    return scores
    
    return scores
