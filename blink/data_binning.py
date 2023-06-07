import numpy as np

def _generate_full_mass_diffs(mass_diffs):
    
    mass_diffs_neg = [-diff for diff in mass_diffs if diff > 0]
    full_diff_list = mass_diffs + mass_diffs_neg
    
    return full_diff_list

def _bin_mzs(mzis, bin_width):
    """bins m/z values based on bin width"""
    mz_bins = np.rint(mzis[0]/bin_width).astype(int)
    return mz_bins

def _calc_pmzdiff_bins(d_spec, pmzs, bin_width):
    """
    input is discretized spectra and pmzs
    ourput is precursor_mz diff matrices
    """
    pmzdiff_bins = (np.rint(np.asarray(pmzs)[d_spec['spec_ids']]/bin_width) - d_spec['mz_bins']).astype(int)
    
    return pmzdiff_bins

def _calc_massdiff_bins(d_spec, mass_diffs, bin_width):
    """
    input is discretized spectra and mass_diffs
    """
    mass_diffs_neg = [-diff for diff in mass_diffs if diff > 0]
    mass_diffs = mass_diffs + mass_diffs_neg
    mass_diffs_binned = [np.rint(diff/bin_width).astype(int) for diff in mass_diffs]
    massdiff_bins_list = [d_spec['mz_bins'] - binned_diffs for binned_diffs in mass_diffs_binned]
   
    return massdiff_bins_list

def _calc_massdiff_dims(massdiff_bins_list):
    massdiff_bins_dims = [np.full((massdiff_bins.shape[0],), idx) for idx, massdiff_bins in enumerate(massdiff_bins_list)]
    return massdiff_bins_dims

def _shift_bins(d_spec1, d_spec2):
    """
    to compensate for negative bin values which would break sparse matrix, all bins are shifted by the minimum value
    """
    bin_min1 = np.concatenate([d_spec1['mz_bins']]).min()
    bin_min2 = np.concatenate([d_spec2['mz_bins']]).min()
    
    shift = -min([bin_min1, bin_min2])
    
    d_spec1['shift'] = shift
    d_spec1['mz_bins'] = d_spec1['mz_bins'] + shift
    
    d_spec2['shift'] = shift
    d_spec2['mz_bins'] = d_spec2['mz_bins'] + shift

def _shift_bins_for_network(d_spec1, d_spec2):
    
    bin_min1 = np.concatenate([d_spec1['pmzdiff_bins'], np.concatenate(d_spec1['massdiff_bins_list'])]).min()
    bin_min2 = np.concatenate([d_spec2['pmzdiff_bins']]).min()
    
    shift = -min([bin_min1, bin_min2])
    
    d_spec1['shift'] = shift
    d_spec1['mz_bins'] = d_spec1['mz_bins'] + shift
    d_spec1['pmzdiff_bins'] = d_spec1['pmzdiff_bins'] + shift
    d_spec1['massdiff_bins_list'] = [massdiff_bins + shift for massdiff_bins in d_spec1['massdiff_bins_list']]
    
    d_spec2['shift'] = shift
    d_spec2['mz_bins'] = d_spec2['mz_bins'] + shift
    d_spec2['pmzdiff_bins'] = d_spec2['pmzdiff_bins'] + shift
    d_spec2['massdiff_bins_list'] = [massdiff_bins + shift for massdiff_bins in d_spec1['massdiff_bins_list']]

def _calc_max_mz(d_spec1, d_spec2, mass_diffs, bin_width):
    
    max_mzs = []
    max_mzs.append(np.max(d_spec1['mz_bins']) + np.rint(np.max(mass_diffs)/bin_width).astype(int))
    max_mzs.append(np.max(d_spec2['mz_bins']) + np.rint(np.max(mass_diffs)/bin_width).astype(int))
    
    if 'pmzdiff_bins' in d_spec1.keys() and 'pmzdiff_bins' in d_spec2.keys():
        max_mzs.append(np.max(d_spec1['pmzdiff_bins']) + np.rint(np.max(mass_diffs)/bin_width).astype(int))
        max_mzs.append(np.max(d_spec2['pmzdiff_bins']) + np.rint(np.max(mass_diffs)/bin_width).astype(int))
    
    max_mz = max(max_mzs)
    return max_mz

def _network_kernel(d_spec, tolerance, bin_width):
    """
    expand bins to have tolerance
    """
    bin_num = int(2*(tolerance/bin_width)-1)
    tol_matrix = np.arange(-bin_num//2+1, bin_num//2+1).flatten()
    
    d_spec['normalized_intensities'] = np.add.outer(d_spec['normalized_intensities'], np.zeros_like(tol_matrix)).flatten()
    d_spec['counts'] = np.add.outer(d_spec['counts'], np.zeros_like(tol_matrix)).flatten()
    d_spec['spec_ids'] = np.add.outer(d_spec['spec_ids'], np.zeros_like(tol_matrix)).flatten()
    d_spec['mz_bins'] = np.add.outer(d_spec['mz_bins'], tol_matrix).flatten()
    
    if 'pmzdiff_bins' in d_spec.keys():
        d_spec['pmzdiff_bins'] = np.add.outer(d_spec['pmzdiff_bins'], tol_matrix).flatten()