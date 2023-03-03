import numpy as np
from .data_binning import _bin_mzs

def _filter_spectra(mzis):
    """remove zero intensities to keep the sparse arrays from breaking"""
    cleaned_mzis = []
    for i,mzi in enumerate(mzis):
        idx = np.argwhere(mzi[1]>0).flatten()
        cleaned_mzis.append(mzi[:,idx])
    return cleaned_mzis

def _trim_empty(mzis):
    """remove empty spectra from list"""
    kept_idxs, mzis = np.array([[idx,mzi] for idx,mzi in enumerate(mzis) if mzi.size>0], dtype=object).T
    
    return kept_idxs, mzis

def _store_empty_ids(spec_ids, kept_idxs):
    """returns all removed empty spectral ids"""
    empty_ids = np.setdiff1d(np.arange(spec_ids[-1]+1),kept_idxs)

    return empty_ids

def _remove_duplicate_ions(mzis, min_diff):
    """remove duplicate fragment ions from a list of mass spectrum vectors (mzis) that are within min_diff by averaging m/zs and summing intensities"""
    duplicates = [i for i,s in enumerate(mzis) if min(np.diff(s[0],prepend=0))<=min_diff]
    for dup_idx in sorted(duplicates, reverse=True):
        idx = np.argwhere(np.diff(mzis[dup_idx][0])<min_diff).flatten()
        idx = sorted(idx,reverse=True)
        for sub_idx in idx:
            dup_mz = mzis[dup_idx][0][sub_idx:sub_idx+2]
            dup_intensities = mzis[dup_idx][1][sub_idx:sub_idx+2]
            new_mz = np.mean(dup_mz)
            new_intensity = np.sum(dup_intensities)
            mzis[dup_idx][0][sub_idx:sub_idx+2] = new_mz
            mzis[dup_idx][1][sub_idx:sub_idx+2] = new_intensity
        mz = np.delete(mzis[dup_idx][0],idx)
        intensity = np.delete(mzis[dup_idx][1],idx)
        mzis[dup_idx] = np.asarray([mz,intensity])

    return mzis

def _array_spec_ids(mzis):
    """arrays spectral ids such that each m/z bin is associate with an id"""
    spec_ids = np.concatenate([[i]*mzi.shape[1] for i,mzi in enumerate(mzis)]).astype(int)
    return spec_ids

def _calc_intensity_norm(mzis, intensity_power):
    """returns the intensity vector norm for each spectrum"""
    inorm = np.array([1./np.linalg.norm(mzi[1]**intensity_power) for mzi in mzis])
    return inorm

def _normalize_spectra(mzis, bin_width, intensity_power, trim_empty, remove_duplicates):
    """bins m/z values and unit-vector normalizes intensity data"""
    mzis = _filter_spectra(mzis) 
    num_ions = [mzi.shape[1] for mzi in mzis]

    if trim_empty:
        kept_idxs, mzis = _trim_empty(mzis)
    if remove_duplicates:
        mzis = _remove_duplicate_ions(mzis, bin_width*2)
    
    spec_ids = _array_spec_ids(mzis)
    inorm = _calc_intensity_norm(mzis, intensity_power)
    cnorm = np.ones(len(mzis))

    mzis = np.concatenate(mzis, axis=1)
    mzis[1] = mzis[1]**intensity_power
    
    normalized_intensities = inorm[spec_ids]*mzis[1]
    counts = cnorm[spec_ids].astype(int)
    mz_bins = _bin_mzs(mzis, bin_width)
    
    n_spec = {'spec_ids':spec_ids,
              'mz_bins':mz_bins,
              'normalized_intensities':normalized_intensities,
              'counts':counts,
              'metadata':{'num_ions':num_ions}}

    if trim_empty:
        n_spec['blanks'] = _store_empty_ids(spec_ids, kept_idxs)

    return n_spec
