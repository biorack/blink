#!/usr/bin/env python

import sys
import os
import argparse
import glob
from timeit import default_timer as timer
import logging

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import norm
import pandas as pd
from pyteomics import mgf

###########################
# Mass Spectra Transforms
###########################

def remove_duplicate_ions(spectra,min_diff=0.002,do_avg=True):
    """
    helper function to remove ions that are within min_diff.
    A good rule of thumb is ions have to be greater than 2 bins apart
    This function averages their m/z and intensities and removes the duplicates
    """
    bad_ones = [i for i,s in enumerate(spectra) if min(np.diff(s[0],prepend=0))<=min_diff]
    for bad_idx in sorted(bad_ones, reverse=True):
        idx = np.argwhere(np.diff(spectra[bad_idx][0])<min_diff).flatten()
        idx = sorted(idx,reverse=True)
        for sub_idx in idx:
            dup_mz = spectra[bad_idx][0][sub_idx:sub_idx+2]
            dup_intensities = spectra[bad_idx][1][sub_idx:sub_idx+2]
            new_mz = np.mean(dup_mz)
            new_intensity = np.max(dup_intensities)
            spectra[bad_idx][0][sub_idx:sub_idx+2] = new_mz
            spectra[bad_idx][1][sub_idx:sub_idx+2] = new_intensity
        mz = np.delete(spectra[bad_idx][0],idx)
        intensity = np.delete(spectra[bad_idx][1],idx)
        spectra[bad_idx] = np.asarray([mz,intensity])
    return spectra

def discretize_spectra(mzis, pmzs=None, bin_width=0.001, intensity_power=0.5, expand=False,remove_duplicates=True):
    if remove_duplicates==True:
        mzis = remove_duplicate_ions(mzis,min_diff=bin_width*2)
    spec_ids = np.concatenate([[i]*m.shape[1] for i,m in enumerate(mzis)]).astype(int)
    mzis = np.concatenate(mzis, axis=1)
    mzis[1] = mzis[1]**intensity_power
    mz_bin_idxs = np.rint(mzis[0]/bin_width).astype(int)

    if pmzs is not None:
        nl_bin_idxs = np.rint(np.asarray(pmzs)[spec_ids]/bin_width).astype(int) - mz_bin_idxs
        mz_bin_idxs = np.concatenate([mz_bin_idxs,mz_bin_idxs.max()+1-nl_bin_idxs])
        mzis = np.concatenate([mzis,(0+1j)*mzis], axis=1)
        spec_ids = np.concatenate([spec_ids,spec_ids])

    num_bins = mz_bin_idxs.max()+1

    intensity = sp.coo_matrix((mzis[1], (spec_ids, mz_bin_idxs)), (spec_ids[-1]+1, num_bins))
    intensity = intensity.multiply(1./norm(intensity.real, axis=1)[:,None]).tocsr()
    count = sp.coo_matrix(((mzis[1].real>0)+(0+1j)*(mzis[1].imag>0), (spec_ids, mz_bin_idxs)))
    count = count.multiply(((count.real.getnnz(axis=1)**0.5)/norm(count.real, axis=1))[:,None]).tocsr()

    S =  {'intensity': intensity.data,
          'count': count.data,
          'indptr' : intensity.indptr,
          'mz': mz_bin_idxs,
          'bin_width': bin_width,
          'intensity_power': intensity_power}

    if expand:
        S = expand_sparse_spectra(**S)

    return S

def expand_sparse_spectra(mz,intensity,count,indptr,**kwargs):
    S = {}

    for d,data in zip(['i','c'],[intensity, count]):
        Sd = sp.csr_matrix((data, mz, indptr), dtype=data.dtype)
        S['mz'+d] = Sd.real
        if Sd.imag.sum() > 0:
            S['nl'+d] = Sd.imag

    return S

def condense_sparse_spectra(sparse_spectra, bin_width=0.001):
    mzis = [np.array([row.indices*bin_width,row.data])
            for row in sparse_spectra]
    return mzis

##########
# Kernel
##########
def network_kernel(n, m, mass_diffs=[0], react_dist=1, bin_width=0.001, tolerance=0.01):

    bin_num = int(2*(tolerance/bin_width)-1)

    mass_diffs = np.sort(mass_diffs)

    mass_diffs = [-m for m in mass_diffs[::-1]]+[m for m in mass_diffs]
    if mass_diffs[len(mass_diffs)//2] == 0:
        mass_diffs.pop(len(mass_diffs)//2)

    mass_diffs = np.rint(np.array(mass_diffs)/bin_width).astype(int)

    def react(mass_diffs, react_dist):
        if react_dist == 1:
            return mass_diffs
        else:
            return np.add.outer(mass_diffs, react(mass_diffs, react_dist-1))

    mass_diffs = np.abs(react(mass_diffs, react_dist))

    mass_diffs = np.add.outer(mass_diffs, np.arange(-bin_num//2+1, bin_num//2+1))
    mass_diffs = np.unique(mass_diffs.flatten())

    N = sp.diags(np.ones_like(mass_diffs), mass_diffs, shape=(n, m), format='csr', dtype=bool)

    return N


#########################
# Biochemical Masses
#########################

biochem_masses = [0.,      # Self
                  12.,     # C
                  1.00783, # H
                  2.01566, # H2
                  15.99491,# O
                  0.02381, # NH2 - O
                  78.95851,# PO3
                  31.97207]# S


#######################
# Mass Spectra Loading
#######################
def read_mgf(in_file):
    msms_df = []
    with mgf.MGF(in_file) as reader:
        for spectrum in reader:
            d = spectrum['params']
            d['spectrum'] = np.array([spectrum['m/z array'],
                                      spectrum['intensity array']])
            d['precursor_mz'] = d['pepmass'][0]
            msms_df.append(d)
    msms_df = pd.DataFrame(msms_df)
    return msms_df

def write_sparse_msms_file(out_file, S):
    np.savez_compressed(out_file, **S)

def open_msms_file(in_file):
    if '.mgf' in in_file:
        logging.info('Processing {}'.format(os.path.basename(in_file)))
        return read_mgf(in_file)
    else:
        logging.error('Unsupported file type: {}'.format(os.path.splitext(in_file)[-1]))
        raise IOError

def open_sparse_msms_file(in_file):
    if '.npz' in in_file:
        logging.info('Processing {}'.format(os.path.basename(in_file)))
        return np.load(in_file)
    else:
        logging.error('Unsupported file type: {}'.format(os.path.splitext(in_file)[-1]))
        raise IOError

#########################
# Command Line Interface
#########################
'''
https://stackoverflow.com/questions/4194948/python-argparse-is-there-a-way-to-specify-a-range-in-nargs
unutbu
'''
def required_length(nmin,nmax):
    class RequiredLength(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            if not nmin<=len(values)<=nmax:
                msg='argument "{f}" requires between {nmin} and {nmax} arguments'.format(
                    f=self.dest,nmin=nmin,nmax=nmax)
                raise argparse.ArgumentTypeError(msg)
            setattr(args, self.dest, values)
    return RequiredLength

def arg_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description='BLINK discretizes mass spectra (given .mgf inputs), and scores discretized spectra (given .npz inputs)')

    parser.add_argument('files',nargs='+', action=required_length(1,2), metavar='F', help='files to process')

    #Discretize options
    discretize_options = parser.add_argument_group()
    discretize_options.add_argument('--trim', action='store_true', default=False, required=False,
                                 help='remove empty spectra when discretizing')
    discretize_options.add_argument('-b','--bin_width', type=float, metavar='B', default=.001, required=False,
                                 help='width of bins in mz')
    discretize_options.add_argument('-i','--intensity_power', type=float, metavar='I', default=.5, required=False,
                                 help='power to raise intensites to in when scoring')

    #Compute options
    compute_options = parser.add_argument_group()
    compute_options.add_argument('-t','--tolerance', type=float, metavar='T', default=.01, required=False,
                                 help='maximum tolerance in mz for fragment ions to match')
    compute_options.add_argument('-d','--mass_diffs', type=float, metavar='D', nargs='*', default=[0], required=False,
                              help='mass diffs to network')
    compute_options.add_argument('-r','--react_dist', type=int, metavar='R', default=1, required=False,
                              help='recursively combine mass_diffs within reaction distance')
    compute_options.add_argument('-s','--min_score', type=float, default=.4, metavar='S', required=False,
                                 help='minimum score to include in output')
    compute_options.add_argument('-m','--min_matches', type=int, default=3, metavar='M', required=False,
                                 help='minimum matches to include in output')

    #Output file options
    output_options = parser.add_argument_group()
    output_options.add_argument('-f', '--force', action='store_true', required=False,
                                help='force file(s) to be remade if they exist')
    output_options.add_argument('-o','--out_dir', type=str, metavar='O', required=False,
                                help='change output location for output file(s)')

    return parser

def main():
    parser = arg_parser()
    args = parser.parse_args()

    logging.basicConfig(filename=os.path.join(os.getcwd(),'blink.log'), level=logging.INFO)

    common_ext = {os.path.splitext(in_file)[1] for f in args.files for in_file in glob.glob(f)}
    if len(common_ext) == 1:
        common_ext = list(common_ext)[0]

    if common_ext == '.mgf':
        logging.info('Discretize Start')

        prefix = os.path.commonprefix([os.path.basename(in_file)
                                         for input in args.files
                                         for in_file in glob.glob(input)])

        out_name = os.path.splitext(os.path.splitext(prefix)[0])[0]

        if out_name == '':
            out_name = '_'.join([os.path.splitext(os.path.splitext(os.path.basename(in_file))[0])[0]
                                 for input in args.files
                                 for in_file in glob.glob(input)])
        # if prefix != suffix:
        #     outname.append('concat')
        # if args.trim:
        #     outname.append('trim')
        # out_name.append(os.path.splitext(os.path.splitext(suffix)[0])[0])
        # out_name.append(str(args.bin_width).replace('.', 'p'))
        # out_name.append(str(args.intensity_power).replace('.', 'p'))
        #
        # out_name = '_'.join(out_name)

        if args.out_dir:
            out_dir = args.out_dir
        else:
            out_dir = os.path.dirname(os.path.abspath(glob.glob(args.files[0])[0]))

        logging.info('Output to {}'.format(out_dir))
        out_loc = os.path.join(out_dir, out_name+'.npz')

        if not args.force and os.path.isfile(out_loc):
            logging.info('{} already exists. Skipping.'.format(out_name))
            logging.info('Discretize End\n')
            sys.exit(0)

        dense_spectra =[open_msms_file(ff)[['spectrum','precursor_mz']]
                        for f in args.files
                        for ff in glob.glob(f)]
        file_ids = np.cumsum(np.array([s.spectrum.shape[0] for s in dense_spectra]))
        pmzs = np.concatenate([s.precursor_mz for s in dense_spectra]).tolist()
        dense_spectra = np.concatenate([s.spectrum for s in dense_spectra])

        start = timer()
        S = discretize_spectra(dense_spectra,pmzs=pmzs,bin_width=args.bin_width,intensity_power=args.intensity_power)
        end = timer()

        S['file_ids'] = file_ids

        if args.trim:
            zero_idxs = np.diff(S['indptr']) == 0
            logging.info('Trimmed {} rows: {}'.format(zero_idxs.sum(), ' '.join(np.where(zero_idxs)[0].tolist())))
            S['indptr'] = np.unique(S['indptr'])
            S['blanks'] = np.where(zero_idxs)[0]

        write_sparse_msms_file(out_loc, S)

        logging.info('Discretize Time: {} seconds, {} spectra'.format(end-start, len(S['indptr'])))
        logging.info('Discretize End\n')

    elif common_ext == '.npz':
        logging.info('Score Start')

        out_name = '_'.join([os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0] for f in args.files])

        if args.out_dir:
            out_dir = args.out_dir
        else:
            out_dir = os.path.dirname(os.path.abspath(args.files[0]))
        logging.info('Output to {}'.format(out_dir))

        out_loc = os.path.join(out_dir, out_name)

        if not args.force and os.path.isfile(out_loc):
            logging.info('{} already exists. Skipping.'.format(out_name))
            logging.info('Score End\n')
            sys.exit(0)

        S1 = open_sparse_msms_file(args.files[0])
        bin_width = S1['bin_width']
        S1_blanks = S1.get('blanks',np.array([]))

        S1 = expand_sparse_spectra(**S1)
        if len(args.files) == 1:
            S2 = S1
            S2_blanks = S1_blanks
        else:
            S2 = open_sparse_msms_file(args.files[1])
            S2_blanks = S1.get('blanks',np.array([]))

            try:
                assert S2['bin_width'] == bin_width
            except AssertionError:
                log.error('Input files have differing bin_width')
                sys.exit(1)
            S2 = expand_sparse_spectra(**S2)

        N = network_kernel(S1['mzi'].shape[1], S2['mzi'].shape[1],
                           args.mass_diffs, react_dist=args.react_dist,
                           bin_width=bin_width, tolerance=args.tolerance)

        S12 = {}
        for k in set(S1.keys()) & set(S2.keys()):
            v1, v2 = S1[k], S2[k]
            start = timer()
            S12[k] = v1.dot(N).dot(v2.T)
            end = timer()
            logging.info('Score Time: {} seconds'.format(end-start))

        if (args.min_score > 0) or (args.min_matches > 0):
            logging.info('Filtering')
            keep_idx =  S12['mzi'] >= args.min_score
            keep_idx = keep_idx.maximum(S12['mzc'] >= args.min_matches)
            if 'nli' in S12.keys():
                keep_idx = keep_idx.maximum(S12['nli'] >= args.min_score)
            if 'nlc' in S12.keys():
                keep_idx = keep_idx.maximum(S12['nlc'] >= args.min_matches)

            for k in S12.keys():
                S12[k] = S12[k].multiply(keep_idx).tocoo()

        out_df = pd.concat([pd.Series(S12[k].data, name=k,
                                      index=list(zip(S12[k].col.tolist(),
                                                     S12[k].row.tolist())))
                            for k in S12.keys()], axis=1)

        out_df.index.names = ['/'.join([str(args.tolerance),
                                        ','.join([str(d) for d in args.mass_diffs]),
                                        str(args.react_dist),
                                        str(args.min_score),
                                        str(args.min_matches)]),'']

        out_df.to_csv(out_loc+'.tab', index=True, sep='\t', columns = sorted(out_df.columns,key=lambda c:c[::-1])[::-1])

        logging.info('Score End\n')

    else:
        logging.error('Input files must only be .mgf or .npz')
        sys.exit(1)

if __name__ == '__main__':
    main()
