#!/usr/bin/python

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
def discretize_spectra(mzis, bin_width=.001, intensity_power=.5):
    spec_ids = np.concatenate([[i]*m.shape[1] for i,m in enumerate(mzis)]).astype(int)
    mzis = np.concatenate(mzis, axis=1)
    mz_bin_idxs =  np.rint(mzis[0]/bin_width).astype(int)
    num_bins = mz_bin_idxs.max()+1

    mzis[1] = mzis[1]**intensity_power

    s = sp.coo_matrix((mzis[1], (spec_ids, mz_bin_idxs)), (spec_ids[-1]+1, num_bins))
    s = s.multiply(1./norm(s, axis=1)[:,None]).tocsr()
    c = sp.coo_matrix((np.ones_like(mzis[1]), (spec_ids, mz_bin_idxs)), (spec_ids[-1]+1, num_bins))
    c = c.multiply(((c.getnnz(axis=1)**.5)/sp.linalg.norm(c, axis=1))[:,None]).tocsr()

    return s,c

def condense_sparse_spectra(sparse_spectra, bin_width=.001):
    mzis = [np.array([row.indices*bin_width,row.data])
            for row in sparse_spectra]
    return mzis

##########
# Kernel
##########
def neighborhood_kernel(n, m, bin_width=.001, tolerance=.01):
    data = np.ones(int(tolerance/bin_width)+1).tolist()
    data = data[::-1]
    data.extend(data[-2::-1])
    data = np.array(data)

    offsets = np.arange(len(data))-(len(data)//2)

    N = sp.diags(data, offsets, shape=(n, m), dtype=bool, format='csr')

    return N

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
            # d['neutral_losses'] = d['precursor_mz'] -
            msms_df.append(d)
    msms_df = pd.DataFrame(msms_df)
    return msms_df

def write_sparse_msms_file(in_file):
    pass

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
        return sp.load_npz(in_file).tocsr()
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

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('-d', '--discretize', nargs='+', required=False,
                      help='convert input file(s) into CSR matrix format (.npz)')
    # mode.add_argument('-c', '--concatenate', nargs='+', required=False,
    #                   help='concatenate CSR matrix files into single file')
    mode.add_argument('-t', '--trim', nargs='+', required=False,
                      help='remove empty spectra from CSR matrix file(s)')
    mode.add_argument('-s', '--score', nargs='+', action=required_length(1,2), required=False,
                      help='score CSR matrix file with itself or another')
    mode.add_argument('-b', '--benchmark', nargs='+', required=False,
                      help='benchmark BLINK performance vs MatchMS')

    #Compute options
    compute_options = parser.add_argument_group()
    compute_options.add_argument('--bin_width', type=float, default=.001, required=False,
                                 help='width of bins in mz')
    compute_options.add_argument('--intensity_power', type=float, default=.5, metavar='I', required=False,
                                 help='power to raise intensites to in when scoring')

    compute_options.add_argument('--tolerance', type=float, default=.01, required=False,
                                 help='maximum tolerance in mz for fragment ions to match')
    compute_options.add_argument('--min_score', type=float, default=.4, metavar='S', required=False,
                                 help='minimum scores to include in output')
    compute_options.add_argument('--min_matches', type=int, default=3, metavar='M', required=False,
                                 help='minimum matches to include in output')

    #Output file options
    output_options = parser.add_argument_group()
    output_options.add_argument('-f', '--force', action='store_true', required=False,
                                help='force file(s) to be remade if they exist')
    output_options.add_argument('--out_dir', type=str, required=False,
                                help='change output location for output file(s)')

    return parser

def main():
    parser = arg_parser()
    args = parser.parse_args()

    logging.basicConfig(filename=os.path.join(os.getcwd(),'blink.log'), level=logging.INFO)

    if args.discretize:
        logging.info('Discretize Start')
        total_time = 0
        number_of_spectra = 0
        for input in args.discretize:
            for in_file in glob.glob(input):
                out_name = '{}_{}_{}'.format(os.path.splitext(os.path.splitext(os.path.basename(in_file))[0])[0],
                                             str(args.bin_width).replace('.', 'p'),
                                             str(args.intensity_power).replace('.', 'p'))
                if args.out_dir:
                    out_dir = args.out_dir
                else:
                    out_dir = os.path.dirname(os.path.abspath(glob.glob(args.discretize[0])[0]))

                logging.info('Output to {}'.format(out_dir))
                out_loc = os.path.join(out_dir, out_name+'.npz')

                if not args.force and os.path.isfile(out_loc):
                    logging.info('{} already exists. Skipping.'.format(out_name))
                    continue
                try:
                    dense_spectra = open_msms_file(in_file).spectrum.tolist()

                    start = timer()
                    sparse_spectra,sparse_counts = discretize_spectra(dense_spectra,
                                                                      bin_width=args.bin_width,
                                                                      intensity_power=args.intensity_power)
                    end = timer()

                    total_time += end-start
                    number_of_spectra += len(dense_spectra)

                    sp.save_npz(out_loc, sparse_spectra+((0+1j)*sparse_counts.astype(complex)))
                except IOError:
                    continue
        end = timer()
        logging.info('Discretize Time: {} seconds, {} spectra'.format(total_time, number_of_spectra))
        logging.info('Discretize End\n')
    #
    # if args.concatenate:
    #     logging.info('Concatenate Start')
    #     prefix = os.path.commonprefix([os.path.basename(in_file)
    #                                      for input in args.concatenate
    #                                      for in_file in glob.glob(input)])
    #     suffix = os.path.commonprefix([os.path.basename(in_file)[::-1]
    #                                    for input in args.concatenate
    #                                    for in_file in glob.glob(input)])[::-1]
    #
    #     out_name = prefix.split('_')
    #     out_name.insert(1, 'concat')
    #
    #     if prefix != suffix:
    #         out_name += suffix.split('_')
    #
    #     out_name = '_'.join(out_name)
    #
    #     if args.out_dir:
    #         out_dir = args.out_dir
    #     else:
    #         out_dir = os.path.dirname(os.path.abspath(glob.glob(args.concatenate[0])[0]))
    #     logging.info('Output to {}'.format(out_dir))
    #
    #     out_loc = os.path.join(out_dir, out_name)
    #
    #     if not args.force and os.path.isfile(out_loc):
    #         logging.info('{} already exists. Skipping.'.format(out_name))
    #         logging.info('Concatenate End\n')
    #         sys.exit(0)
    #
    #     start = timer()
    #     indptr = sp.vstack([open_sparse_msms_file(in_file)
    #                                        for input in args.concatenate
    #                                        for in_file in glob.glob(input)])
    #     end = timer()
    #     logging.info('Concatenate Time: {} seconds'.format(end-start))
    #
    #     sp.save_npz(out_loc, sparse_spectra_counts)
    #
    #     logging.info('Concatenate End\n')

    if args.trim:
        logging.info('Trim Start')
        total_time = 0
        number_of_blanks = 0
        for input in args.trim:
            for in_file in glob.glob(input):
                out_name = os.path.splitext(os.path.splitext(os.path.basename(in_file))[0])[0].split('_')
                out_name.insert(-2, 'trim')
                out_name = '_'.join(out_name)

                if args.out_dir:
                    out_dir = args.out_dir
                else:
                    out_dir = os.path.dirname(os.path.abspath(glob.glob(args.trim[0])[0]))

                logging.info('Output to {}'.format(out_dir))
                out_loc = os.path.join(out_dir, out_name+'.npz')

                if not args.force and os.path.isfile(out_loc):
                    logging.info('{} already exists. Skipping.'.format(out_name))
                    continue
                try:
                    sparse_spectra_counts = open_sparse_msms_file(in_file)

                    start = timer()
                    zero_idxs = np.diff(sparse_spectra_counts.indptr) == 0
                    sparse_spectra_counts = sparse_spectra_counts[~zero_idxs]
                    end = timer()

                    number_of_blanks += zero_idxs.sum()
                    total_time += end-start

                    sp.save_npz(out_loc, sparse_spectra_counts)
                except IOError:
                    continue
        end = timer()
        logging.info('Trim Time: {} seconds, {} blanks trimmed'.format(total_time, number_of_blanks))
        logging.info('Trim End\n')

    if args.score:
        logging.info('Score Start')
        try:
            assert args.score[0].split('_')[-1:] == args.score[-1].split('_')[-1:]
        except AssertionError:
            log.error('Input files have differing bin_width')
            sys.exit(1)

        out_name = '{}_{}_{}_{}_{}'.format(os.path.splitext(os.path.splitext(os.path.basename(args.score[0]))[0])[0],
                                           os.path.splitext(os.path.splitext(os.path.basename(args.score[1]))[0])[0]
                                           if len(args.score) == 2 else '',
                                           str(args.tolerance).replace('.', 'p'),
                                           str(args.min_score).replace('.', 'p'),
                                           str(args.min_matches))

        if args.out_dir:
            out_dir = args.out_dir
        else:
            out_dir = os.path.dirname(os.path.abspath(args.score[0]))
        logging.info('Output to {}'.format(out_dir))

        out_loc = os.path.join(out_dir, out_name)

        if not args.force and os.path.isfile(out_loc):
            logging.info('{} already exists. Skipping.'.format(out_name))
            logging.info('Concatenate End\n')

        sparse_spectra_counts1 = open_sparse_msms_file(args.score[0])
        sparse_spectra1 = sparse_spectra_counts1.real
        sparse_counts1 = sparse_spectra_counts1.imag.astype(bool)

        if len(args.score) == 1:
            sparse_spectra2 = sparse_spectra1
            sparse_counts2 = sparse_counts1
        else:
            sparse_spectra_counts2 = open_sparse_msms_file(args.score[1])
            sparse_spectra2 = sparse_spectra_counts2.real
            sparse_counts2 = sparse_spectra_counts2.imag.astype(bool)

        bin_width = float(os.path.splitext(args.score[0])[0].split('_')[-2].replace('p', '.'))

        N = neighborhood_kernel(sparse_spectra1.shape[1], sparse_spectra2.shape[1], bin_width, args.tolerance)

        start = timer()
        scores = sparse_spectra1.dot(N).dot(sparse_spectra2.T)
        matches = sparse_counts1.dot(N).dot(sparse_counts2.T)
        end = timer()
        logging.info('Score Time: {} seconds'.format(end-start))

        if (args.min_score > 0) or (args.min_matches > 0):
            logging.info('Filtering')
            keep_idx = (scores >= args.min_score).maximum(matches >= args.min_matches)

            scores = scores.multiply(keep_idx)
            matches = matches.multiply(keep_idx)

        sp.save_npz(out_loc, scores+((0+1j)*matches.astype(complex)))

        logging.info('Score End\n')

    # if args.benchmark:
    #     from matchms import Spectrum
    #     from matchms.similarity import CosineGreedy
    #
    #     logging.info('Benchmark Start')
    #
    #     out_name = '{}_{}_{}_{}_{}'.format(os.path.splitext(os.path.splitext(os.path.basename(args.benchmark[0]))[0])[0],
    #                                        '',
    #                                        str(args.tolerance).replace('.', 'p'),
    #                                        '',
    #                                        '')
    #
    #     if args.out_dir:
    #         out_dir = args.out_dir
    #     else:
    #         out_dir = os.path.dirname(os.path.abspath(args.benchmark[0]))
    #     logging.info('Output to {}'.format(out_dir))
    #
    #     out_loc = os.path.join(out_dir, out_name)
    #
    #     dense_spectra = open_msms_file(args.benchmark[0]).spectrum.to_numpy()
    #
    #     sparse_spectra, sparse_counts = discretize_spectra(dense_spectra,
    #                                                       bin_width=args.bin_width,
    #                                                       intensity_power=args.intensity_power)
    #     dense_spectra = np.array([Spectrum(mz=msms[0],intensities=msms[1])
    #                               for msms in dense_spectra])
    #
    #     N = neighborhood_kernel(sparse_spectra.shape[1], sparse_spectra.shape[1], args.bin_width, args.tolerance)
    #
    #     sqrt_cosine = CosineGreedy(tolerance=args.tolerance+args.bin_width/2, intensity_power=args.intensity_power)
    #
    #     for log_iter in range(1,6):
    #         idxs = np.random.randint(0,len(dense_spectra),(3,10**log_iter))
    #
    #         for rep in range(3):
    #             sparse_spectra_sample = sparse_spectra[idxs[rep]]
    #             sparse_counts_sample = sparse_counts[idxs[rep]]
    #
    #             start = timer()
    #             scores = sparse_spectra_sample.dot(N).dot(sparse_spectra_sample.T)
    #             matches = sparse_counts_sample.dot(N).dot(sparse_counts_sample.T)
    #             end = timer()
    #
    #             logging.info('BLINK Time: {} spectra, {} seconds'.format(10**log_iter, end-start))
    #             sp.save_npz(out_loc+'_blink-benchmark-{}'.format(log_iter), scores+((0+1j)*matches.astype(complex)))
    #
    #         for rep in range(3):
    #             dense_spectra_sample = dense_spectra[idxs[rep]].tolist()
    #
    #             start = timer()
    #             scores_matches = sqrt_cosine.matrix(dense_spectra_sample, dense_spectra_sample, is_symmetric=True)
    #             end = timer()
    #
    #             logging.info('MatchMS Time: {} spectra, {} seconds'.format(10**log_iter, end-start))
    #             sp.save_npz(out_loc+'_matchms-benchmark-{}'.format(log_iter), sp.csr_matrix(scores_matches['score'])+((0+1j)*sp.csr_matrix(scores_matches['matches']).astype(complex)))
    #
    #     logging.info('Benchmark End\n')


if __name__ == '__main__':
    main()
