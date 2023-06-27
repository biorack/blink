import argparse
import os

def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise NotADirectoryError(string)

def create_rem_parser():

    parser = argparse.ArgumentParser(description='REM-BLINK Efficiently Performs Analog Searches Accounting For Multiple Mass Differences')

    parser.add_argument('query_file', action='store', type=file_path, metavar='Q', help='MGF or mzML files containing experimental spectra')
    parser.add_argument('reference_file', action='store', type=file_path, metavar='R', help='MGF or mzML files containing reference spectra')
    parser.add_argument('output_file', action='store', type=str, metavar='O', help='path to output file')
    parser.add_argument('pos_model_path', action='store', type=file_path, metavar='PM', help='path to REM-BLINK model trained for positive spectral comparisons')
    parser.add_argument('neg_model_path', action='store', type=file_path, metavar='NM', help='path to REM-BLINK model trained for negative spectral comparisons')
    parser.add_argument('polarity', action='store', type=str, metavar='P', choices=['positive', 'negative'], help='ion mode of query spectra. Determines the model used.')
    parser.add_argument('-mds', '--mass_diffs', nargs='+', type=float, required=True, action='store', metavar='MD', help='m/z differences used to calculate score vector. Must match those used to train model')

    #Discretize options
    discretize_options = parser.add_argument_group()
    discretize_options.add_argument('-t', '--tolerance', action='store', type=float, default=0.01, required=False,
                                    help='allowed dalton tolerance between matched MS/MS fragment peaks')
    
    discretize_options.add_argument('-b','--bin_width', action='store', type=float, default=0.001, required=False,
                                 help='width of bins in m/z. Larger bins will be faster at the expense of precision.')
    
    discretize_options.add_argument('-i','--intensity_power', action='store', type=float, default=0.5, required=False,
                                 help='exponent used to adjust intensities prior to scoring')
    
    discretize_options.add_argument('--trim', action='store_true', default=False, required=False,
                                    help='remove empty spectra when discretizing')
    
    discretize_options.add_argument('--dedup', action='store_true', default=False, required=False,
                                    help='deduplicate fragment ions within 2 times bin_width')

    #Filtering options
    filter_options = parser.add_argument_group()

    filter_options.add_argument('-s','--min_score', type=float, default=0.2, required=False,
                                 help='minimum cosine score to include in output. This should be lower than is typical since further filtering occurs after REM prediction.')
    filter_options.add_argument('-m','--min_matches', type=int, default=3, required=False,
                                 help='minimum matches to include in output. This should be lower than is typical since further filtering occurs after REM prediction.')
    filter_options.add_argument('-o', '--override_matches', type=int, default=5, required=False,
                                help='number of matches to keep comparison regardless of score')
    filter_options.add_argument('-p', '--min_predict', type=float, default=0.005, required=False,
                                help='minimum REM-BLINK predicted score to include in output.')

    return parser