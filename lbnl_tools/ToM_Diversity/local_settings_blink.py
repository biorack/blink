import sys
import os

metatlas_compound_filename='/global/cfs/cdirs/metatlas/projects/unique_compounds.csv.gz'

berklab_ref_spectra_filename='/global/cfs/cdirs/metatlas/projects/spectral_libraries/BERKELEY-LAB.mgf'
berklab_mcs_filename = '/global/homes/b/bpb/repos/blink/lbnl_tools/ToM_Diversity/berklab-df.pkl'
berklab_compound_metadata_filename = '/global/homes/b/bpb/repos/blink/lbnl_tools/tree_metadata.csv'
berklab_tree_filename = '/global/homes/b/bpb/repos/blink/lbnl_tools/ToM.newick'

path_to_tom_diversity_code = '/global/homes/b/bpb/repos/blink/lbnl_tools/ToM_Diversity/'
path_to_blink_code = '/global/homes/b/bpb/repos/blink/'

min_matches=5
good_score=0.7
good_matches=20
calc_network_score=True
        
sys.path.insert(0,path_to_blink_code)
sys.path.insert(1,path_to_tom_diversity_code)