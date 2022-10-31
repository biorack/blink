import pickle
import pandas as pd
import sys
import pickle

from rdkit.Chem import InchiToInchiKey, MolFromSmiles

from scipy.cluster.hierarchy import linkage, dendrogram
import skbio
import scipy.sparse as sp
import numpy as np
    
class metabolite_diversity(object):
    import local_settings_blink as params
    # import blink here since path would have just been added by local settings
    import blink
    
    def __init__(self,**kwargs):
        self.__dict__.update(kwargs)

    def get_treecompound_metadata(self):
        df = pd.read_csv(self.params.berklab_compound_metadata_filename)
        self.compound_metadata = df
        return self  
    
    def make_linkage_from_mcs_matrix(self):
        comparisons = pd.read_pickle(self.params.berklab_mcs_filename)
        num_mols = comparisons[['key1','key2']].max().max() + 1
        n = sp.csr_matrix((comparisons['jaccard_bonds'].values,(comparisons['key1'].values,comparisons['key2'].values)),shape=(num_mols,num_mols),dtype=float)        
        n.setdiag(1)
        n = n.todense()
        n = 1 - n
        i_lower = np.tril_indices(self.compound_metadata.shape[0], -1)
        n[i_lower] = n.T[i_lower]
        z = linkage(n,method='single')
        self.mcs_linkage = z
        
        return self

    def make_skbio_tree(self):
        self.node_ids = self.compound_metadata['inchi_key'].tolist()
        self.ToM = skbio.TreeNode.from_linkage_matrix(self.mcs_linkage, self.node_ids)    
        return self

    def blink_score_file(self):
        spectra_df = self.blink.open_msms_file(self.query_file)
        if 'spectrum' in spectra_df.columns:
            self.query_spectra = self.blink.discretize_spectra(spectra_df['spectrum'].tolist(),pmzs=spectra_df['precursor_mz'].tolist(),
                                     remove_duplicates=False,metadata=spectra_df.drop(columns=['spectrum']).to_dict(orient='records'))
            S12 = self.blink.score_sparse_spectra(self.query_spectra, self.reference_spectra)
            S12 = self.blink.filter_hits(S12,
                                         min_matches=self.params.min_matches,
                                         good_matches=self.params.good_matches,
                                         good_score=self.params.good_score,
                                         calc_network_score=self.params.calc_network_score)
            D = self.blink.create_blink_matrix_format(S12,self.params.calc_network_score)
            df = pd.DataFrame(D,columns=['raveled_index','query','ref','score','matches'])
            df = pd.merge(df,pd.DataFrame(S12['S1_metadata']).add_suffix('_query'),left_on='query',right_index=True,how='left')
            df = pd.merge(df,pd.DataFrame(list(S12['S2_metadata'])).add_suffix('_ref'),left_on='ref',right_index=True,how='left')
            df.sort_values('score',ascending=False,inplace=True)
            df.drop_duplicates('inchi_key_ref',inplace=True)
            self.num_unique_inchikey_hits = df.shape[0]
            temp = pd.DataFrame()
            temp['node_ids'] = self.node_ids
            df = pd.merge(temp,df,left_on='node_ids',right_on='inchi_key_ref',how='left')
            df['present'] = pd.notna(df['score'])
            idx = df['present']==True
            df.loc[idx,'present'] = 1
            idx = df['present']==False
            df.loc[idx,'present'] = 0
            self.num_unique_inchikey_hits_in_network = df.shape[0]
            self.hits = df
        return self

    def pd_diversity_file(self):
        diversity = skbio.diversity.alpha.faith_pd(self.hits['present'], self.node_ids, self.ToM,)
        self.pd_diversity = diversity
        return self
    
    def get_berklab_ref(self):
        ref = self.blink.open_msms_file(self.params.berklab_ref_spectra_filename)
        ref = ref[ref['ionmode']==self.polarity]
        ref['inchi_key'] = ref['inchi'].apply(lambda x: InchiToInchiKey(x))
        ref['num_ions'] = ref['spectrum'].apply(lambda x: len(x[0]))
        self.reference_spectra = self.blink.discretize_spectra(ref['spectrum'].tolist(),
                                                               pmzs=ref['precursor_mz'].tolist(),
                                                               remove_duplicates=False,
                                                               metadata=ref.drop(columns=['spectrum']).to_dict(orient='records'))
        return self


    
    
#     def calc_faithspd(self):
#         d = faith_pd(counts,ids,tree)
#         self.faith_pd = d
#         return self
#     def get_blink_hits():
#         hits = get_blink_hits(query,ref=ref,merge_metadata=False,calc_network_score=False,good_score=0.7,good_matches=10000)

#     def write_tree_newick(self):
#         tree.write(self.params.berklab_tree_filename,format='newick')
#     return self
    
#     def pickle_newick_tree(self):
#         sys.setrecursionlimit(100000)
#         with open('tree_nist20_compounds.pkl', 'wb') as handle:
#             pickle.dump(tree, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         return self