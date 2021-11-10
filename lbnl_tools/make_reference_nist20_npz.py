import numpy as np
import pandas as pd

import sys
sys.path.insert(0,'/global/homes/b/bpb/repos/blink')
import blink

reference_data = '/global/cfs/cdirs/metatlas/projects/spectral_libraries/all_nist20_pos.mgf'
# reference_data = '/global/cscratch1/sd/bpb/simile-paper/pos_nist20_spectra_2.mgf'
df = blink.open_msms_file(reference_data)
# filter precursor mz greater
s = []
for i,row in df.iterrows():
    idx = np.argwhere(row['spectrum'][0]<(row['precursor_mz']+0.20)).flatten()
    s.append(row['spectrum'][:,idx])
df['spectrum'] = s
df['sum_spectra'] = df['spectrum'].apply(lambda x: sum(x[1]))
df['num_ions'] = df['spectrum'].apply(lambda x: len(x[1]))
df = df[df['sum_spectra']>0]

# from rdkit.Chem import MolFromInchi,ReplaceSidechains,MolToSmiles
# cpds = pd.read_csv('/global/cfs/cdirs/metatlas/projects/unique_compounds.csv.gz',usecols=['inchi_key','inchi','name','mono_isotopic_molecular_weight'])
# ik_lookup = pd.read_csv('/global/homes/b/bpb/repos/simile/nist20_inchikey_to_integer.csv')
# ik_lookup = pd.merge(cpds,ik_lookup,left_on='inchi_key',right_on='inchikey',how='inner')
# ik_lookup.set_index('key',drop=True,inplace=True)
# ik_lookup.drop(columns='inchikey',inplace=True)

df['inchi_key'] = df['id'].apply(lambda x: x.split('\t')[1])
df['adduct'] = df['id'].apply(lambda x: x.split('\t')[2])
df['collision_energy'] = df['id'].apply(lambda x: x.split('\t')[3])
df['instrument'] = df['id'].apply(lambda x: x.split('\t')[4])
df['id'] = df['id'].apply(lambda x: int(x.split('\t')[0]))

# df = pd.merge(df,ik_lookup,left_on='id',right_index=True,how='left')

S2 = blink.discretize_spectra(df['spectrum'].tolist(),
                              pmzs=df['precursor_mz'].tolist(),
                              remove_duplicates=False,
                              metadata=df.drop(columns=['pepmass','spectrum','name']).to_dict(orient='records'))

sparse_nist_file = '/global/cfs/cdirs/metatlas/projects/spectral_libraries/nist20_sparse_discretized_spectra.npz'
blink.write_sparse_msms_file(sparse_nist_file,S2)
