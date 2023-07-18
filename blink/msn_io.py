import numpy as np
import pandas as pd
from pyteomics import mgf
import networkx as nx
import pymzml

def _read_mzml(mzml_path: str) -> pd.DataFrame:
    """parses a given mzml file from path and returns a pandas DataFrame containing data

    For files with MS2, returns spectra, precursor m/z, intensity,
    and retention time

    For files with MS^n, returns the above plus relationships to the
    spectrum collection and what the particular precursor was.
    """
    def make_spectra(tuple_spectrum):
        mzs = []
        intensities = []
        for m,i in tuple_spectrum:
            mzs.append(m)
            intensities.append(i)
        np_spectrum = np.asarray([mzs,intensities])
        return np_spectrum

    precision_dict = {}
    for i in range(100):
        precision_dict[i] = 1e-5

    run = pymzml.run.Reader(mzml_path, MS_precisions=precision_dict)
    spectra = list(run)

    df = []
    for s in spectra:
        
        if s.ms_level is None:
            continue

        if s.ms_level>=2:
            for precursor_dict in s.selected_precursors:
                data = {'id':s.ID,
                        'ms_level':s.ms_level,
                        'rt':s.scan_time_in_minutes(),
                        'spectrum':s.peaks('centroided')}
                if precursor_dict['precursor id'] is not None:
                    for k in precursor_dict.keys():
                        data[k] = precursor_dict[k]
                df.append(data)

    df = pd.DataFrame(df)
    df.dropna(subset=['precursor id'],inplace=True)

    df['spectrum'] = df['spectrum'].apply(make_spectra)
    df['id'] = df['id'].astype(int)
    df['precursor id'] = df['precursor id'].astype(int)

    if df['ms_level'].max()>2:
        G=nx.from_pandas_edgelist(df, source='precursor id', target='id')
        sub_graph_indices=list(nx.connected_components(G))
        sub_graph_indices = [(i, v) for i,d in enumerate(sub_graph_indices) for k, v in enumerate(d)]
        sub_graph_indices = pd.DataFrame(sub_graph_indices,columns=['spectrum_collection','id'])
        df = pd.merge(df,sub_graph_indices,left_on='id',right_on='id',how='left')
        prec_mz_df = df[df['ms_level']==2].copy()
        prec_mz_df.rename(columns={'mz':'root_precursor_mz',
                                   'i':'root_precursor_intensity',
                                   'rt':'root_precursor_rt'},inplace=True)
        df.drop(columns=['i'],inplace=True)
        df.rename(columns={'mz':'precursor_mz'},inplace=True)
        df = pd.merge(df,
                      prec_mz_df[['spectrum_collection',
                                  'root_precursor_mz',
                                  'root_precursor_intensity','root_precursor_rt']],
                      left_on='spectrum_collection',
                      right_on='spectrum_collection')
        df.rename(columns={},inplace=True)
        df.reset_index(inplace=True,drop=True)
    else:
        df.drop(columns=['precursor id'],inplace=True)
        df.rename(columns={'mz':'precursor_mz'},inplace=True)
        df.reset_index(inplace=True,drop=True)

    return df

def _read_mgf(mgf_path: str) -> pd.DataFrame:
    msms_df = []
    with mgf.MGF(mgf_path) as reader:
        for spectrum in reader:
            d = spectrum['params']
            d['spectrum'] = np.array([spectrum['m/z array'],
                                      spectrum['intensity array']])
            if 'precursor_mz' not in d:
                d['precursor_mz'] = d['pepmass'][0]
            else:
                d['precursor_mz'] = float(d['precursor_mz'])
            msms_df.append(d)
    msms_df = pd.DataFrame(msms_df)
    return msms_df