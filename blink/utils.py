import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def maximum_entropy_normalization(y):
    #MaximumEntropyNormalization   Normalize Data using principle of maximum entropy.
    # n=1/(1+exp(-(y-mean(y))/std(y)));
    m=y.mean()
    s=y.std()
    n=1/(1+np.exp(-1*(y-m)/s))
    return n

def get_topk_blink_matrix(D,k=5,score_col=4,query_col=1):
    """
    sort by score_col and query_col
    get top k for each query_col entry

    returns filtered blink matrix
    """

    # Do a quick hand calc to make sure this is working:
    # https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.ndarray.sort.html
    D = D[np.lexsort((D[:, query_col], -D[:, score_col])),:]
    # idx = np.argsort(D[:,score_col])[::-1] #sort in descending order by blink score
    # D = D[idx,:]
    # idx = np.argsort(D[:,query_col]) # sort in ascending order by query number
    # D = D[idx,:]



    #return indices of first instance of each query number
    u,u_idx = np.unique(D[:,query_col],return_index=True)

    #for each query number get k next best hits
    hits = []
    for i,idx in enumerate(u_idx[:-1]):
        if (u_idx[i+1]-idx)>=k: #check the >=
            hits.append(np.arange(k)+idx)
        else:
            hits.append(np.arange(u_idx[i+1]-idx)+idx)
    hits = np.concatenate(hits)
    D = D[hits,:]
    return D

def filter_component_additive(G, max_component_size,edge_score='score'):
    if max_component_size == 0:
        return

    all_edges = list(G.edges(data=True))
    G.remove_edges_from(list(G.edges))

    all_edges = sorted(all_edges, key=lambda x: x[2][edge_score], reverse=True)
    counter = 0
    for i,edge in enumerate(all_edges):
        G.add_edges_from([edge])
        largest_cc = max(nx.connected_components(G), key=len)
        if len(largest_cc) > max_component_size:
            G.remove_edge(edge[0], edge[1])
        counter +=1
        if counter==1000:
            print(i,len(list(G.edges(data=True))))
            counter=0
        
def filter_top_k(G, top_k,edge_score='score'):
    print("Filter Top_K", top_k)
    #Keeping only the top K scoring edges per node
    print("Starting Numer of Edges", len(G.edges()))

    node_cutoff_score = {}
    for node in G.nodes():
        node_edges = G.edges((node), data=True)
        node_edges = sorted(node_edges, key=lambda edge: edge[2][edge_score], reverse=True)

        edges_to_delete = node_edges[top_k:]
        edges_to_keep = node_edges[:top_k]

        if len(edges_to_keep) == 0:
            continue

        node_cutoff_score[node] = edges_to_keep[-1][2][edge_score]

        #print("DELETE", edges_to_delete)


        #for edge_to_remove in edges_to_delete:
        #    G.remove_edge(edge_to_remove[0], edge_to_remove[1])


    #print("After Top K", len(G.edges()))
    #Doing this for each pair, makes sure they are in each other's top K
    edge_to_remove = []
    for edge in G.edges(data=True):
        cosine_score = edge[2][edge_score]
        threshold1 = node_cutoff_score[edge[0]]
        threshold2 = node_cutoff_score[edge[1]]

        if cosine_score < threshold1 or cosine_score < threshold2:
            edge_to_remove.append(edge)

    for edge in edge_to_remove:
        G.remove_edge(edge[0], edge[1])

    print("After Top K Mutual", len(G.edges()))

def make_mirror_plot(r,q,figsize=(12,6),mz_tol=0.01,color='black',match_color='red',fontsize=20,grid=True,ax=None,fig=None,vshade=None):
    """
    Make a mirror plot
    You should normalize the spectra before passing to this.
    q = query spectrum np.array([mzs,intensities])
    r = reference spectrum np.array([mzs,intensities])
    returns figure and axes handles
    """
    # return_fig = False
    if ax is None:
        fig,ax = plt.subplots(figsize=figsize)
        # return_fig = True
        
    dd = abs(np.subtract(r[0][None,:], q[0][:, None],))<mz_tol

    ax.vlines(q[0],q[1]*0,q[1],color,alpha=0.6,linewidth=2)
    idx = dd.any(axis=1)
    ax.vlines(q[0][idx],q[1][idx]*0,q[1][idx],match_color,linewidth=2)

    ax.vlines(r[0],r[1]*0,-1*r[1],color,alpha=0.6,linewidth=2)
    idx = dd.any(axis=0)
    ax.vlines(r[0][idx],r[1][idx]*0,-1*r[1][idx],match_color,linewidth=2)
    ax.set_ylabel('Normalized intensity (au)',fontsize=fontsize)
    ax.set_xlabel('m/z',fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=(fontsize-2))
    if vshade is not None:
        ax.axvline(vshade[0],linewidth=vshade[1],alpha=vshade[2],color=vshade[3])
    if grid==True:
        ax.grid()
    # if return_fig==True
    # return fig,ax