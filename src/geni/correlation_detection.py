import numpy as np
from fastdist import fastdist
from .utils import get_clustered_elements

from scipy.spatial.distance import cdist

def find_entity_correlations(cur_ent,ent_dict,c_labels,th_value):
    search_space = get_clustered_elements(c_labels, ent_dict, cur_ent)
    corrs=[]
    for s in search_space:
        corr=fastdist.euclidean(ent_dict[cur_ent],ent_dict[s])
        if abs(corr) <= th_value:
            corrs.append((s,corr))
    return corrs


def find_direct_correlations(cur_ent,cur_rel,corrs,rel_embeddings,ent_embeddings,type,th_value):
    rel_matrix = np.array(list(rel_embeddings.values())) # (|R|, dim)
    direct_corrs=[]
    for e in corrs:
        if type=='semantic':
            matrix_1=np.matmul(rel_embeddings[cur_rel],ent_embeddings[cur_ent],dtype=np.float64)
            matrix_2=np.matmul(rel_matrix,ent_embeddings[e[0]],dtype=np.float64)
        elif type=='translation':
            matrix_1=(rel_embeddings[cur_rel]+ent_embeddings[cur_ent]).astype(np.float64)
            matrix_2=(rel_matrix+ent_embeddings[e[0]]).astype(np.float64)

        if matrix_1.shape == ():
            return direct_corrs 
        diff = cdist([matrix_1], matrix_2, metric='euclidean') 
        
        max_corr_idx = np.unravel_index(np.argmin(diff, axis=None), diff.shape)
        r2 = np.array(rel_embeddings[list(rel_embeddings.keys())[max_corr_idx[0]]],dtype=np.float64)
        if type=='semantic':
            u2=np.matmul(r2,ent_embeddings[e[0]],dtype=np.float64)
        elif type=='translation':
            u2=r2+ent_embeddings[e[0]]
        corr=fastdist.euclidean(matrix_1,u2)
        if abs(corr) <= th_value:
            direct_corrs.append((e[0],list(rel_embeddings.keys())[max_corr_idx[0]],corr))
    return direct_corrs

def find_triangular_correlations(cur_ent,cur_rel,rel_embeddings,ent_embeddings,type_,th_value):
    ent_matrix=np.array(list(ent_embeddings.values()),dtype=np.float64)
    direct_corrs = []
    if type_ == 'semantic':
        matrix_1 = np.matmul(rel_embeddings[cur_rel], ent_embeddings[cur_ent],dtype=np.float64)
    elif type_ == 'translation':
        matrix_1 = rel_embeddings[cur_rel] + ent_embeddings[cur_ent]

    if matrix_1.shape == ():
        return direct_corrs 

    diff = cdist([matrix_1], ent_matrix, metric='euclidean') 
    max_corr_idx = np.unravel_index(np.argmin(diff, axis=None), diff.shape)
    corr=diff[max_corr_idx]
    triangular_corr = list(ent_embeddings.keys())[max_corr_idx[0]]
    if abs(corr) <= th_value:
        direct_corrs.append((triangular_corr,corr))
    return direct_corrs

def evaluate_direct_correlations(correlation_list,facts):
    corrs={}
    for corr in correlation_list:
        search = facts[corr[1]]
        tails=[i[1] for i in search]
        if corr[0] in tails:
            corrs[corr]=1
        else:
            corrs[corr]=0
    return corrs

def evaluate_triangular_correlations(s, correlation_list, facts):
    corrs = {}
    for corr in correlation_list:
        search=[(v[0],k,v[1]) for k,f in facts.items() for v in f if v and(v[0]==corr[0] or v[1]==corr[0])]
        head_ents=[i[0] for i in search]
        tail_ents=[i[2] for i in search]
        if s in head_ents or s in tail_ents:
            corrs[corr]=1
        else:
            corrs[corr]=0
    return corrs

