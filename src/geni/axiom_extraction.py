import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from itertools import permutations
from scipy.spatial import distance

def _get_symmetric(th_value,embeddings, current_rel,type):
    curr_emb=embeddings[current_rel]
    curr_emb = curr_emb.astype(np.float64)
    if type=='translation':
        zeros=np.zeros(curr_emb.shape[0])
        sum=curr_emb+curr_emb
        if distance.euclidean(sum,zeros)<= th_value:
            return True
        else:
            return False

    elif type=='semantic':
        identity=np.identity(curr_emb.shape[0],dtype=np.float64)
        if np.linalg.norm(abs(identity-curr_emb))<=th_value:
            return True
        else:
            return False

def _get_transitive(th_value,embeddings, current_rel,type):
    curr_emb=embeddings[current_rel]
    curr_emb = curr_emb.astype(np.float64)
    if type=='translation':
        sum=curr_emb+curr_emb
        sum=sum.reshape(1,curr_emb.shape[0])
        curr_emb=curr_emb.reshape(1,curr_emb.shape[0])
        if cosine_similarity(sum,curr_emb) <=th_value:
            return True
        else:
            return False

    elif type=='semantic':
        mul=np.matmul(curr_emb,curr_emb,dtype=np.float64)
        if np.linalg.norm(abs(mul-curr_emb))<=th_value:
            return True
        else:
            return False

def _get_equivalent(th_value,embeddings,current_rel,type,search_space):
    curr_emb=embeddings[current_rel]
    if type=='translation':
        eqs=[]
        for r in search_space:
            opt_emb=embeddings[r]
            if distance.euclidean(curr_emb,opt_emb)<=th_value:
                eqs.append(r)

    elif type=='semantic':
        eqs=[]
        for r in search_space:
            opt_emb=np.array(embeddings[r],dtype=np.float64)
            if np.linalg.norm(abs(curr_emb-opt_emb))<=th_value:
                eqs.append(r)
    return eqs

def _get_inverse(th_value,embeddings,current_rel,type,search_space):
    curr_emb = embeddings[current_rel]
    curr_emb = curr_emb.astype(np.float64)
    if type == 'translation':
        invs = []
        for r in search_space:
            opt_emb = -1*embeddings[r]
            if distance.euclidean(curr_emb,opt_emb) <= th_value:
                invs.append(r)

    elif type == 'semantic':
        invs = []
        identity=np.identity(curr_emb.shape[0],dtype=np.float64)
        for r in search_space:
            opt_emb = np.array(embeddings[r],dtype=np.float64)
            comp=np.matmul(curr_emb,opt_emb,dtype=np.float64)
            if np.linalg.norm(abs(comp-identity)) <=th_value:
                invs.append(r)
    return invs

def _get_chain(th_value,embeddings,current_rel,type,search_space):
    curr_emb = embeddings[current_rel]
    curr_emb = curr_emb.astype(np.float64)
    if type == 'translation':
        chains = []
        search_space.append(current_rel)
        combs=permutations(search_space,2)
        for tuple in combs:
            r1_emb=embeddings[tuple[0]]
            r2_emb=embeddings[tuple[1]]
            res=r1_emb+r2_emb
            if distance.euclidean(res,curr_emb) <=th_value:
                chains.append(tuple)

    elif type == 'semantic':
        chains = []
        search_space.append(current_rel)
        perms=permutations(search_space,2)
        for tuple in perms:
            r1_emb=embeddings[tuple[0]]
            r2_emb=embeddings[tuple[1]]
            res=np.matmul(r1_emb,r2_emb,dtype=np.float64)
            curr_emb=np.array(curr_emb,dtype=np.float64)
            if np.linalg.norm(abs(res-curr_emb)) <=th_value:
                chains.append(tuple)
    return chains

def get_axioms(th_value, embeddings,current_rel,type,search_space):
    axioms={'symmetric':False,'transitive':False,'equivalent':[],'inverse':[],'chain':[]}
    axioms['symmetric']=_get_symmetric(th_value,embeddings,current_rel,type)
    axioms['transitive']=_get_transitive(th_value,embeddings,current_rel,type)
    axioms['equivalent']=_get_equivalent(th_value,embeddings,current_rel,type,search_space)
    axioms['inverse']=_get_inverse(th_value,embeddings,current_rel,type,search_space)
    axioms['chain']=_get_chain(th_value,embeddings,current_rel,type,search_space)
    return axioms

def evaluate_fact_axiom(cur_fact,axioms,facts,dataset):
    triggered_rules = {'symmetric': False, 'transitive': [], 'equivalent': [], 'inverse': [], 'chain': []}
    for type,ax in axioms.items():
        if type=='symmetric' and ax is True:
            search=facts[cur_fact[2]]
            if (cur_fact[1],cur_fact[0]) in search:
                triggered_rules['symmetric']=True
                break
        elif type=='transitive' and ax is True:
            search=facts[cur_fact[2]]
            head_values=[i[0] for i in search if i[1]==cur_fact[1]]
            tail_values=[i[1] for i in search if i[0]==cur_fact[0]]
            if set(head_values) & set(tail_values):
                b=list(set(head_values)&set(tail_values))
                triggered_rules['transitive']=b
                break
        elif type=='equivalent' and ax!=[]:
            for r in ax:
                search=facts[r]
                if (cur_fact[0],cur_fact[1]) in search:
                    if r not in triggered_rules['equivalent']:
                        triggered_rules['equivalent'].append(r)
                        break
        elif type=='inverse' and ax!=[]:
            for r in ax:
                search=facts[r]
                if (cur_fact[1],cur_fact[0]) in search:
                    if r not in triggered_rules['inverse']:
                        triggered_rules['inverse'].append(r)
                    break
        elif type=='chain' and ax!=[]:
            for r_chain in ax:
                head_search=facts[r_chain[0]]
                tail_search=facts[r_chain[1]]
                head_values = [i[1] for i in head_search if i[0] == cur_fact[0]] # tutte le tail delle triple che hanno come head il soggetto della predizione
                tail_values = [i[0] for i in tail_search if i[1] == cur_fact[1]] # tutte le head delle triple che hanno come tail l'oggetto della predizione
                if set(head_values)&set(tail_values):
                    a=list(set(head_values)&set(tail_values))
                    if r_chain not in triggered_rules['chain']:
                        triggered_rules['chain'].append({"predicates": r_chain, "entities": a})
                    break

    return triggered_rules

