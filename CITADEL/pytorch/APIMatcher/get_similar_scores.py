import os
import csv
import pickle
import copy
import numpy as np
import json
import sys
sys.path.append('./dprl')
from classes.torch_api import TorchAPI, TorchArgument
from classes.argdef import ArgDef
from classes.database import TorchDatabase
from munkres import Munkres, print_matrix

def similar_score(string1, string2,method):
    from Levenshtein import distance
    if method=='intersection':
        count_inter=sum(1 for s1, s2 in zip(string1, string2) if s1 == s2 and s1 != '0')
        count1=string1.count('1')
        count2=string2.count('1')
        result=(count_inter*100/count1,count_inter*100/count2)

    elif method=='levenshtein':
        length = len(string1)
        new_string1 = ''.join([string1[i] for i in range(length) if string1[i] != string2[i] or string1[i] != '0'])
        new_string2 = ''.join([string2[i] for i in range(length) if string1[i] != string2[i] or string1[i] != '0'])
        l_distance=(1-distance(new_string1, new_string2)/(len(new_string2)))*100
        result=(l_distance,l_distance)# same/total
    elif method=='local':
        max_length=5#3
        count=0
        if len(string1)>=len(string2):
            test_s=string2
            test_l=string1
        else:
            test_s=string1
            test_l=string2
        max_length=min(max_length,len(test_s))
        test_s = list(map(str, test_s))
        test_l = list(map(str, test_l))
        long_seg='-'.join(test_l)
        for i in range(len(test_s)-max_length+1):
            string_seg='-'.join(test_s[i:i+max_length])
            if string_seg in long_seg:
                count+=1
        score=count/(len(test_s)-max_length+1)*100 # percent of local similarity in the smallest list
        result=(score,score)
            
    return result

def analyze_simi(relation_dict,save_path,method='intersection'):
    '''
    method: 
    'intersection': calculate the intersection/union of two sets through Jaccard similarity
    'levenshtein': levenshtein distance
    '''
    function_relation_dict={}
    function_relation_dict['api call']={}
    function_relation_dict['similarity score']=[]
    function_relation_dict['call stack']=copy.deepcopy(relation_dict['call_stack'])
    
    # step1: finish api call process
    func_name_list=[]
    relation_table=relation_dict['table']
    relation_vocab=relation_dict['vocabulary']
    function_relation_dict['vocabulary']=relation_vocab
    api_list=list(relation_table['api'])
    for i in range(len(relation_vocab)):
        func_name=relation_vocab[i].replace('test_','')
        string=''
        for a in api_list:
            if i in a:
                string+='1'
            else:
                string+='0'
        if '1' in string:    
            func_name_list.append(func_name)
            function_relation_dict['api call'][func_name]=string
            
    # step2: finish similarity score
    
    for i in range(len(func_name_list)):
        for j in range(i+1,len(func_name_list)):
            if method == 'intersection' or method =='levenshtein':
                score=similar_score(function_relation_dict['api call'][func_name_list[i]],
                            function_relation_dict['api call'][func_name_list[j]],
                            method=method)
                
            elif method =='local': #'structure'
                score=similar_score(function_relation_dict['call stack'][relation_vocab[i]],
                            function_relation_dict['call stack'][relation_vocab[j]],
                            method=method)
            function_relation_dict['similarity score'].append([(func_name_list[i],func_name_list[j]),score])
    
    with open(save_path, 'wb') as f:
        pickle.dump(function_relation_dict, f)
    
    return function_relation_dict

def update_similarity_score(score_list,save_path,method):
    if not os.path.exists(save_path):
        save_dict={}
    else:
        with open(save_path, 'rb') as f:#input,bug type,params
            save_dict = pickle.load(f)
    for sc in score_list:
        key=f'{sc[0][0]}-{sc[0][1]}'
        if key not in save_dict.keys():
            #'intersection' or method=='levenshtein' or method=='local'
            save_dict[key]={}
            save_dict[key]['intersection']=None
            save_dict[key]['levenshtein']=None
            save_dict[key]['local']=None
        save_dict[key][method]=np.mean(list(sc[1]))
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)

def extract2csv(function_relation_dict,save_dir,threshold=60,method=None):
    score_list=function_relation_dict['similarity score']
    
    update_similarity_score(score_list,save_path=os.path.join(save_dir,'total_similar_score.pkl'),method=method)
    # score_list[i][0] is name list, contains 2 names; 
    # score_list[i][1] is score tuple, contains 2 names; 
    
    output_list=[]
    ready_list=[]
    
    avg_score_list=[]
    
    for i in score_list:
        avg_score_list.append(np.mean(list(i[-1])))
        if method=='intersection' or method=='levenshtein' or method=='local':
            if min(i[-1])>=threshold:
                output_list.append([i[0],i[-1]])
    return output_list

def cluster_relation_pairs(relation_list,save_path,value):
    # cluster relation list roughly
    final_save_path=save_path.replace('.pkl','_{}_{}.pkl'.format(method,value))
    if os.path.exists(final_save_path):
        return None,None
    new_list=[]
    r_length=len(relation_list)
    tmp_list=[]
    result_dict={}
    result_dict['similar_functions']={}
    relation_dict=result_dict['similar_functions']
    for i in range(r_length):
        name1=relation_list[i][0][0]
        name2=relation_list[i][0][1]
        score=relation_list[i][1]
        tmp_list.append([name1,name2])
        
        for n in range(len(relation_list[i][0])):
            if relation_list[i][0][n] not in relation_dict.keys():
                relation_dict[relation_list[i][0][n]]=[]
            relation_dict[relation_list[i][0][n]].append(relation_list[i][0][n-1])
    
    # # save
    # result_dict['relation_list']=relation_list
    with open(final_save_path, 'wb') as f:
        pickle.dump(result_dict, f)
    
    
    return relation_dict


def remove_useless_operation(relation_dict,threshold=30):
    # we consider that the mostly used operations can not represent a function/api's characteristic,therefore 
    # we removed them here to reduce their impact.
    tmp_table=relation_dict['table']
    length=tmp_table.shape[0]
    drop_list=[]
    for i in range(length):
        if len(tmp_table.loc[i,'api'])>=threshold:
            drop_list.append(i)
    new_table=tmp_table.drop(labels=drop_list,axis=0) 
    relation_dict['table']=new_table
    if method=='local':# clean call stack
        for key in relation_dict['call_stack'].keys():
            old_list=relation_dict['call_stack'][key]
            new_list=[]
            for ele in old_list:
                if ele not in drop_list:
                    new_list.append(ele)
            relation_dict['call_stack'][key]=new_list

    return relation_dict

def get_raw_similarity(relation_path,save_dir,final_save_path,method,threshold):
    with open(relation_path, 'rb') as f:#input,bug type,params
        relation_dict = pickle.load(f)
    if method=='intersection' or method=='levenshtein':# or method=='local'
        relation_dict=remove_useless_operation(relation_dict)
    print(f'=====Computing Similarity Score with {method} Method=====')
    function_relation_path=os.path.join(save_dir,'finegrained_relation-{}.pkl'.format(method))
    function_relation_dict=analyze_simi(relation_dict,function_relation_path,method=method)
    
    with open(function_relation_path, 'rb') as f:#input,bug type,params
        function_relation_dict = pickle.load(f)
    print(f'=====Analyzing Similarity=====')
    function_relation_list=extract2csv(function_relation_dict,save_dir,threshold=threshold,method=method)
    print(f'=====Updating Raw Similarity Relationship=====')
    relation_dict=cluster_relation_pairs(function_relation_list,final_save_path,value)

def combine_similairy(path_list,final_combine_path):
    combine_dict={}
    for path in path_list:
        with open(path, 'rb') as f:#input,bug type,params
            result = pickle.load(f)
        if 'similar_functions' not in combine_dict.keys():
            combine_dict['similar_functions']=result['similar_functions']
            continue
        for key in result['similar_functions'].keys():
            if key not in combine_dict['similar_functions'].keys():
                combine_dict['similar_functions'][key]=result['similar_functions'][key]
            else:
                combine_dict['similar_functions'][key]=list(set(combine_dict['similar_functions'][key]).union(set(result['similar_functions'][key])))
    with open(final_combine_path, 'wb') as f:
        pickle.dump(combine_dict, f)

def read_dprel(combine_dict_path,candidate_dir,threshold=0.6):
    save_path=combine_dict_path.replace('.pkl', '+dlrel.pkl')
    dprel_score_list=[]
    # if os.path.exists(save_path):
    #     with open(save_path, 'rb') as f:#input,bug type,params
    #         combine_dict = pickle.load(f)
    #     return combine_dict
    with open(combine_dict_path, 'rb') as f:#input,bug type,params
        combine_dict = pickle.load(f)['similar_functions']
    covered_api=list(combine_dict.keys())
    file_list=os.listdir(candidate_dir)
    for file_name in file_list:
        api_name = file_name.replace(".json", "")
        similar_pairs = []
        with open(os.path.join(candidate_dir, file_name)) as f:
            for line in f.read().split("\n"):
                if len(line):
                    similar_pairs.append(json.loads(line))
        similar_pairs = similar_pairs[:30]
        cover=False
        if api_name in covered_api:
            cover=True
        
        for pair in similar_pairs:
            similar_api_name = pair[0]
            similarity = pair[1]
            if similar_api_name == api_name:
                continue
            if similarity<threshold: # if dlrel's similarity is too low, we pass these functions
                break
            
            
            if not cover :
                combine_dict[api_name]=[]
            if similar_api_name not in combine_dict[api_name]:
                combine_dict[api_name].append(similar_api_name)     
                dprel_score_list.append([(api_name,pair[0]),similarity])           
                
    for key in combine_dict.keys():
        combine_dict[key]=list(set(combine_dict[key]))
    with open(save_path, 'wb') as f:
        pickle.dump(combine_dict, f)
    
    tmp_save_path='./result/total_similar_score.pkl'
    if not os.path.exists(tmp_save_path):
        save_dict={}
    else:
        with open(tmp_save_path, 'rb') as f:#input,bug type,params
            save_dict = pickle.load(f)
    for sc in dprel_score_list:
        key=f'{sc[0][0]}-{sc[0][1]}'
        if key not in save_dict.keys():
            #'intersection' or method=='levenshtein' or method=='local'
            save_dict[key]={}
        save_dict[key]['argument']=sc[1]
    with open(tmp_save_path, 'wb') as f:
        pickle.dump(save_dict, f)
    
    return combine_dict

def match_argument(args_A: list['ArgDef'], args_B: list['ArgDef']):
    """
    map argument definition list A to B
    return a mapping list: [(index_A, index_B)]
    """
    sim = ArgDef.similarity(args_A, args_B)
    # sim_matrix = [[5 - y for y in x] for x in sim]
    sim_matrix = [[5 - y for y in x] for x in sim]
    m = Munkres()
    indices = m.compute(sim_matrix)
    indices.sort(key=lambda x: x[1])
    # print(indices)
    return indices

def verify_similarity(new_combine_dict):
    TorchDatabase.database_config("127.0.0.1", 27017, "torch")
    for key in new_combine_dict.keys():
        try:
            api_A = TorchAPI(key)
        except:
            continue
        new_simi_list=[]
        for simi_api in new_combine_dict[key]:
            try:
                api_B = TorchAPI(simi_api)
            except:# no this api
                new_simi_list.append(simi_api)
                continue
            jump_sign=False
            if len(api_A.arg_defs) == 0 or len(
                    api_B.arg_defs) == 0 or (api_A.is_class != api_B.is_class):
                continue

            indices = match_argument(api_A.arg_defs, api_B.arg_defs)

            # check whether there is any unmatched argument in source API
            source_matched_indices = [p[0] for p in indices]
            if len(source_matched_indices) < len(api_A.arg_defs):
                for i in range(len(api_A.arg_defs)):
                    if i not in source_matched_indices:
                        if api_A.arg_defs[i].is_optional:
                            api_A.arg_defs[i].ignore = True
                        else:
                            jump_sign=True
                            print(api_B.api)
                            break
            if jump_sign: continue
            
            # check whether there is any unmatched argument in target API
            target_matched_indices = [p[1] for p in indices]
            if len(target_matched_indices) < len(api_B.arg_defs):
                for i in range(len(api_B.arg_defs)):
                    if i not in target_matched_indices and not api_B.arg_defs[i].is_optional:
                        jump_sign=True
                        print(api_B.api)
                        break
            if jump_sign: continue
            
            new_simi_list.append(simi_api)
            
        new_combine_dict[key]=new_simi_list
        
    new_combine_dict = {key: value for key, value in new_combine_dict.items() if value != []}
    return new_combine_dict


if __name__=='__main__':
    
    relation_path='./dynamic_profiler_result/relation-combine.pkl'
    save_dir='./result'
    final_combine_path='./result/context_combine.pkl'
    final_save_path='./result/tmp_context_relation.pkl'
    method_list=['intersection','levenshtein']
    value=60
    
    # # 1. use different method to determine the similarity pairs
    for method in method_list:
        #method='local'#'local'#'intersection'#'levenshtein'
        get_raw_similarity(relation_path,save_dir,final_save_path,method,threshold=value)

    # # # 2. combine relations
    path_list=[]
    for method in method_list:
        path_list.append("./result/tmp_context_relation_{}_{}.pkl".format(method,value))
    combine_similairy(path_list,final_combine_path)
    
    # # 3. append deerel results and use argument similarity to remove some improper results
    candidate_dir='./dprl/torch-candidate'
    new_combine_dict=read_dprel(final_combine_path,candidate_dir)# load similarity relation from deep_rel
    print(1)




    print('finish')