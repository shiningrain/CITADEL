import os
import csv
import pickle
import copy
import numpy as np
import json
import sys
sys.path.append('./dprl')
from classes.tf_api import TFAPI
from classes.argdef import ArgDef
from classes.library_def import tf_lib_def
from classes.database import TFDatabase
from munkres import Munkres
from tqdm import trange


def write_csv(relation_dict,csv_path):
    relation_table=relation_dict['table']
    relation_vocab=relation_dict['vocabulary']
    for i in range(relation_table.shape[0]):
        new_list=list(set(relation_table.loc[i,'api']))
        new_list=sorted(set(relation_table.loc[i,'api']),key=relation_table.loc[i,'api'].index)
        relation_table.loc[i,'api']=new_list
    relation_table.to_csv(csv_path)

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

def get_related_api_dict(relation_list):
    r_length=len(relation_list)
    # tmp_list=[]
    result_dict={}
    result_dict['similar_functions']={}
    relation_dict=result_dict['similar_functions']
    for i in range(r_length):
        name1=relation_list[i][0][0]
        name2=relation_list[i][0][1]
        # tmp_list.append([name1,name2])
        
        for n in range(len(relation_list[i][0])):
            if relation_list[i][0][n] not in relation_dict.keys():
                relation_dict[relation_list[i][0][n]]=[]
            relation_dict[relation_list[i][0][n]].append(relation_list[i][0][n-1])
    return relation_dict

def load_similarity_score(similarity_path,threshold):
    with open(similarity_path, 'rb') as f:#input,bug type,params
        score_dict = pickle.load(f)
    similar_score_list=[]

    path_list=[f'./result/rq3_score_sys_{threshold}.pkl',
    f'./result/rq3_pairs_sys_{threshold}.pkl',
    './result/rq3_score_deeprel.pkl',
    './result/rq3_pairs_deeprel.pkl']
    if os.path.exists(path_list[0]):
        return path_list

    if os.path.exists('./result/rq3_score_deeprel.pkl'):
        with open('./result/rq3_score_deeprel.pkl', 'rb') as f:#input,bug type,params
            dprel_list = pickle.load(f)
        record=False
    else:
        dprel_list=[]
        record=True
    score_dict_keys=list(score_dict.keys())
    for k in trange(len(score_dict_keys)):
        key=score_dict_keys[k]
        if 'intersection' in score_dict[key].keys():
            context_score=max(score_dict[key]['intersection'],score_dict[key]['levenshtein'])
            if context_score>threshold:
                tmp=key.split('-')
                tmp.sort()
                if tmp not in similar_score_list:
                    similar_score_list.append([tmp,context_score])
        if record and 'argument' in score_dict[key].keys() and score_dict[key]['argument']>0.8:
            tmp=key.split('-')
            tmp.sort()
            if tmp not in dprel_list:
                dprel_list.append([tmp,score_dict[key]['argument']])

    context_pairs=get_related_api_dict(similar_score_list)
    if record:#TODO:
        deeprel_pairs=get_related_api_dict(dprel_list)
    
    # # save
    with open(path_list[0], 'wb') as f:
        pickle.dump(similar_score_list, f)
    with open(path_list[1], 'wb') as f:
        pickle.dump(context_pairs, f)

    if record:
        with open(path_list[2], 'wb') as f:
            pickle.dump(dprel_list, f)
        with open(path_list[3], 'wb') as f:
            pickle.dump(deeprel_pairs, f)
    return path_list

def extract2csv(function_relation_dict,csv_path,threshold=60,method=None):
    score_list=function_relation_dict['similarity score']
    
    update_similarity_score(score_list,save_path=os.path.join(os.path.dirname(csv_path),'total_similar_score.pkl'),method=method)
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
            
    # import matplotlib.pyplot as plt 
    # plt.boxplot(avg_score_list)
    # plt.savefig('./tmp1.pdf')
            
    with open(csv_path.replace('.csv','_{}_{}.csv'.format(method,threshold)),"w") as csvfile: 
        writer = csv.writer(csvfile)

        #先写入columns_name
        writer.writerow(["function pairs","score"])
        #写入多行用writerows
        writer.writerows(output_list)
    return output_list

def extract_similarity(save_dir,threshold=60,method=None):
    # score_list=function_relation_dict['similarity score']
    
    # update_similarity_score(score_list,save_path=os.path.join(os.path.dirname(csv_path),'total_similar_score.pkl'),method=method)
    path_list=load_similarity_score(os.path.join(save_dir,'total_similar_score.pkl'),threshold)
    return path_list


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
                    
    # # use 30 as threshold in 0404version
    # # from the boxplot, we use 50 as threshold to determine the operation that are mostly used.
    # length_list=[]
    # for i in range(length):
    #     if len(tmp_table.loc[i,'api'])!=0:
    #         length_list.append(len(tmp_table.loc[i,'api']))
    # import matplotlib.pyplot as plt 
    # plt.boxplot(length_list)
    # plt.ylim(ymax=100)
    # plt.savefig('./tmp-0404.pdf')
    return relation_dict

def get_raw_similarity(relation_path,csv_path,final_save_path,method,threshold):
    with open(relation_path, 'rb') as f:#input,bug type,params
        relation_dict = pickle.load(f)
    if method=='intersection' or method=='levenshtein':# or method=='local'
        relation_dict=remove_useless_operation(relation_dict)
    # write_csv(relation_dict,csv_path)
    print(f'=====Computing Similarity Score with {method} Method=====')
    function_relation_path=os.path.join(os.path.dirname(csv_path),'finegrained_relation-{}-0404.pkl'.format(method))
    function_relation_dict=analyze_simi(relation_dict,function_relation_path,method=method)
    
    with open(function_relation_path, 'rb') as f:#input,bug type,params
        function_relation_dict = pickle.load(f)
    print(f'=====Analyzing Similarity=====')
    # threshold=50#60
    if method=='local':
        threshold=80
    function_relation_list=extract2csv(function_relation_dict,csv_path,threshold=threshold,method=method)
    print(f'=====Updating Raw Similarity Relationship=====')
    relation_dict=cluster_relation_pairs(function_relation_list,final_save_path,value)

def new_get_raw_similarity(relation_path,save_dir,final_save_path,method,threshold):
    path_list=extract_similarity(save_dir,threshold=threshold)
    return path_list

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
    combine_dict={}
    covered_api=list(combine_dict.keys())
    file_list=os.listdir(candidate_dir)
    for file_name in file_list:
        api_name = file_name.replace(".json", "")
        similar_pairs = []
        with open(os.path.join(candidate_dir, file_name)) as f:
            for line in f.read().split("\n"):
                if len(line):
                    similar_pairs.append(json.loads(line))
        similar_pairs = similar_pairs[:20]
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
            dprel_score_list.append([(api_name,pair[0]),similarity])     
            
            if not cover :
                combine_dict[api_name]=[]
            if similar_api_name not in combine_dict[api_name]:
                combine_dict[api_name].append(similar_api_name)     
                # dprel_score_list.append([(api_name,pair[0]),similarity])
                
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
    TFDatabase.database_config("127.0.0.1", 27017, "tf")
    
    tf_lib_def.load_apis(lazy=True)
    for key in new_combine_dict.keys():
        try:
            api_A = TFAPI(key)
        except:
            continue
        new_simi_list=[]
        for simi_api in new_combine_dict[key]:
            try:
                api_B = TFAPI(simi_api)
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
                            # print(api_B.api)
                            break
            if jump_sign: continue
            
            # check whether there is any unmatched argument in target API
            target_matched_indices = [p[1] for p in indices]
            if len(target_matched_indices) < len(api_B.arg_defs):
                for i in range(len(api_B.arg_defs)):
                    if i not in target_matched_indices and not api_B.arg_defs[i].is_optional:
                        jump_sign=True
                        # print(api_B.api)
                        break
            if jump_sign: continue
            
            new_simi_list.append(simi_api)
            
        new_combine_dict[key]=new_simi_list
        
    new_combine_dict = {key: value for key, value in new_combine_dict.items() if value != []}
    return new_combine_dict

def verify_score(new_combine_dict,score_path,method,threshold):
    with open(score_path, 'rb') as f:#input,bug type,params
        score_list = pickle.load(f)
    score_dict={}
    for i in score_list:
        score_dict[f'{i[0][0]}-{i[0][1]}']=i[1]
    # rq3_final_path=os.path.join(os.path.dirname(score_path),f'rq3_final-{threshold}.pkl')
    # if os.path.exists(rq3_final_path):
    #     with open(rq3_final_path, 'rb') as f:#input,bug type,params
    #         final_dict = pickle.load(f)
    # else:
    #     final_dict={}
    #     final_dict['score']={}
    #     final_dict['context_matched']={}
    #     final_dict['argument_matched']={}
    tmp_combine_dict={}
    tmp_score_dict={}
    # if method=='context':
    #     tmp_combine_dict=final_dict['context_matched']
    # elif method =='argument':
    #     tmp_combine_dict=final_dict['argument_matched']
    #     tmp_score_list={}
    for key in new_combine_dict.keys():
        new_matched_list=[]
        for matched in new_combine_dict[key]:
            tmp=[key,matched]
            tmp.sort()
            new_key=f'{tmp[0]}-{tmp[1]}'
            if new_key in score_dict.keys():
                new_matched_list.append(matched)
                # if new_key not in final_dict['score'].keys():
                #     final_dict['score'][new_key]={}
                # final_dict['score'][new_key][method]=score_dict[new_key]
                # if method=='argument':
                tmp_score_dict[new_key]=score_dict[new_key]
        tmp_combine_dict[key]=list(set(new_matched_list))
    # with open(rq3_final_path, 'wb') as f:
    #     pickle.dump(final_dict, f)
    # if method=='argument':
    tmp_list=list(tmp_combine_dict.keys())
    for key in tmp_list:
        for api in tmp_combine_dict[key]:
            if api not in tmp_combine_dict.keys():
                tmp_combine_dict[api]=[]
            if key not in tmp_combine_dict[api]:
                tmp_combine_dict[api].append(key)

    return [tmp_combine_dict,tmp_score_dict]
    # return final_dict

def new_verify_similarity(new_combine_dict,score_path,method,threshold):
    TFDatabase.database_config("127.0.0.1", 27017, "tf")
    
    tf_lib_def.load_apis(lazy=True)
    for key in new_combine_dict.keys():
        try:
            api_A = TFAPI(key)
        except:
            continue
        new_simi_list=[]
        for simi_api in new_combine_dict[key]:
            try:
                api_B = TFAPI(simi_api)
            except:# no this api
                new_simi_list.append(simi_api)
                continue
            jump_sign=False
            if len(api_A.arg_defs) == 0 or len(
                    api_B.arg_defs) == 0 or (api_A.is_class != api_B.is_class):
                continue

            
            api_A_argument_set=set([i.name for i in api_A.arg_defs])
            api_B_argument_set=set([i.name for i in api_B.arg_defs])
            if len(api_A_argument_set)>1 and len(api_B_argument_set)>1:
                if (api_B_argument_set.issubset(api_A_argument_set) or api_A_argument_set.issubset(api_B_argument_set)):
                    new_simi_list.append(simi_api)
                    print(f'{api_A.api} and {api_B.api} have subset relationship!!')
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
                            # print(api_B.api)
                            break
            if jump_sign: continue
            
            # check whether there is any unmatched argument in target API
            target_matched_indices = [p[1] for p in indices]
            if len(target_matched_indices) < len(api_B.arg_defs):
                for i in range(len(api_B.arg_defs)):
                    if i not in target_matched_indices and not api_B.arg_defs[i].is_optional:
                        jump_sign=True
                        # print(api_B.api)
                        break
            if jump_sign: continue
            
            new_simi_list.append(simi_api)
            
        new_combine_dict[key]=new_simi_list
        
    new_combine_dict = {key: value for key, value in new_combine_dict.items() if value != []}

    new_combine_dict=verify_score(new_combine_dict,score_path,method,threshold=threshold)

    return new_combine_dict

def verify(threshold, path_list):
    sys_pair_path=path_list[1]
    sys_score_path=path_list[0]
    deeprel_score_path=path_list[2]
    deeprel_pair_path=path_list[3]
    deeprel_verify_path=path_list[3].replace('rq3_pairs','rq3_verify')
    sys_verify_path=path_list[1].replace('rq3_pairs','rq3_verify')
    
    with open(sys_pair_path, 'rb') as f:#input,bug type,params
        sys_pair = pickle.load(f)
    with open(deeprel_pair_path, 'rb') as f:#input,bug type,params
        deeprel_pair = pickle.load(f)
    deeprel_pair={pair:deeprel_pair[pair] for pair in deeprel_pair.keys() if deeprel_pair[pair]!=[]}
    sys_pair={pair:sys_pair[pair] for pair in sys_pair.keys() if sys_pair[pair]!=[]}
    if os.path.exists(deeprel_verify_path):
        dprel_v=False
    else:
        dprel_v=True
    
    if os.path.exists(sys_verify_path):
        sys_v=False
    else:
        sys_v=True

    if sys_v:
        sys_verify=new_verify_similarity(sys_pair,sys_score_path,method='context',threshold=threshold)
        with open(sys_verify_path, 'wb') as f:
            pickle.dump(sys_verify, f)
    else:
        with open(sys_verify_path, 'rb') as f:#input,bug type,params
            sys_verify = pickle.load(f)
    
    if dprel_v:
        deeprel_verify=new_verify_similarity(deeprel_pair,deeprel_score_path,method='argument',threshold=threshold)
        with open(deeprel_verify_path, 'wb') as f:
            pickle.dump(deeprel_verify, f)
    else:
        with open(deeprel_verify_path, 'rb') as f:#input,bug type,params
            deeprel_verify = pickle.load(f)
        # new_combine_dict['argument_matched']=deeprel_verify[-1]

    new_combine_dict=combine_verify(sys_verify,deeprel_verify,threshold)
    return new_combine_dict

def combine_verify(sys_verify,deeprel_verify,threshold):
    final_combine_dict={}
    final_combine_dict['context_pair']=sys_verify[0]
    final_combine_dict['context_score']=sys_verify[1]
    final_combine_dict['argument_score']=deeprel_verify[1]
    final_combine_dict['argument_pair']=deeprel_verify[0]
    tmp_pair={}
    tmp_score={}
    pair_keys=list(set(deeprel_verify[0].keys()).union(set(sys_verify[0])))
    for key in pair_keys:
        tmp_pair[key]=[]
        if key in deeprel_verify[0].keys():
            tmp_pair[key]+=deeprel_verify[0][key]
        if key in sys_verify[0].keys():
            tmp_pair[key]+=sys_verify[0][key]
        tmp_pair[key]=list(set(tmp_pair[key]))
        for matched in tmp_pair[key]:
            tmp=[key,matched]
            tmp.sort()
            name=f"{tmp[0]}-{tmp[1]}"
            tmp_score[name]={}
            if name in sys_verify[1].keys():
                tmp_score[name]['context']=sys_verify[1][name]
            if name in deeprel_verify[1].keys():
                tmp_score[name]['argument']=deeprel_verify[1][name]
            if tmp_score[name]=={}:
                print(1)
    final_combine_dict['total_pair']=tmp_pair
    final_combine_dict['total_score']=tmp_score
    return final_combine_dict

if __name__=='__main__':
    # Finish: formalize the function name in /home/zxy/main/DL_compiler_test/1_test_code/matching_similar_relation/relations/result_0105
    relation_path='./result/relation-combine.pkl'
    save_dir='./result/'
    final_save_path='./result/final_relation_0404.pkl'
    method_list=['intersection','levenshtein']# ,'local' TODO: not using local method in 0404 version. because too many FP cases
    value = 80

    path_list=new_get_raw_similarity(relation_path,save_dir,final_save_path,method,threshold=value)

    new_combine_dict=verify(value,path_list)# use arugment to verify the similarity functions
    
    with open(f'./result/final_relation_0404_{value}.pkl', 'wb') as f:
        pickle.dump(new_combine_dict, f)


    print('finish')