import os
import pickle
import re
import itertools
import string
import sys
sys.path.append('.')
import simi_utils as su
import argu_utils as au
import pymongo
from tqdm import trange
import copy
import csv
import numpy as np



function_list=['void', 'int', 'double', 'static']

def detect_start_blank(string1):
    blank_num=0
    while string1[blank_num]==' ':
        blank_num+=1
    return blank_num
    


def save_function(filename,program_list):
    file = open(filename, 'r',encoding='gb18030', errors='ignore')
    text_list=file.readlines()
    tmp_list=[]
    program_in_file=[]
    for l in range(len(text_list)):
        blank_num=detect_start_blank(text_list[l])

        if blank_num!=0:
            text_list[l]=text_list[l][blank_num:]

            if tmp_list==[]:

                if text_list[l-1].startswith('#'):
                    tmp_list.append(text_list[l-2])
                tmp_list.append(text_list[l-1])
            tmp_list.append(text_list[l])
        if blank_num==0 and tmp_list!=[]:
            if "}" not in text_list[l]:
                tmp_list.append(text_list[l])
                continue
            tmp_list.append(text_list[l])

            tmp_function=''.join(tmp_list)

            program_in_file.append(tmp_function)
            tmp_list=[]
    if program_in_file!=[]:
        program_list.append((filename.replace('/home/zxy/main/DL_compiler_test/1_test_code/clone_detection/pytorch/', '/pytorch/'),program_in_file))
    return program_list

def file_name(file_dir,program_list,file_type=['cpp']):
    for root,dirs,files in os.walk(file_dir):
        for eachfile in files:
            if eachfile.split('.')[-1] in file_type:#):#
                filename=root+'/'+eachfile
                program_list=save_function(filename,program_list)
    return program_list


def extract_msg(program_list):
    program_dict={}
    for pl in program_list:
        key=pl[0].replace('/pytorch/aten','').split('.')[0]
        function_list=pl[1]
        # program_dict[key]={}
        
        for i in range(len(function_list)):
            tmp_dict,name=extract_function(function_list[i])
            if name!=None:
                count=0
                
                while key+'-'+name in program_dict.keys():
                    if count==0:
                        name=name+str(count)
                    elif count<=10:
                        name=name[:-1]+str(count)
                    else:
                        name=name[:-2]+str(count)
                    count+=1
                program_dict[key+'-'+name]=tmp_dict
    return program_dict

def extract_function(function_string):
    tmp_dict={}
    tmp_dict['input']=[]
    tmp_dict['output']=[]
    tmp_dict['call']=[]
    function_string=function_string.replace('\n',' ')
    function_string=function_string.replace('\t',' ')
    function_string=' '+function_string
    # 'void check_supported_cuda_type(cudaDataType cuda_type) {\n\tif (cuda_type == CUDA_R_16F) {\n\t\tcudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();\n\t\tTORCH_CHECK(\n\t\t\t\tprop->major >= 5 && ((10 * prop->major + prop->minor) >= 53),\n\t\t\t\t"Sparse operations with CUDA tensors of Float16 type are not supported on GPUs with compute capability < 5.3 (current: ",\n\t\t\t\tprop->major,\n\t\t\t\t".",\n\t\t\t\tprop->minor,\n\t\t\t\t")");\n\t}\n#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000\n\tif (cuda_type == CUDA_R_16BF) {\n\t\tcudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();\n\t\tTORCH_CHECK(\n\t\t\t\tprop->major >= 8,\n\t\t\t\t"Sparse operations with CUDA tensors of BFloat16 type are not supported on GPUs with compute capability < 8.0 (current: ",\n\t\t\t\tprop->major,\n\t\t\t\t".",\n\t\t\t\tprop->minor,\n\t\t\t\t")");\n\t}\n#endif\n}\n'
    orig_result_list=re.findall(r'[ ]+[\w:\.]+\(', ' '+function_string)#[ \\]+[\w:\.]+\(
    if orig_result_list==[]:
        return tmp_dict,None
    orig_result_list=clean_list(orig_result_list)
    # print(orig_result_list)
    result_list=list(set(orig_result_list)) # remove duplication
    result_list.sort(key=orig_result_list.index)
    # print(result_list)
    # function_message_list=result_list[0].split('(')
    function_message_list=function_string.split('{')[0].split('(')
    name=result_list[0].split('(')[0]
    # only record output type
    tmp_dict['output'].append(function_string.split(' '+name)[0].split(' ')[-1])
    function_list=result_list[1:]
    for fl in function_list:
        tmp_dict['call'].append(fl.split('(')[0])
    
    try:
        input_msg=function_message_list[1].replace(')','').split(',')
    except Exception as e:
        print(e)
        input_msg=[]
    tmp_dict['input']+=input_msg

    tmp_dict['input']=clean_list(tmp_dict['input'],back=True)# remove extra space in the begin of string
    tmp_dict['argument']=[]# The format of this list is (type,name)
    for i in range(len(tmp_dict['input'])):
        argument_msg=tmp_dict['input'][i].split(' ')
        if len(argument_msg)<2: # include the situtation that input=''
            tmp_arg=au.Arguments(tmp_dict['input'][i],None)
            tmp_dict['argument'].append(tmp_arg)
        else:
            tmp_arg=au.Arguments(argument_msg[-1],tmp_dict['input'][i].replace(' {}'.format(argument_msg[-1]),''))
            tmp_dict['argument'].append(tmp_arg)
            
    tmp_dict['output']=clean_list(tmp_dict['output'],back=True)
    tmp_dict['call']=clean_list(tmp_dict['call'],back=True)
    # print(1)
    return tmp_dict,name
  
def clean_list(result_list,back=False):
    new_list=[]
    for rl in result_list:
        try:
            while r" " == rl[0]:
                rl=rl[1:]
            if back:
                while r" " == rl[-1]:
                    rl=rl[:-1]
        except:
            pass
        new_list.append(rl)
    return new_list

def extract_similar_get_all(program_dict):
    def generate_tuples(lst):
        for i in trange(len(lst)):
            for j in range(i+1, len(lst)):
                yield (lst[i], lst[j])

    pair_list=[]# each sublist contains [name1, name2, path1, path2]
    function_name_list=list(program_dict.keys())
    function_name_list=clean_function(function_name_list)
    all_list=np.ones((len(function_name_list),len(function_name_list)))
    with open('./tmp/c++_index-2024.pkl', 'wb') as f:
        pickle.dump(function_name_list, f)
    for tup in generate_tuples(function_name_list):
        call_score,call_subset=similar_score(program_dict[tup[0]],program_dict[tup[1]],'call')
        input_score,input_subset=similar_score(program_dict[tup[0]],program_dict[tup[1]],'input')
        input_score=round(input_score,3)
        call_score=round(call_score,3)
        tup=[function_name_list.index(i) for i in tup]
        all_list[tup[0],tup[1]]=input_score+call_score
        all_list[tup[1],tup[0]]=input_score+call_score
        # if (input_score>0.5) or (call_score>0.5 and call_score<1.0001):
        if (input_score>0.8) and (call_score>0.8 and call_score<1.0001):
            pair_list.append([tup,[input_score,call_score]])
        
            if len(pair_list)%100==0:
                with open('./tmp/c++_new_pair-2024.pkl', 'wb') as f:
                    pickle.dump(pair_list, f)
    with open('./tmp/c++_new_pair-2024.pkl', 'wb') as f:
        pickle.dump(pair_list, f)
    return pair_list

    
def similar_score(dict_1, dict_2, key):
    signal=False
    len_union=len(set(dict_1[key]).union(set(dict_2[key])))
    if len_union==0:
        return 1.1,signal
    len_intersection=len(set(dict_1[key]).intersection(set(dict_2[key])))
    if len_intersection==len(dict_1[key]) or len_intersection==len(dict_2[key]):
        signal=True
    result=len_intersection/len_union
    return result,signal

def clean_function(function_list):
    new_list=[]
    remove_list=['TORCH_META_FUNC']
    for fn in function_list:
        remove=False
        for rem in remove_list:
            if rem in fn:
                remove=True
                break
        if not remove:
            new_list.append(fn)
    return new_list

def get_index(lst=None, item=''):
    return [index for (index,value) in enumerate(lst) if value == item]

def process_similar(pair_list):
    new_list=[]
    path_list=[]
    for pl in pair_list:
        base_path=pl[1][0]
        similar_function_list=pl[0]
        if base_path not in path_list:
            path_list.append(base_path)
            new_list.append(similar_function_list)
        else:
            index_list=get_index(path_list,base_path)
            # pre_similar_function=new_list[index]
            sign=False
            for index in index_list:
                if (similar_function_list[0] not in new_list[index]) and (similar_function_list[1] not in new_list[index]):
                    # path_list.append(base_path)
                    # new_list.append(similar_function_list)
                    # #TODO:
                    pass
                    # simi=[''.join([i for i in similar_function_list[0] if not i.isdigit()]),
                    # ''.join([i for i in similar_function_list[1] if not i.isdigit()])]
                    # sign=check_simi(simi,new_list[index])
                else:
                    sign=True
                    break
            if sign:
                for sf in similar_function_list:
                    if sf not in new_list[index]:
                        new_list[index].append(sf)
            else:
                path_list.append(base_path)
                new_list.append(similar_function_list)

    maxlen=0
    for nl in new_list:
        maxlen=max(maxlen,len(nl))
    
    return new_list,path_list,maxlen

def process_new_similar(pair_list):
    new_pair_list=[]
    for pl in pair_list:
        if (pl[1][0]>0.8) or (pl[1][-1]>0.8):
            new_pair_list.append(pl)
    
    
    new_list=[]
    for pl in new_pair_list:
        similar_function_list=pl[0]
        if new_list==[]:
            new_list.append(list(similar_function_list))
            continue
        sign=False
        for index in range(len(new_list)):
            if (similar_function_list[0] not in new_list[index]) and (similar_function_list[1] not in new_list[index]):
                pass
            else:
                sign=True
                break
        if sign:
            for sf in similar_function_list:
                if sf not in new_list[index]:
                    new_list[index].append(sf)
        else:
            new_list.append(list(similar_function_list))

        # 一个不在则补，两个不在则新建
    maxlen=0
    for nl in new_list:
        maxlen=max(maxlen,len(nl))
    
    return new_list,maxlen

def save_csv(similar_pair,maxlenth,save_path):
    with open(save_path,"w+") as csvfile: 
        writer = csv.writer(csvfile)
        title_list=['similar_function','path']
        writer.writerow(title_list)

    for i in range(len(similar_pair)):
        result_list=[]
        result_list+=similar_pair[i]
        while len(result_list)<maxlenth:
            result_list.append('/')

        with open(save_path,"a+") as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerow(result_list)

if __name__=='__main__':
    program_list=[]
    save_path='./tmp_data.pkl'
    program_path='./program_msg.pkl'
    save_path_1='./tmp/c++_new_pair-2024.pkl'
    csv_path='./new_similar_pair.csv'
    simi_path='./new_similar_pair.pkl'
    index_path='./tmp/c++_index-2024.pkl'
    # # 1. extract cpp functions  #/home/zxy/main/DL_compiler_test/1_test_code/clone_detection/pytorch/aten
    # program_list=file_name('./pytorch/aten', program_list)
    # # use 'pytorch/aten' in the pytorch repositorch as the path.
    # with open(save_path, 'wb') as f:
    #     pickle.dump(program_list, f)
    # print(1)
    with open(save_path, 'rb') as f:#input,bug type,params
        program_list = pickle.load(f) 
    
    # # 2. extract input output calls
    # program_dict=extract_msg(program_list)
    # with open(program_path, 'wb') as f:
    #     pickle.dump(program_dict, f)
    
    # 3. find similarity pairs
    with open(program_path, 'rb') as f:#input,bug type,params
        program_dict = pickle.load(f) 
    similar_pair=extract_similar_get_all(program_dict)
    # WARNING: this function may take 2 hours to extract and evaluate the static similairy among over 7,700 functions
    # You can just load the preserved similarity

    # validate_function(program_list)
    with open(save_path_1, 'rb') as f:
        similar_pair = pickle.load(f)
    # Comments: Each pair is like [[a,b],[value1,value2]], a and b are the indexes of functions that you can use index list to find its name;
    # and value1 and value2 are the input and output arguments similarity and called functions similarity.
    with open(index_path, 'rb') as f:#input,bug type,params
        index_list = pickle.load(f)
    sorted_similar_pair,maxlen=process_new_similar(similar_pair)# get the similar functions

    # use function names to replace indexes
    new_sorted_similar_pair=[]
    for ssp in sorted_similar_pair:
        new_sorted_similar_pair.append([index_list[s] for s in ssp])
    
    # save and write results
    with open(simi_path, 'wb') as f:
        pickle.dump(new_sorted_similar_pair, f)
    save_csv(new_sorted_similar_pair,maxlen,csv_path)