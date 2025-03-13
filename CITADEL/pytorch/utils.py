import os
import pickle
import numpy as np
# from pydantic import EnumError
from torch import nn
import torch.autograd.profiler as profiler
from functools import wraps
import time
import inspect
import re
import copy

from configparser import ConfigParser

def read_config(path='./CITADEL.conf'):
    config = ConfigParser()
    config.read(path, encoding='UTF-8')
    return config

def decorator_performance(func,save=True,save_dir='./result'):
    @wraps(func)
    def profile_function(*args, **kwargs):# args[0]=input
        name=args[-1]
        args=args[:-1]
        record_file=os.path.join(save_dir,'record.pkl')

        with open(record_file, 'rb') as f:#input,bug type,params
            record = pickle.load(f)
        
        
        with profiler.profile(with_stack=True, profile_memory=True,record_shapes=True) as prof:# TODO: 1.7 ver has no ",with_modules=True"
            if record['method']=='perf':# record time and call stack and return
                time0=time.time()
                func(*args,**kwargs)
                time_cost=time.time()-time0 # could use prof.self_cpu_time_total to calculate time cost
            elif record['method']=='stat':
                try:
                    func(*args,**kwargs)
                    status='Pass' # pass
                except Exception as e:
                    status=type(e).__name__
                    print(e)
            elif record['method']=='value':
                output_value=func(*args,**kwargs)
            else:
                print('not implement')
                os._exit(0)
        profiler_dir=os.path.join(save_dir,'profiler',record['method'],record['code_name'])
        if not os.path.exists(profiler_dir):
            os.makedirs(profiler_dir)
        pickle_path=os.path.join(profiler_dir,'','{}-{}-{}.pkl'.format(record['iter'],record['code'],name))
        if save:
            with open(pickle_path, 'wb') as f:
                pickle.dump(prof.function_events, f)
        
        if record['method']=='perf':
            return prof, record['code'], time_cost# prof.self_cpu_time_total
        elif record['method']=='stat':
            return prof, record['code'], status
        elif record['method']=='value':
            return prof, record['code'], output_value
        else:
            print('not implement')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return profile_function

def traversalDir_FirstFile(path):
    tmplist = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file1 in files:
            m = os.path.join(path,file1)
            if (os.path.isfile(m)):
                tmplist.append(m)
                # tmplist1.append(file1)
    return tmplist


def traversalDir_FirstDir(path):
    tmplist = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file1 in files:
            m = os.path.join(path,file1)
            if (os.path.isdir(m)):
                tmplist.append(m)
                # tmplist1.append(file1)
    return tmplist

@decorator_performance
def test_record(func,*args):
    result=func(*args)
    return result


def save_result(result,simi_func_list,func_cls_list,save_dir='./result'):
    save_dict={}
    save_dict['data']=result[0]
    save_dict['score']=result[1]
    save_dict['api_list']=[func_cls_list[0].__name__]
    for simi in simi_func_list:
        save_dict['api_list'].append(simi[0].__name__)
    save_path=os.path.join(save_dir,'-'.join(save_dict['api_list'])+'.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)

def add_import_info(path_list):
    processed_path=[]
    for key in path_list.keys():
        file_path=path_list[key][0]
        if file_path in processed_path:
            continue
        f = open(file_path, mode="r")
        lines=f.readlines()
        f.close()
        lines.insert(0,"import sys\nsys.path.append('.')\nfrom utils import test_record\n")
        fo = open(file_path, "w")
        fo.writelines( lines )
        fo.close()
        processed_path.append(file_path)

def code_process(func_cls,simi_func_list,origin_code_path,method='preprocess'):
    # 1st step: get argument
    try:
        origin_parameter=dict(inspect.signature(func_cls).parameters)
        origin_parameter_dict={}
        origin_constraint_dict={}
        for name,param in origin_parameter.items():
            # print(name,parm.default)
            # if param.default==inspect.Parameter.empty
            origin_parameter_dict[name]=param.default
            origin_constraint_dict[name]=str(param.annotation)
        origin_parameter_name_list=list(origin_parameter_dict.keys())
    except Exception as e:
        # if no signature, just return origin path
        print('No signature! Stop processing!')
        new_path_list={}
        for simi_func in simi_func_list:
            new_path_list[simi_func.__name__]=[origin_code_path,0]
        return new_path_list
    
    
    
    all_simi_parameter_dict={}
    all_simi_constraint_dict={}
    remove_list=[]
    for simi_func in simi_func_list:
        try: 
            simi_parameter=dict(inspect.signature(simi_func).parameters)
            all_simi_parameter_dict[simi_func.__name__]={}
            all_simi_constraint_dict[simi_func.__name__]={}
            single_simi_parameter_dict=all_simi_parameter_dict[simi_func.__name__]
            single_simi_constraint_dict=all_simi_constraint_dict[simi_func.__name__]
            for name,param in simi_parameter.items():
                # print(name,parm.default)
                # if param.default==inspect.Parameter.empty
                single_simi_parameter_dict[name]=param.default
                single_simi_constraint_dict[name]=str(param.annotation)
        except:
            remove_list.append(simi_func)
    if remove_list!=[]:
        for rm in remove_list:
            simi_func_list.remove(rm)
    # 2nd step: modify files
    new_path_list={}
    for key in all_simi_parameter_dict.keys():
        print(key)
        single_simi_parameter_dict=all_simi_parameter_dict[key]
        single_simi_constraint_dict=all_simi_constraint_dict[key]
        simi_parameter_name_list=list(single_simi_parameter_dict.keys())
        
        arg_status=check_sub_list(origin_parameter_name_list,simi_parameter_name_list)# strict check whether the origin params are a subset of simi params
        diff_constraints=check_constraints(origin_constraint_dict,single_simi_constraint_dict)
        if not arg_status or diff_constraints!={}:
            # if dimension=2, it means this similar function need to expand 2 dimension
            # default dimension is 0
            new_path,dimension=generate_new_code(origin_parameter_dict,single_simi_parameter_dict,diff_constraints,origin_code_path,key)
            if new_path==False:
                if method=='run':
                    new_path_list[key]=[origin_code_path,dimension]
                continue # fail to generate new code, some important arguments are not covered
            if new_path!=None:
                new_path_list[key]=[new_path,dimension]# key=simi_func.__name__
            else:
                new_path_list[key]=[origin_code_path,dimension]
        else:
            new_path_list[key]=[origin_code_path,0]

    if method=='preprocess':
        add_import_info(new_path_list)
    elif method=='run':
        pass
    
    return new_path_list


def generate_new_code(origin_parameter_dict,simi_parameter_dict,diff_constraints,origin_code_path,simi_func_name,func_name='func_cls'):
    diff_constraints_list=[]
    dimension=0
    if diff_constraints!={}:
        diff_constraints_list=list(diff_constraints.keys())
        for key in diff_constraints_list:
            pattern=r"(Tuple\[)(.)*(\]){1}"
            try:
                origin_constraint=re.search(pattern, diff_constraints[key][0]).group().replace('Tuple[','').replace(']','')
                simi_constraint=re.search(pattern, diff_constraints[key][1]).group().replace('Tuple[','').replace(']','')
            except:
                print("{}-{}".format(diff_constraints[key][0],diff_constraints[key][1]))
                continue
            origin_length=len(origin_constraint.split(','))
            diff_length=len(simi_constraint.split(','))-origin_length
            if diff_length!=dimension:
                dimension=diff_length
           
    # 1st step: check whether need to replace
    f = open(origin_code_path, mode="r")
    origin_codes_list=f.readlines()
    # argument_list=[]
    output_codes_list=[]
    write_new_file=False
    for codes in origin_codes_list:
        tmp_key=func_name+'('
        if tmp_key not in codes:
            output_codes_list.append(codes)
            continue
        
        origin_codes=copy.deepcopy(codes)
        tmp_simi_parameter=simi_parameter_dict#copy.deepcopy(simi_parameter_dict)
        
        codes=codes.strip('\n')
        cuda_part=None
        # if '.cuda' in codes:# handle the codes with '.cuda '
        #     cuda_part=codes.replace(codes.split('.cuda')[0],'')
        #     codes=codes.split('.cuda')[0]
        # if '.to(' in codes:
        #     to_part=
        if ').' in codes:# handle the codes with '.cuda '
            cuda_part=codes.replace(codes.split(').')[0],'')[1:]# not contain ")"
            codes=codes.split(').')[0]+')'
        
        codes_split_list=codes.split(tmp_key)
        if len(codes_split_list)!=2:
            print('error code split, exit!!!!!')
            print(codes)
            os._exit(0)
        new_codes=codes_split_list[-1]
        # final_part=""
        # for i in range(len(new_codes)):# delete the last ")"
        #     if new_codes[-1-i]==")":
        #         final_part=new_codes[-1-i:]
        #         new_codes=new_codes[:-1-i]
        #         break
        tmp_count=1
        # for i in range(len(new_codes)):
        #     if tmp_count==0:
        #         final_part=new_codes[i-1:]
        #         new_codes=new_codes[:i-1]
        #         break
        #     if new_codes[i]=='(':
        #         tmp_count+=1
        #     if new_codes[i]==')':
        #         tmp_count-=1
        for i in range(len(new_codes)+1):
            if tmp_count==0:
                final_part=new_codes[i-1:]
                new_codes=new_codes[:i-1]
                break
            if new_codes[i]=='(':
                tmp_count+=1
            if new_codes[i]==')':
                tmp_count-=1           
        
        new_codes_list = re.split(r",(?![^(]*\))", new_codes)
        new_codes_list=[i for i in new_codes_list if i!='']# remove '' elements in the list
        
        try:
            default_argument_dict,modify_argument_list=extract_argument_dict(origin_parameter_dict,new_codes_list)
        except:# fail to extrat arguments
            return origin_code_path,0
        
        if modify_argument_list==[]:
            print(0)
        
        # 1. 对比代码中修改的变量是否有问题
        delete_index=[]
        simi_argument_list=list(tmp_simi_parameter.keys())
        variable_name_sign=False
        for j in range(len(modify_argument_list)):
            if j>(len(simi_argument_list)-1) or simi_argument_list[j]!=modify_argument_list[j][0] and ('=' not in new_codes_list[j]):#相似函数对应位置的参数名称不同
                if  modify_argument_list[j][0] in simi_argument_list: # 位置变了，但是还有该变量   
                    if modify_argument_list[j][0] in diff_constraints_list:
                        new_codes_list[j],output_codes_list=process_dimension(output_codes_list,diff_constraints,modify_argument_list[j],new_codes_list[j])
                    new_codes_list[j]=modify_argument_list[j][0]+'='+new_codes_list[j]
                    # delete the arguments that are not used in similar functions
                    if modify_argument_list[j][0] not in simi_argument_list:
                        delete_index.append(j)
                    else:
                        tmp_simi_parameter[modify_argument_list[j][0]]=new_codes_list[j]
                else:# simi func没有该变量,准备统一删除
                    delete_index.append(j)
            else:
                if modify_argument_list[j][0] in diff_constraints_list:
                    new_codes_list[j],output_codes_list=process_dimension(output_codes_list,diff_constraints,modify_argument_list[j],new_codes_list[j])
                # delete the unused argument
                if modify_argument_list[j][0] not in simi_argument_list:
                    delete_index.append(j)
                else:
                    tmp_simi_parameter[modify_argument_list[j][0]]=new_codes_list[j]#modify_argument_list[j][1]
            
            if "=" in new_codes_list[j]:
                variable_name_sign=True
            elif variable_name_sign:# if any position use a=b, then the following arguments should use x=x
                new_codes_list[j]=modify_argument_list[j][0]+'='+new_codes_list[j]
            
        new_codes_list=[new_codes_list[i] for i in range(len(new_codes_list)) if (i not in delete_index)]# delete index
        # 首先判断是不是在别的位置
        # 如果在别的位置，移除原本的，插入新的
        # 如果
        # if simi_parameter_list[j][0]!=inspect.Parameter.empty:
        #     pass
        # # 2. 对比默认变量的值
        for key in default_argument_dict.keys():
            if key in simi_argument_list and tmp_simi_parameter[key]!=default_argument_dict[key]:
                new_codes_list.append(key+'='+str(default_argument_dict[key]))
                tmp_simi_parameter[key]=default_argument_dict[key]
        
        # # 3. 检查是否有必要变量遗漏
        for simi_key in tmp_simi_parameter.keys():
            if tmp_simi_parameter[simi_key]==inspect.Parameter.empty:
                print('Some important arguments are not coverred, exit!!!!')
                return False,None
        
        output_codes=codes_split_list[0]+tmp_key+', '.join(new_codes_list)+final_part
        if cuda_part!=None:
            output_codes+=cuda_part
        output_codes+='\n'
        #+'\n'
        if output_codes!=origin_codes:
            write_new_file=True
        
        output_codes_list.append(output_codes)
        
    # write codes
    if write_new_file:
        new_file_path=os.path.join(os.path.dirname(origin_code_path),os.path.basename(origin_code_path).replace('.py',f'-{simi_func_name}.py'))
        if os.path.exists(new_file_path):
            print('This file has already been generated!!!!')
        fo = open(new_file_path, "w")
        fo.writelines( output_codes_list )
        fo.close()
    else:
        new_file_path=None# not generate new files
    return new_file_path,dimension

def process_dimension(previous_codes_list,diff_constraints,argument,origin_argument):
    # in diff_constraints, dc[key][0] is the constraints of origin argument, dc[key][1] is the constraints of similar function argument
    # we only handle the tuple shape change.
    # output: a modified code segment and a (modified) previous codes, and update dimension (use in modifying data)
    if 'padding' in argument:# not modify padding argument
        return origin_argument,previous_codes_list
    try:
        # int input, no need to modify
        value=int(argument[1])
        return origin_argument,previous_codes_list
    except:
        if '(' in argument[1]:
            # tuple input
            tmp_value=argument[1].replace('(',"").replace(')',"")
            origin_value_list=[int(i) for i in tmp_value.split(',')]
            value=modify_constraint_value(origin_value_list,diff_constraints[argument[0]])
            argument=(argument[0],str(value))
        else:
            # string, a variable
            variable_line=None
            for l in range(len(previous_codes_list)):
                if argument[1]+'=' in previous_codes_list[l]:
                    variable_line=previous_codes_list[l].strip('\n')
                    break
            if variable_line!=None:
                tmp_argument=variable_line.replace(argument[1]+'=','')
                if '(' in tmp_argument:
                    tmp_value=tmp_argument.replace('(',"").replace(')',"")
                    origin_value_list=[int(i) for i in tmp_value.split(',')]
                    value=modify_constraint_value(origin_value_list,diff_constraints[argument[0]])
                    previous_codes_list[l]=previous_codes_list[l].replace(tmp_argument,str(value))
                else:
                    value=int(tmp_value)# if error, the tmp_value still be varible, TODO: in future, handle this situtation
    return argument[1],previous_codes_list

def modify_constraint_value(value,constraints):
    pattern=r"(Tuple\[)(.)*(\]){1}"
    try:
        origin_constraint=re.search(pattern, constraints[0]).group().replace('Tuple[','').replace(']','')
        simi_constraint=re.search(pattern, constraints[1]).group().replace('Tuple[','').replace(']','')
    except:
        print("{}-{}".format(constraints[0],constraints[1]))
        return tuple(value)
    origin_length=len(origin_constraint.split(','))
    if origin_length!=len(value):
        print("length problem, origin value length should be {}, but given {}".format(origin_length,len(value)))
    diff_length=len(simi_constraint.split(','))-origin_length
    if diff_length>0:
        while(diff_length>0):
            value.append(value[-1])# use the -1 axis to expand
            diff_length-=1
    elif diff_length<0:
        while (diff_length<0):
            value.remove(value[-1])# delete the -1 axis to reduce
            diff_length+=1
    else:
        print('error length, two arguments have same length.')
    return tuple(value)

def extract_argument_dict(origin_parameter_dict,new_codes_list):
    argument_dict=copy.deepcopy(origin_parameter_dict)
    new_argument_list=[]
    key_list=list(argument_dict.keys())
    for s in range(len(new_codes_list)):
        if new_codes_list[s].startswith(' '):
            new_codes_list[s]=new_codes_list[s][1:]
        if '=' in new_codes_list[s]:
            argument_name=new_codes_list[s].split('=')[0]
            value=new_codes_list[s].split('=')[1]
            error_sign=True
            
            for arg in key_list:
                if arg==argument_name:
                    # argument_list[arg]=value
                    new_argument_list.append((argument_name,value))
                    del argument_dict[arg]
                    error_sign=False
                    break
            if error_sign:
                print('no argument???')
        else:
            # argument_list[key_list[s]]=new_codes_list[s]
            new_argument_list.append((key_list[s],new_codes_list[s]))
            del argument_dict[key_list[s]]
    
    # for new_argument in new_argument_list:
    #     if new_argument in argument_list:
    #         argument_list.remove(new_argument)
    
    return argument_dict,new_argument_list

def check_constraints(origin_constraint_dict,single_simi_constraint_dict):
    diff_constraint={}
    simi_argument_list=list(single_simi_constraint_dict.keys())
    origin_argument_list=list(origin_constraint_dict.keys())
    for key in origin_argument_list:
        if key in simi_argument_list and origin_constraint_dict[key]!=single_simi_constraint_dict[key]:
            diff_constraint[key]=(origin_constraint_dict[key],single_simi_constraint_dict[key])
    return diff_constraint

def check_sub_list(list1,list2):
    if len(list1)>len(list2):
        return False
    for l in range(len(list1)):
        if list1[l]!=list2[l]:
            return False
    return True

if __name__=="__main__":
    func_cls=nn.AvgPool2d
    result=find_simi_func(func_cls)
    if result!=[]:
        print(True)
    else:
        print(False)
    print(result)