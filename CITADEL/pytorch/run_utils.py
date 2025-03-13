import sys
import os
import pickle
from utils import *
import torch as TCH
import subprocess
import time
import numpy as np
import re
from tqdm import trange



def string2function(string):
    # convert string to pytorch functions
    origin_attr_list=string.split('.')[1:]
    func_cls=TCH
    for attrs in origin_attr_list:
        try:
            func_cls=getattr(func_cls, attrs)
        except Exception as e:
            func_cls=getattr(func_cls, attrs.capitalize())# upper 1st character
    return func_cls

def filter_candidate(candidate_list,origin_code_path):
    f=open(origin_code_path,'r')
    codes=f.readlines()
    codes=''.join(codes)
    output_list=[]
    score_list=[]
    for candidate in candidate_list:
        if candidate in codes:
            output_list.append(candidate)
        else:
            candidate_seg=candidate.split('.')
            for i in range(len(candidate_seg)):
                if '.'.join(candidate_seg[i:]) in codes:
                    score_list.append(len(candidate_seg)-i)
    if output_list!=[]:
        return output_list
    else:
        if score_list==[]:
            return score_list
        maximum=max(score_list)
        
        return [candidate_list[i] for i in range(len(candidate_list)) if score_list[i]==maximum]

def get_candidate(name,total_functions,functions_name,origin_code_path):
    candidate=[]
    for i in range(len(functions_name)):
        func=functions_name[i]
        if func==name or name+'1d'==func or name+'2d'==func or name+'3d'==func:#.split('.')[-1]
            candidate.append(total_functions[i])
            # function=string2function(total_functions[i]
    if len(candidate)>1:
        candidate=filter_candidate(candidate,origin_code_path)
    # if 'torch.nn.conv2d' in candidate:
    #     candidate.remove('torch.nn.conv2d')
    return candidate

def get_simialr_function(candidate_list,func_relation):
    similar_function=[]
    for candidate in candidate_list:
        similar_function+=func_relation[candidate]
    return similar_function


class Code_Processor:
    def __init__(self,origin_code_path,new_save_path,candidate_func,method):
        self.new_save_path=new_save_path
        self.func_name=candidate_func
        self.method=method
        f=open(origin_code_path,'r')
        self.codes=f.readlines()
        f.close()

    def add_import(self,codes):
        new_codes=[]
        add_sign=False
        finish_sign=False
        for line in codes:
            if line.lstrip()!='' and line.lstrip()[0]=='#':
                new_codes.append(line)
                continue
            if add_sign and not finish_sign:
                new_codes.append("import sys\n")
                new_codes.append("import pickle\n")
                new_codes.append("sys.path.append('..')\n")
                new_codes.append("from run_utils import string2function\n")
                new_codes.append("func_cls=string2function(sys.argv[1])\n")
                finish_sign=True
            if 'import ' in line:
                add_sign=True
            
            new_codes.append(line)
        return new_codes
    
    def modify_function(self,codes,func_name):
        pattern=f'([a-zA-Z_]*(\.)*)*({func_name})'
        # pattern=r'(\w+\.)*({})'.format(func_name)
        # origin_string=func_name+'('
        target_string='func_cls('
        for i in range(len(codes)):
            if codes[i].lstrip()!='' and codes[i].lstrip()[0]=='#':
                continue
            result=re.search(pattern, codes[i])
            if result!=None:
                codes[i]=codes[i].replace(result.group()+'(',target_string)
        
        if self.method=='v': # need to capture output value
            for j in range(len(codes)):
                pattern="(print ?\()[\s\S]+\)"
                if codes[-1-j]=='\n':
                    continue
                result=re.search(pattern, codes[-1-j])
                if result!=None:
                    if 'print(' in codes[-1-j]:
                        codes[-1-j]=codes[-1-j].replace('print(','r_e_s=')
                    else:
                        codes[-1-j]=codes[-1-j].replace('print (','r_e_s=')
                    index = codes[-1-j].rfind(')')
                    codes[-1-j] = codes[-1-j][:index] + codes[-1-j][index + 1:]
                    break
        
        return codes

    def preprocess_origin_code(self):       
        self.codes=self.add_import(self.codes)
        self.codes=self.modify_function(self.codes,self.func_name)
        self.write_files(self.codes,self.new_save_path)
        
        return self.new_save_path
    
    def write_files(self,codes,new_save_path):
        f=open(new_save_path,'w')
        f.writelines(codes)
        f.close()

def generate_codes(func_list,code_path,input_process=False):
    simi_func_list=[]
    for i in range(len(func_list)):
        simi_func=string2function(func_list[i])
        if simi_func=='':
            continue
        simi_func_list.append(simi_func)
    code_path_dict=code_process(simi_func_list[0],simi_func_list[1:],code_path,method='run')
    if input_process:# prototype, may have bugs here.
        process_input_dimension(code_path_dict)
    return code_path_dict

def process_input_dimension(code_path_dict):
    for key in code_path_dict.keys():
        dim=code_path_dict[key][1]
        if dim==0:
            continue
        path=code_path_dict[key][0]
        
        f= open(path,'r')
        codes=f.readlines()
        f.close()
        
        new_codes=find_solve_input(codes,dim)
        f=open(path,'w')
        f.writelines(new_codes)
        f.close()
    print('Success!! Modified Input Dimension!!')

def get_indent(line):
    indent=''
    for i in line:
        if i==' ':
            indent+=' '
        else:
            break
    return indent

def find_solve_input(codes,dim):
    variable=''
    remnant=''
    input_string=None
    # find input variable
    for line in codes:
        if remnant== '' and 'func_cls(' in line:
            line=line.replace('func_cls(','')#TODO:try
            variable=line.split('=')[0]
            remnant=line.split(variable)[-1]
            variable=variable.replace(' ','')
            pattern=r"{}\([\s\S]+\)".format(variable)
        if variable!= '' and variable in line:
            result=re.search(pattern, line)
            if result!=None:
                input_string=result.group().replace(f'{variable}(','').split(')')[0]

    # process dimension
    if input_string!=None:
        pattern_input=r"{}[\s\S]+=".format(input_string)
        for li in range(len(codes)):
            line=codes[li]
            if 'padding' in line:
                continue# skip padding line
            result=re.search(pattern_input, line)
            if result!=None:
                indent=get_indent(line)
                origin_line=line.replace(' ','')
                pattern_dimension=r"([0-9]*,)+[0-9]*"
                result=re.search(pattern_dimension, origin_line)
                shape=result.group()
                shape_list=shape.split(',')
                if len(shape)<3:
                    break
                while dim>0:
                    shape_list.append(shape_list[-1])
                    dim-=1
                while dim<0:
                    shape_list=shape_list[:-1]
                    dim+=1
                shape_list=[i for i in shape_list if i!='']
                new_shape=','.join(shape_list)

                codes[li]=indent+origin_line.replace(shape,new_shape)
                return codes

    # directly find the longest number list in codes
    candidate_line=None
    candidate_li=None
    candidate=None
    best_length=0
    for li in range(len(codes)):
        line=codes[li]
        if 'padding' in line:
            continue# skip padding line
        pattern_dimension=r"([a-zA-Z0-9_-]*,)+[a-zA-Z0-9_-]*"
        indent=get_indent(line)
        origin_line=line.replace(' ','')
        result=re.search(pattern_dimension, origin_line)
        if result!=None:
            # any error, plz check the regex matching result
            shape=result.group()
            length=len(shape.split(','))
            if length>best_length:
                best_length=length
                candidate_li=li
                candidate_line=origin_line
                candidate=shape

    if candidate_li==None:# cant find, exit.
        print('error!!!! cant find input!!!')
        return 0
    
    shape_list=candidate.split(',')
    while dim>0:
        shape_list.append(shape_list[-1])
        dim-=1
    while dim<0:
        shape_list=shape_list[:-1]
        dim+=1
    shape_list=[i for i in shape_list if i!='']
    new_shape=','.join(shape_list)

    codes[candidate_li]=indent+candidate_line.replace(candidate,new_shape)
    return codes


def execute_codes(code_path,origin_func_list,func_list,version,method,input_process=False):
    # code_name=os.path.basename(code_path)
    docker_dict=read_config()['docker-id']
    root_dir=read_config()['root_dir']['dir_name']
    # DOCKER_13 = docker_dict['DOCKER_13']
    # DOCKER_12 = docker_dict['DOCKER_12']
    # # torch11 is our default environment
    # DOCKER_10 = docker_dict['DOCKER_10']
    # DOCKER_9  = docker_dict['DOCKER_9']
    # DOCKER_8  = docker_dict['DOCKER_8']
    # DOCKER_7  = docker_dict['DOCKER_7']

    dir_name=os.path.dirname(code_path)
    docker_dir_name=os.path.abspath(dir_name).replace(root_dir,'/workspace')
    func_list.insert(0,origin_func_list[0])
    result_list=[]
    error_message_list=[]
    
    if '.' in version:
        version=version.split('.')[1]
    if version!='None':
        version=int(version)
    
    code_path_dict=generate_codes(func_list,code_path,input_process=input_process)# generate codes for some similar functions if neccessary
    # if code_path_dict=={}:
    #     return []
    
    
    for i in trange(len(func_list)):
        # i==0: origin funcion; i=other: similar functions
        func=func_list[i]
        if i==0:
            run_code_path=code_path
        else:
            if code_path_dict=={}:
                run_code_path=code_path# if fail to generate codes, just use the original codes
            elif func.split('.')[-1] not in code_path_dict.keys():
                run_code_path=code_path
            else:
                run_code_path=code_path_dict[func.split('.')[-1]][0]
        
        code_name=os.path.basename(run_code_path)
        use_docker=True
        if version==13:
            command=f"docker exec -it {docker_dict['DOCKER_13']} bash -c 'cd {docker_dir_name} ; python {code_name} {func}'"
        elif version==12:
            command=f"docker exec -it {docker_dict['DOCKER_12']} bash -c 'cd {docker_dir_name} ; python {code_name} {func}'"
        elif version==10:
            command=f"docker exec -it {docker_dict['DOCKER_10']} bash -c 'cd {docker_dir_name} ; python {code_name} {func}'"
        elif version==9:
            command=f"docker exec -it {docker_dict['DOCKER_9']} bash -c 'cd {docker_dir_name} ; python {code_name} {func}'"
        elif version==8:
            command=f"docker exec -it {docker_dict['DOCKER_8']} bash -c 'cd {docker_dir_name} ; python {code_name} {func}'"
        elif version==7:
            command=f"docker exec -it {docker_dict['DOCKER_7']} bash -c 'cd {docker_dir_name} ; python {code_name} {func}'"
        else:
            command=f'cd {dir_name}; python {code_name} {func}'
            use_docker=False
            # default: run in gan test envs
        
        output_value=None
        
        time0=time.time()
        try:
            output=subprocess.run(command, timeout=300, shell=True,capture_output=True)
        except:
            if i==0:
                return []
            else:
                result_list.append('None')
        time_cost=time.time()-time0
        save_path=os.path.join(os.path.dirname(code_path),'tmp_result.pkl')
        if os.path.exists(save_path):
            try:
                with open(save_path, 'rb') as f:#input,bug type,params
                    output_value = pickle.load(f)
            except Exception as e:
                print(e)
                output_value=None
            os.remove(save_path)
        # error_message=str(output.stderr)
        if use_docker:
            error_message=str(output.stdout)
        else:
            error_message=str(output.stderr)
        if error_message!="b''":
            status=error_message.split(r'\n')[-2]
            if 'Error:' in error_message:
                status=error_message.split('Error:')[-1]
        # elif str(output.stdout)!="b''":
        #     status=str(output.stdout)
        else:
            status='None'
        error_message_list.append(error_message)

        if method=='s':
            if i==0 and status=='None':
                return []
            result_list.append(status)
        elif method=='p':
            result_list.append(time_cost)
        elif method=='v':
            if isinstance(output_value, bool):
                output_value=str(output_value)
            elif isinstance(output_value,TCH.Tensor):
                output_value=output_value.float().cpu().numpy()
            elif isinstance(output_value,float) or isinstance(output_value,int) \
            or isinstance(output_value, np.ndarray) or isinstance(output_value, str):
                pass# standard output: pass
            elif isinstance(output_value,tuple) or isinstance(output_value,list):
                output_value=np.array(output_value)
            else:
                if output_value==None:
                    if i==0:
                        return []
                    output_value=status
                # print('warning! unknown output type')
            result_list.append(output_value)
    return result_list

    # def run_code(code):
    #     results = dict()
    #     results[ERR_CPU_KEY] = None
    #     results[ERR_GPU_KEY] = None
    #     results[ERR_HIGH_KEY] = None
    #     results[ERR_LOW_KEY] = None
    #     error = None
    #     MARK_DONE_FLAG=False
    #     try:
    #         exec(code)
    #         MARK_DONE_FLAG = True
    #     except Exception as e:
    #         error = str(e)
    #     return results, error, MARK_DONE_FLAG
    
    
    
def compare_outputs(outputs,
                    candidate_func_list,
                    similar_func_list,
                    method,
                    threshold_s=0.8,
                    threshold_p=0.7,
                    threshold_v=0.8): #TODO: finish value threshold
    result_dict={}
    result_dict['raw_outputs']=outputs
    result_dict['method']=method
    if candidate_func_list[0] not in similar_func_list:
        result_dict['functions']=candidate_func_list+similar_func_list
    else:
        result_dict['functions']=similar_func_list
    result_dict['similar_issue']=[]# add function names
    result_dict['score_list']=[]
    
    origin_output=outputs[0]
    similar_output=outputs[1:]
    
    for i in range(len(similar_output)):
        if method=='s':
            score=similar_score(origin_output,similar_output[i],'string')
            threshold=threshold_s
        elif method=='p':
            score=similar_score(origin_output,similar_output[i],'perf')
            threshold=threshold_p
        elif method=='v':
            if isinstance(origin_output, str):# string output
                score=similar_score(origin_output,similar_output[i],'string')
                threshold=threshold_s
            else:# value output
                if isinstance(similar_output[i],str):
                    score=0
                else:
                    score=similar_score(origin_output,similar_output[i],'value')
                threshold=threshold_v
        result_dict['score_list'].append(score)
        if score>threshold:
            result_dict['similar_issue'].append(result_dict['functions'][i+1])
    if result_dict['similar_issue']!=[]:
        print(result_dict['similar_issue'])
    return result_dict


def get_cos_similar_matrix(v1, v2):
    if len(v1.shape)>1:
        v1=v1.reshape(-1)# reshape
    if len(v2.shape)>1:
        v2=v2.reshape(-1)
    v1=v1.astype(float) # covert to float
    v2=v2.astype(float)
    
    num = float(np.dot(v1, v2))  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0

def similar_score(input1, input2,method):
    from Levenshtein import distance
    if method=='string':
        import textdistance
        score=textdistance.levenshtein.normalized_similarity(input1, input2)
        # l_distance=(1-distance(input, input)/(len(new_string2)))*100
    elif method=='value':
        epsl=1
        if isinstance(input1, float) or isinstance(input1,int):
            # # perf type
            epsl=1e-7
            score=1-(abs(input1-input2)+epsl)/input1
        # elif isinstance(input1,int):
            # score=float(np.allclose(input1, input2, equal_nan=True))
        else:
            if input1.shape!=input2.shape:
                score=0
            else:
                # #epsl=0.000001 inf -> 0
                # score1=1/(np.linalg.norm(input1-input2)+epsl)# 1 -> 0
                try: # use cosine similarity first
                    assert (not (True in np.isnan(input1)))
                    assert (not (True in np.isinf(input1)))
                    score=get_cos_similar_matrix(input1,input2)
                except Exception as e: # if exception
                    score=float(np.allclose(input1, input2, equal_nan=True))
    elif method=='perf':
        epsl=1e-7
        # score=input1/(abs(input1-input2)+epsl)
        score=1-(abs(input1-input2)+epsl)/input1
    return score


if __name__=="__main__":
    s = "[1,2,(3,4),5]"

    tmp_list=[]
    skip_list=['(',')',',','[',']']
    add=''
    for i in s:
        if i in skip_list:
            if add=='':
                continue
            else:
                tmp_list.append(add)
                add=''
                continue
        else:
            add+=i
    print(1)