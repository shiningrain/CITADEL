import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import argparse
import numpy as np
import csv
import re
import subprocess
import pickle

from run_utils import *

def save_result(result_dict,save_dir,log_path,number):
    save_path=os.path.join(save_dir,f'{number}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(result_dict, f)
    f=open(log_path,'a+')#TODO:modify path
    f.writelines([f'{number}\n'])
    f.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='similar function case testing')
    parser.add_argument('--relation_path','-rp',default='./misc/final_relation_0404.pkl', help='relation pickle file')
    parser.add_argument('--case_csv','-cc',default='./misc/detected_bug_list.csv', help='case_dir')
    parser.add_argument('--save_dir','-sd',default='./result', help='save dir')
    parser.add_argument('--log_path','-lp',default='./result/log-new', help='log file')
    parser.add_argument('--fail_path','-fp',default='./result/fail', help='record failure')
    parser.add_argument('--not_reproduce','-nr',default='./result/not_reproduce', help='record the issue that cant reproduce')
    #                     help='perf values the difference of time cost between similar functions; \
    #                         stat tests the crash or other status difference and value evaluate the output difference')
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    case_list=[]
    with open(args.case_csv, 'r') as csvfile: 
        reader = csv.reader(csvfile) 
        for row in reader:
            case_list.append(row)
    with open(args.relation_path, 'rb') as f:#input,bug type,params
        func_relation = pickle.load(f)
    contain_functions=list(func_relation.keys())
    contain_functions_name=[i.split('.')[-1] for i in contain_functions]
    
    for case in case_list:
        if case[0]=='number':
            continue
        
        tested_cases=[]
        if os.path.exists(args.log_path):
            f=open(args.log_path,'r')
            tested_cases=f.readlines()
            f.close()
        if os.path.exists(args.not_reproduce):
            f=open(args.not_reproduce,'r')
            tested_cases+=f.readlines()
            f.close()
        if case[0]+'\n' in tested_cases: # skip the tested cases
            continue
        if os.path.exists(args.fail_path):
            f=open(args.fail_path,'r')
            failed_cases=f.readlines()
            f.close()
            if case[0]+'\n' in failed_cases: # skip the tested cases
                continue
        
        print(f'Testing {case[0]}')
        # get similar functions
        origin_code_path=case[1]
        problem_type=case[7]# p: performance, s: status, v: value
        # if case[8]=='1':
        #     problem_type='v'
        version=case[3]
        function_name=case[6]
        code_name=os.path.basename(origin_code_path)
        new_save_path=os.path.join(args.save_dir,code_name)
        
        candidate_func_list=get_candidate(function_name,contain_functions,contain_functions_name,origin_code_path)# just get the origin function, not list in this implementation
        if len(candidate_func_list)>1:
            print(1)
        if candidate_func_list==[]:
            f=open(args.fail_path,'a+')
            f.writelines([f'{case[0]}\n'])
            f.close()
            continue
        
        # if '.keras' not in candidate_func_list[0] and 'compat.v1.layers.' not in candidate_func_list[0]:
        #     continue
        
        similar_func_list=get_simialr_function(candidate_func_list,func_relation)# TODO: if multiple candidate func?
        # if not os.path.exists(new_save_path):
            # replace the function in codes \ add new lines \ generate new files
        for candidate_func in candidate_func_list:
            Processor=Code_Processor(origin_code_path,new_save_path,function_name,problem_type)
            new_save_path=Processor.preprocess_origin_code()


        outputs=execute_codes(new_save_path,candidate_func_list,similar_func_list,version,problem_type)
        if outputs==[]:
            f=open(args.not_reproduce,'a+')
            f.writelines([f'{case[0]}\n'])
            f.close()
            continue
        
        result=compare_outputs(outputs,candidate_func_list,similar_func_list,problem_type)
        
        save_result(result,args.save_dir,args.log_path,case[0])
        print(f'finish {case[0]}')
    print(1)

# np.random.seed(42)
# torch.manual_seed(0)
  
# save_path='./tmp_result.pkl'
# with open(save_path, 'wb') as f:
#     pickle.dump(r_e_s, f)

