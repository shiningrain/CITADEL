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
    f=open(log_path,'a+')
    f.writelines([f'{number}\n'])
    f.close()

    detect_result_csv=os.path.join(save_dir,'detected_bugs_result.csv')
    # Check if the file exists
    file_exists = os.path.isfile(detect_result_csv)

    with open(detect_result_csv, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # If the file doesn't exist, write the header row
        if not file_exists:
            writer.writerow(['origin issue number', 'potential new bugs'])

        # Write each row of data to the file
        # for row in data:
        row=[number,result_dict['similar_issue']]
        writer.writerow(row)



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='similar function case testing')
    parser.add_argument('--relation_path','-rp',default='./misc/final_relation_0408.pkl', help='Mactched API pickle file')
    parser.add_argument('--case_csv','-cc',default='./misc/detected_bug_list.csv', help='case_csv')
    # parser.add_argument('--number','-num',default='83328', help='case_csv')
    # parser.add_argument('--case_csv','-cc',default='./misc/raw_issue_list.csv', help='case_csv')
    parser.add_argument('--save_dir','-sd',default='./result', help='save dir')
    parser.add_argument('--log_path','-lp',default='./result/log', help='log file')
    parser.add_argument('--fail_path','-fp',default='./result/fail', help='record failure')
    parser.add_argument('--not_reproduce','-nr',default='./result/not_reproduce', help='record the issue that cant reproduce')
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
        problem_type=case[-1]# p: performance, s: status, v: value
        version=case[3]
        function_name=case[6]
        code_name=os.path.basename(origin_code_path)
        new_save_path=os.path.join(args.save_dir,code_name)
        if function_name not in contain_functions:
            # if not cover this API
            f=open(args.fail_path,'a+')
            f.writelines([f'{case[0]}\n'])
            f.close()
            continue
        candidate_func_list=[function_name]

        # candidate_func_list=get_candidate(function_name,contain_functions,contain_functions_name,origin_code_path)# just get the origin function, not list in this implementation
        # if len(candidate_func_list)>1:
        #     print(1)
        # if candidate_func_list==[]:
        #     f=open(args.fail_path,'a+')
        #     f.writelines([f'{case[0]}\n'])
        #     f.close()
        #     continue
        
        similar_func_list=get_simialr_function(candidate_func_list,func_relation)
        if not os.path.exists(new_save_path):
            # replace the function in codes \ add new lines \ generate new files
            for candidate_func in candidate_func_list:
                Processor=Code_Processor(origin_code_path,new_save_path,function_name,problem_type)
                new_save_path=Processor.preprocess_origin_code()


        outputs=execute_codes(new_save_path,candidate_func_list,similar_func_list,version,problem_type)# enable the input_process to solve the dimension mismatch problem
        if outputs==[]:
            # if fail to reproduce the bug
            f=open(args.not_reproduce,'a+')
            f.writelines([f'{case[0]}\n'])
            f.close()
            continue
        
        result=compare_outputs(outputs,candidate_func_list,similar_func_list,problem_type)
        
        save_result(result,args.save_dir,args.log_path,case[0])
        print(f'finish {case[0]}')
    print(1)