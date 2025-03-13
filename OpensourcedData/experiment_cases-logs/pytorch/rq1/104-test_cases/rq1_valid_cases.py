import os

import pickle


def traversalDir_FirstFile(path):
    tmplist = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file1 in files:
            m = os.path.join(path,file1)
            if (os.path.isfile(m) ):
                tmplist.append(m)
                # tmplist1.append(file1)
    return tmplist

reported_csv='/home/zxy/main/DL_compiler_test/1_test_code/CITADEL/OpensourcedData/experiment_cases-logs/rq1/104-test_cases/raw_unique_bug-0425.csv'
generated_dir='/home/zxy/main/DL_compiler_test/1_test_code/CITADEL/OpensourcedData/experiment_cases-logs/rq1/104-test_cases/cases'

f=open(reported_csv,'r')
reported_list=f.readlines()
reported_list=[r.strip() for r in reported_list]
f.close()

reported_number_list=[]
for reported in reported_list:
    reported_number_list.append(reported.split(',')[1])

py_list=traversalDir_FirstFile(generated_dir)
py_list=[i for i in py_list if '.pkl' not in i]

print('Total test cases:'+ str(len(py_list)))

count=0
for py in py_list:
    if os.path.basename(py).split('-')[0][1:].replace('.py','') in reported_number_list:
        count+=1

print('valid Test Cases: '+str(count))