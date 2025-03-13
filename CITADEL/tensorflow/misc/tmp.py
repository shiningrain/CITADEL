import csv
import os
import shutil

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

# csv_list=[]
# with open('/home/zxy/main/DL_compiler_test/1_test_code/CITADEL/CITADEL/tf/misc/raw_issue_list.csv', 'r') as csvfile: 
#     reader = csv.reader(csvfile) 
#     for row in reader:
#         csv_list.append(row)#


# with open('',"w+") as csvfile: 
#     writer = csv.writer(csvfile)
#     writer.writerows(csv_list)

csv_list=[]
total_csv_list=[]
with open('/home/zxy/main/DL_compiler_test/1_test_code/CITADEL/CITADEL/tf/misc/raw_issue_list.csv', 'r') as csvfile: 
    reader = csv.reader(csvfile) 
    for row in reader:
        csv_list.append(row[0])#
        total_csv_list.append(row)

valid_list=[]
result_list=[]
with open('/home/zxy/main/DL_compiler_test/1_test_code/Frame_test_TMP/tf_cases/tmp-0910_tf.csv', 'r') as csvfile: 
    reader = csv.reader(csvfile) 
    for row in reader:
        valid_list.append(row[1])#

file_list=traversalDir_FirstFile("/home/zxy/main/DL_compiler_test/1_test_code/Frame_test_TMP/tf_cases/demo")

for fi in file_list:
    basename=os.path.basename(fi)
    tmp_number=basename[1:].split('.')[0]
    if tmp_number in csv_list:
        shutil.copyfile(fi,os.path.join("/home/zxy/main/DL_compiler_test/1_test_code/CITADEL/CITADEL/tf/cases/total",basename))
        if tmp_number in valid_list:
            valid_list.remove(tmp_number)
            shutil.copyfile(fi,os.path.join("/home/zxy/main/DL_compiler_test/1_test_code/CITADEL/CITADEL/tf/cases/valid",basename))
            result_list.append(total_csv_list[csv_list.index(tmp_number)])
result_list.sort()
result_list.reverse()
with open('/home/zxy/main/DL_compiler_test/1_test_code/CITADEL/CITADEL/tf/misc/detected_bug_list.csv',"w+") as csvfile: 
    writer = csv.writer(csvfile)
    writer.writerows(result_list)

print(valid_list)