import os
import csv
import pickle
import shutil
import re



def check_function(check_list,paired_list,tmp_dict):
    count=0
    potential_func_list=['','','']
    for description in check_list:
        des1=description.upper()
        for string in paired_list:
            if string not in potential_func_list and (string in description or string.lower() in description or string.upper() in des1):
                if string not in tmp_dict.keys():
                    tmp_dict[string]=0
                tmp_dict[string]+=1
                potential_func_list[count]=string
                count+=1
                # TODO:
                if count>2:
                    return True,potential_func_list
    while '' in potential_func_list:
        potential_func_list.remove('')
    
    if count==0:
        return False,potential_func_list
    else:
        return True,potential_func_list

def check_func_in_code(function_list,codes):
    function_list=list(set(function_list))
    for func in function_list:
        pattern=f"({func})([0-9]d)?\("
        for c in codes:
            result=re.search(pattern, c)
            if result!=None:
                func=result.group()[:-1]
                return True,func
    return False,None

def get_version(version_list):
    if version_list=='None':
        return 'None'
    version_list=''.join([v for v in version_list if v!="'"])
    version_list=version_list[1:-1].split(',')
    score_list=[]
    for version in version_list:
        version_bit=version.strip().split('.')
        if version_bit[0]!='1':
            score_list.append(0)# skip torch 2.0 and torch 0.x
            continue
        power=len(version_bit)
        score=0
        for bit in range(power):
            score+=int(version_bit[bit])*(10**(power-bit))
        score_list.append(score)
    if max(score_list)==0:
        return 'None'
    final_version=version_list[score_list.index(max(score_list))].lstrip()
    return final_version

if __name__=="__main__":

    with open('./covered_api.pkl', 'rb') as f: 
        paired_list = pickle.load(f)

    
    paired_list=[i.split('.')[-1] for i in paired_list]# only keep name
        
    remove_list=['all','tensor','t','le','ge','to','backward','t_','it','it_','set','set_']
    paired_list = [i for i in paired_list if i not in remove_list]# remove high frequency
    paired_list = [i for i in paired_list if len(i)>2]# remove short words
    # combine 1d2d3d together
    new_paired_list=[]
    for pair in paired_list:
        if '1d' == pair[-2:] or '2d' == pair[-2:] or '3d' == pair[-2:]:
            pair=pair[:-2]
        new_paired_list.append(pair)
    new_paired_list=list(set(new_paired_list))
    paired_list = sorted(new_paired_list,key=lambda i: len(i),reverse=True)
    


    tmp_dict={}
    line=0
    new_write_list=[]
    with open('./total-issues.csv', 'r') as csvfile: 
        reader = csv.reader(csvfile) 
        for row in reader:
            if line==0:
                line+=1
                new_write_list.append(row)
                continue

            title=row[1]
            description=row[2]
            result,potential_func_list=check_function([title,description],paired_list,tmp_dict)
            if result:
                row.append(potential_func_list)
                new_write_list.append(row)
            line+=1


    # check whether the func in its codes:
    output_list=[]
    for i in range(len(new_write_list)):
        if i==0:continue
        issue_msg=new_write_list[i]
        code_path=os.path.abspath(issue_msg[3])
        function_list=issue_msg[-1]
        f=open(code_path,'r')
        codes=f.readlines()
        result,function=check_func_in_code(function_list,codes)
        if result:
            new_path=os.path.join('../total_cases',os.path.basename(code_path))
            
            issue_msg.append(function)
            version=get_version(issue_msg[10])#TODO:
            issue_msg=[issue_msg[0],new_path,issue_msg[4],version,
                       issue_msg[10],issue_msg[11],issue_msg[12],]
            output_list.append(issue_msg)
        else:
            print(issue_msg[0])

    import csv
    with open('./sort-issues.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(output_list)
    print(1)
