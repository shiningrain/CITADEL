# GITHUB_TOKEN="xxx"
import sys
import csv
from github import Github
import os
from pprint import pprint
import requests
import time
import collections
import pickle
import re
from tqdm import trange
import copy

GITHUB_TOKEN="xxx"

def check_version(string):
    index_list=[substr.start() for substr in re.finditer('torch',string)]
    index_list+=[substr.start() for substr in re.finditer('Torch',string)]
    des_list=[]
    for index in index_list:
        des_list.append(string[index-5:index+20])
    
    pattern=r'(\d+\.)+\d'
    version_list=[]
    for des in des_list:
        search_result=re.search(pattern, des)
        if search_result!=None:
            version_list.append(search_result.group())
    if version_list == []:
        return None
    
    version_list=list(set(version_list))
    version_list_bk=copy.deepcopy(version_list)
    if len(version_list)>1:
        version_list=[int(ver.split('.')[1]) for ver in version_list]
        version=max(version_list)
    else:
        version=int(version_list[0].split('.')[1])
    return version,version_list_bk
    

def check_reproduce_code(issue,save_dir):
    des,num=issue["description"],issue['number']
    if 'torch' in des or 'Torch' in des:
        version=check_version(des)#(\d+\.)+\d
    if ("```" not in des) and ('import ' not in des):
        return False,None, None
    code_dir=os.path.join(save_dir, str(num // 10000))#,'c'+str(num)+'.py')
    if not os.path.exists(code_dir):
        os.makedirs(code_dir)
    code_path=os.path.join(code_dir,'c'+str(num)+'.py')
    pattern=r"(```)[\s\S]*(import)+[^\`]+(```)"#"(```)[\s\S]*(import)+[\s\S]*(```)"
    search_result=re.search(pattern, des)
    if search_result==None:
        return False,None,None
    origin_code=search_result.group().replace('```','')
    # check 'python' string
    if 'python' in origin_code.split('\n')[0]:
        origin_code=origin_code.replace(origin_code.split('\n')[0],'')
    f=open(code_path,'w')
    f.writelines(origin_code)
    f.close()
    print('finish code extract! save in {}'.format(code_path))
    return True, version,code_path

def closed_without_commit(state,time_line,headers):
    result=True
    if state!='closed':
        return False
    r1=requests.get(time_line, headers=headers).json()
    for time in r1:
        if 'commit_id'in time.keys():
            if time['commit_id'] != None or time['commit_url']!=None:
                result=False
                return result
    return result


if __name__=="__main__":
    code_save_dir='./codes'

    # # type 2: using request
    token = os.getenv('GITHUB_TOKEN', GITHUB_TOKEN)
    owner = "pytorch"
    repo = "pytorch"
    query_url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    issues_payload = {
            "per_page": 100,
            "page": 63,#60,
            "state": "all",#"closed",
            "number_limitation": 60000,
            "max_get":64,# 600
            }
    headers = {'Authorization': f'token {token}'}
    raw=[]
    result=[]
    
    
    r = requests.get(query_url, headers=headers,params=issues_payload).json()
    break_sign=False
    while (issues_payload["page"]<issues_payload["max_get"]):
        if int(r[99]['number'])>90000:
            issues_payload["page"]+=1
            r = requests.get(query_url, headers=headers, params=issues_payload).json()
            continue
        else:
            raw+=r
        print("-------Current Page is {}------".format(issues_payload["page"]))
        if len(r) == 100:
            issues_payload["page"] += 1
            # print("Page-{}".format(issues_payload["page"]))
            r = requests.get(query_url, headers=headers, params=issues_payload).json()
        else:
            break
        for tmp in r:
            try:
                if int(tmp["number"])<=issues_payload["number_limitation"]:
                    break_sign=True
                    break
            except:
                break_sign=True
                break
        if break_sign==True:break
        
    for e in raw:
        try:
            tmp_a=str(e["number"])
        except:
            continue

        issue = collections.OrderedDict()
        issue["id"] = e["id"]
        issue["number"] = e["number"]
        issue["repo_url"] = e["repository_url"]
        issue["issue_url"] = e["url"]
        issue["events_url"] = e["events_url"]
        issue["state"] = e["state"]
        issue["html_url"] = e["html_url"]
        if '/pull/' in e["html_url"]:
            continue
        issue["title"] = e["title"]
        issue["description"] = e["body"]
        issue["comments"] = e["comments"]
        if e['comments']<3:
            continue
        issue["time_line"]=e['timeline_url']
        tmp=closed_without_commit(issue["state"],issue["time_line"],headers)
        if tmp:
            print(e["number"])
            continue
        issue["created_at"] = e["created_at"]
        issue["updated_at"] = e["updated_at"]
        issue["closed_at"] = e["closed_at"]
        issue["label_list"]=[]
        if not e["milestone"]:
            issue["milestone"] = "null"
        else:
            issue["milestone"] = e["milestone"]["title"]

        labels = []

        for label in e["labels"]:
            labelIssue = collections.OrderedDict()
            labelIssue["issue_repo_url"] = e["repository_url"]
            labelIssue["issue_id"] = e["id"]
            labelIssue["issue_number"] = e["number"]
            labelIssue["label_id"] = label["id"]
            labelIssue["label"] = label["name"]
            labels.append(labelIssue)
            issue["label_list"].append(label["name"])
            # result.append(labelIssue)
        issue["labels"] = labels
        

        result.append(issue)


    label_list=[]# add
    new_issue_list=[]
    for i in result:
        if i['labels']==[] or i['comments']<3:
            continue # not bug if no label
        new_issue_list.append(i)

    with open("issues.csv", 'w', encoding='utf-8' ,newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("number", "description", "title", "code_path", "issue_url", "state","#comments","labels",
       "created_at", "uploaded_at", "closed_at",'pytorch version','pytorch version-total'))

        for i in trange(len(new_issue_list)):
            issue=new_issue_list[i]
            try:
                result,version,path=check_reproduce_code(issue,code_save_dir)# only save the issue with codes
                if result:
                    if version==None:
                        version=='None'
                    writer.writerow((issue["number"], issue["description"][:100],  issue["title"],path, issue["html_url"], 
                                    issue["state"], issue["comments"],issue["label_list"], issue["created_at"], issue["updated_at"], issue["closed_at"],version[0],version[1]))
            except Exception as e:
                print(e)

        # =======================================================================================
        print ("Creating labels.csv...")
