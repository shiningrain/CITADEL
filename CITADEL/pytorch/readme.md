## How to use CITADEL

### Bug Detection

[`run_demo.py`](./run_demo.py) and [`run_detect_bugs.py`](./run_detect_bugs.py) are scripts for bug detection and identification.

[`run_demo.py`](./run_demo.py) provides several demo cases, you can directly use this script to detect and identify 5 bugs in the [demo.csv](./demo/demo.csv).
These bugs include the motivation case, the cases in the Case Study of the Evaluation and the Design sections.
Their test results will be saved in the [demo](./demo) folder, and a brief table for them will be created in the [detected_bugs_result.csv](./demo/detected_bugs_result.csv).

[`run_detect_bugs.py`] provides a complete test pipeline, where:
1. `--case_csv` assigns the path of the origin issue list, you can choose only to test [the cases that can detect new bugs by CITADEL](./misc/detected_bug_list.csv) or test [all 104 cases](./misc/raw_issue_list.csv).
2. `--relation_path` specifies the path of matching result of similar APIs, the default path is [this](./misc/final_relation_0408.pkl).
3. `--save_dir` specifies the path to save the detection results, the default is [the `result` directory](./result).
4. `--log_path/fail_path/not_reproduce` specify several paths to store the results of different exceptions, which are defaulted to be in [the `result` directory](./result)/

### Optional: Static Analyzer

In `StaticSimilarity` folder, we provide codes and data of the static analyzer.
You can get into this folder and then run [static_analyzer.py](./StaticSimilarity/static_analyzer.py) to match and save similar source code functions.

The results will be saved in the [csv file](./StaticSimilarity/new_similar_pair.csv) and [pkl file](./StaticSimilarity/new_similar_pair.pkl).
They show the source code functions that share static similarities and are clustered in groups.

```python
cd ./StaticSimilarity
python static_analyzer.py
```

### Optional: APIMatcher

#### setup
Since the argument similarity of DeepRel is supplemented in API Matcher, we need to use the open-sourced mongo database in DeepRel.
The setting of their database is shown in [this link](https://github.com/ise-uiuc/DeepREL#2-setting-up-with-dataset).
```
mongorestore -h 127.0.0.1:27017 --db torch dprl/torch/
```

#### usage

In [get_similar_scores.py](./APIMatcher/get_similar_scores.py), we calculate and collected the context and argument similarity values between APIs and save them in this [file](./APIMatcher/result/total_similar_score.pkl).
We have already prepared the file, so it is not required to execute this script.

In [match_APIs.py](./APIMatcher/match_APIs.py), we use the threshold $\beta$ (in RQ3) to match similar APIs and verify their arguments.
The matched and verified API pairs are save in this [pickle file](./APIMatcher/result/1_final_relation.pkl).
This is a dictory and the key are APIs and values are similar APIS of their key.

```python
python match_APIs.py
```


### Note:
*Since this implementation is currently a prototype of CITADEL, there may be bugs or exceptions in execution, and we will fix and improve them in time after we find these problems.
As we mentioned in our paper, some APIs have expected abnormal behavior (e.g., [c73622.py](./demo/c73622.py) are exactly), therefore the detection results need to be verified based on the generated test cases and the pickle file after testing (e.g., [c73622.py](./demo/c73622.py) and [73622.pkl](./demo/73622.pkl)).*
