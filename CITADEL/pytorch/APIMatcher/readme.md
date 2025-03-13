## API Matcher

### setup
Since the argument similarity of DeepRel is supplemented in API Matcher, we need to use the open-sourced mongo database in DeepRel.
The setting of their database is shown in [this link](https://github.com/ise-uiuc/DeepREL#2-setting-up-with-dataset).
```
mongorestore -h 127.0.0.1:27017 --db torch dprl/torch/
```

### usage

In [get_similar_scores.py](./get_similar_scores.py), we calculate and collected the context and argument similarity values bewtween APIs and save them in this [file](./result/total_similar_score.pkl).
We have already prepared the file, so it is not required to execute this script.

In [match_APIs.py](./match_APIs.py), we use the threshold $\beta$ (in RQ3) to match similar APIs and verify their arguments.
The matched and verified API pairs are save in this [pickle file](./result/1_final_relation.pkl).
This is a dict and the key are APIs and values are similar APIS of their key.

```python
python match_APIs.py
```