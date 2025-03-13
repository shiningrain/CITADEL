### APIMatcher

#### setup
Since the argument similarity of DeepRel is supplemented in API Matcher, we need to use the open-sourced mongo database in DeepRel.
The setting of their database is shown in [this link](https://github.com/ise-uiuc/DeepREL#2-setting-up-with-dataset).
```
mongorestore -h 127.0.0.1:27017 --db tf dprl/tf/
```

#### usage


In [match_APIs.py](./match_APIs.py), we use the threshold $\beta$ (in RQ3) to match similar APIs and verify their arguments.
The matched and verified API pairs are save in this [pickle file](./result/final_relation_04040.pkl).
This is a dictory and the key are APIs and values are similar APIS of their key.

```python
python match_APIs.py
```