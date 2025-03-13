## Supplemental Experiment Results

### RQ1-RQ3
We saved the experimental results and raw data of RQ1-RQ3 in [this folder](./DataForRQ1-3).

Take PyTorch as an example.
The folder `RQ1/pytorch` contains the 196 generated test cases on 104 collected issues.

The [`rq2_api_pair.pkl` file](./DataForRQ1-3/pytorch/RQ2/rq2_api_pair.pkl) in the `RQ2/pytorch` folder contains the covered APIs and pairs of CITADEL and baselines in RQ2 of our paper, and the figures in the folder show the corresponding Venn diagram.

The [`rq3_record.pkl file](./DataForRQ1-3/pytorch/RQ3/rq3_record.pkl) in the `RQ3/pytorch` folder includes API coverage and the number of detected bugs in different values of threshold $\beta$ (10-90).
The three figures correspond to the figures of RQ3 section in our paper, showing changes in indicators such as coverage APIs and detectable bugs.