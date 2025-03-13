# CITADEL

## TL;DR
An automated bug-finding method for DL frameworks.
Orthogonal to existing bug-finding tools, CITADEL aims to find new bugs that are similar to reported ones that have known test oracle.
It can accelerate the finding of new bugs in terms of efficiency and effectiveness.

*Our system is still a prototype, and we will continue to optimize and improve this system.*

## Repo Structure

```
- CITADEL/
    - pytorch/ (tensorflow/ has similar structure)
        - APIMatcher
        - cases
        - demo
        - misc
        - result
        - StaticSimilarity
        - readme.md                               
- SupplementalExperimentResults/   
    - DataForRQ1-3
    - BugList.md
    - readme.md
- OpensourcedData/   
- README.md
- requirements.txt    
```

## Setup

CITADEL is implemented on Python 3.9.0.
It can also execute test cases on different PyTorch (1.7-1.13) and TensorFlow (2.1-2.13) versions through the virtual environment.
To install all dependencies, please get into this directory and run the following command.
```
pip install requirements.txt
```
Then, you need to download Docker environments of PyTorch/TensorFlow to ensure that CITADEL can effectively test the bugs on different versions.
The Docker files can be downloaded from [pytorch/pytorch](https://hub.docker.com/r/pytorch/pytorch) and [tensorflow/tensorflow](https://hub.docker.com/r/tensorflow/tensorflow) of docker_hub.
You can build the container with the following commands:
```
sudo docker run --runtime=nvidia --name YOUR_ENVS_NAME -v YOUR_CITADEL_PATH:/workspace -it -d DOCKER_IMAGE_ID
```
Next, you need to write your CONTAINER ID to the configuration file, for example, [the configuration for PyTorch](./CITADEL/pytorch/CITADEL.conf), therefore CITADEL can load them and execute scripts in them.
You also need to **set the path of this directory** in this configuration file.

## Usage

CITADEL is very easy to use.
You only need to get into the CITADEL directory and execute [this script](./CITADEL/pytorch/run_demo.py) in the installed Python environment to experience the test results on several demo cases on PyTorch (i.e., the motivation case and some cases in the Case Study.).
[Here](./CITADEL/pytorch/readme.md) is a detailed usage instruction.

## Experiment Results

To evaluate the effectiveness of CITADEL in detecting real-world bugs, we conduct experiments on the **PyTorch and TensorFLow repositories** and report the test results to developers for confirmation.
Our experiment counts the number of API bugs detected by CITADEL, that is, when calling an API with certain inputs triggers one bug, CITADEL will consider that an API bug is detected.
Finally, CITADEL detected a total of 151 API bugs in three types, and developers have officially confirmed 92 of them, shown as follows:

|  Framework | #Total |        |       |        | #Rejected |        |       |        | #Duplicated |        |       |        |  #New |        |       |        | #Confirmed |        |       |        |
|:----------:|:------:|:------:|:-----:|:------:|:---------:|:------:|:-----:|:------:|:-----------:|:------:|:-----:|:------:|:-----:|:------:|:-----:|:------:|:----------:|:------:|:-----:|:------:|
|            |  Total | Status | Value | Performance|   Total   | Status | Value | Performance|    Total    | Status | Value | Performance| Total | Status | Value | Performance|    Total   | Status | Value | Performance|
|   PyTorch  |   77   |   52   |   15  |   10   |     1     |    0   |   1   |    0   |      18     |   16   |   2   |    0   |   58  |   36   |   12  |   10   |     36     |   21   |   7   |    8   |
| TensorFlow |   74   |   51   |   20  |    3   |     2     |    2   |   0   |    0   |      6      |    6   |   0   |    0   |   66  |   43   |   20  |    3   |     56     |   36   |   17  |    3   |
|    Total   |   151  |   103  |   35  |   13   |     3     |    2   |   1   |    0   |      24     |   22   |   2   |    0   |  124  |   79   |   32  |   13   |     92     |   57   |   24  |   11   |

We provide a table for reported issues [here](./SupplementalExperimentResults/BugList.md) and a full version of the [Excel form](./SupplementalExperimentResults/BugList.xlsx) here (including the original issue, new bug information, issue status, etc.).


In addition, due to the limitation of the length of the paper, we provide supplemental experimental results in the `SupplementalExperimentResults` folder, you can get more information from [readme.md](./SupplementalExperimentResults/readme.md).

## Reproduction

Take PyTorch as an example.
The instructions in the [readme.md](./CITADEL/pytorch/readme.md) provide methods and guidelines for reproducing our work.

All the necessary open-source data are displayed in the `OpensourcedData` directory, where:

1. The [`static_analyzer_results`](./OpensourcedData/static_analyzer_results/) shows similar source code functions matched by the static analyzer. If you want to reproduce the results of the static analyzer, you can refer to the instructions [here](./CITADEL/pytorch/readme.md).
3. The [`dynamic_profiler_results`](./OpensourcedData/dynamic_profiler_results) shows part of the API call stacks collected by our dynamic profiler, and the names of each file are their corresponding API names.
1. The [`API_matcher_results`](./OpensourcedData/API_matcher_results) contains the raw data of APImatcher, which is saved in pickle files.[`1_final_relation.pkl`](./OpensourcedData/API_matcher_results/pytorch/1_final_relation.pkl) saves the matched similar API and [`1_all_matched_results.pkl`](./OpensourcedData/API_matcher_results/pytorch/1_all_matched_results.pkl) contains more detailed information, including similarity values of argument and context similarity of different API pairs.
4. [`experiment_cases-logs`](./OpensourcedData/experiment_cases-logs) shows the experimental results and original data. The data is the same as that in [this folder](./SupplementalExperimentResults/DataForRQ1-3), which is described in [readme.md](./SupplementalExperimentResults/readme.md). 

