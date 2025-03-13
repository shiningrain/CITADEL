
# MPS Version
from transformers import AutoTokenizer, BertForSequenceClassification
import timeit
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = BertForSequenceClassification.from_pretrained("bert-base-cased").to(torch.device("mps"))

tokens = tokenizer.tokenize("Hello world, this is michael!")
tids = tokenizer.convert_tokens_to_ids(tokens)
with torch.no_grad():
    t_tids = torch.tensor([tids]*64, device=torch.device("mps"))
    res = timeit.timeit(lambda: model(input_ids=t_tids), number=100)
print(res)
