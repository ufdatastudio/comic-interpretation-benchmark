# CWDE615 3/13/25
# Run evaluation metrics on the data in output.

from transformers import BertTokenizer, BertModel
import numpy as np
import torch
import os

class evaluation:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        self.model = BertModel.from_pretrained("bert-base-multilingual-uncased")

    def vectorizeText(self, text, is_groundtruth):
        encoded_input = self.tokenizer(text, return_tensors='pt', max_length=512)
        with torch.no_grad():
            output = self.model(**encoded_input)
        token_embeddings = output.last_hidden_state
        attention_mask = encoded_input['attention_mask'].unsqueeze(-1)

        # Mean pooling with attention mask
        sum_embeddings = (token_embeddings * attention_mask).sum(dim=1)
        num_tokens = attention_mask.sum(dim=1)
        embedding = sum_embeddings / num_tokens
        if is_groundtruth:
            self.ground_truth = embedding[0]
        else:
            self.vlm_output = embedding[0]
            
    
    def cosine_sim(self):
        return np.dot(self.ground_truth, self.vlm_output) / (np.linalg.norm(self.ground_truth) * np.linalg.norm(self.vlm_output))
    
    def kl_divergence(self):
        return torch.nn.functional.kl_div(self.ground_truth, self.vlm_output, reduction="batchmean", log_target=True)


def read(file):
    with open(file, 'r') as f:
        text = f.read()
    return text

if __name__ == "__main__":
	# TODO: create arg structure for running functions that calculate evaluation metrics (KL Div, Cos Sim.)
    evaluation = evaluation()
    

    ground_truth_dir = os.getenv('GT_DIR')
    ground_truths = os.listdir(ground_truth_dir)
    vlms_dir = os.getenv('VLM_DIR')
    vlms = os.listdir(vlms_dir)
    cosine_sim = {}
    kl_div = {}
    
    for i, ground_truth in enumerate(ground_truths):
        evaluation.vectorizeText([read(ground_truth_dir+"/"+ground_truth)], 1)
        for k, vlm in enumerate(vlms):
            vlm_outputs = os.listdir(vlms_dir+"/"+vlm)
            for j, vlm_output in enumerate(vlm_outputs):
                if (vlm_output.split(".")[0] == ground_truth.split(".")[0]):
                    print(i, "comparing ", ground_truth, vlm_output, "for vlm ", vlm)
                    evaluation.vectorizeText([read(vlms_dir+"/"+vlm+"/"+vlm_output)], 0)
                    if i not in cosine_sim:
                        cosine_sim[i] = []
                    cosine_sim[i].append(evaluation.cosine_sim())
                    if i not in kl_div:
                        kl_div[i] = []
                    kl_div[i].append(evaluation.kl_divergence().item())
                    break

    print(cosine_sim)
    print(kl_div)

    mean_cosine_sim = [0 for i in range(6)]
    mean_kl_div = [0 for i in range(6)]
    denominator = 0

    for i in cosine_sim.keys():
        for j in range(len(cosine_sim[i])):
            mean_cosine_sim[j] += float(cosine_sim[i][j])
            mean_kl_div[j] += kl_div[i][j]
        denominator += 1

    for i in range(6):
        mean_cosine_sim[i] /= denominator
        mean_kl_div[i] /= denominator

    print(mean_cosine_sim)
    print(mean_kl_div)
