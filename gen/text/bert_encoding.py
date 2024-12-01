from transformers import BertTokenizer, BertModel
import os, yaml, json
import pickle as pkl
import torch
import numpy as np


device = torch.device('cuda:0')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased").to(device)


text_path = './data/HITR_tasks/HITR_train.json'
text_feat_path = './data/HITR_tasks/HITR_train_text_feat.json'

texts = yaml.load(open(text_path), Loader=yaml.FullLoader)
texts_feat = []
for text in texts:
    encoded_input = tokenizer(text, return_tensors='pt')
    encoded_input = {k:v.to(device) for k,v in encoded_input.items()}
    output = model(**encoded_input)
    text_feat = model(**encoded_input)['pooler_output'][0].cpu().data
    texts_feat.append(text_feat)
texts_feat = torch.stack(texts_feat, dim=0)


# texts_feat_mean = texts_feat.mean(0)
# texts_feat_std = texts_feat.std(0)
# print([np.round(x,2) for x in texts_feat_mean.tolist()])

texts_feat_mean = [-0.84, -0.26, -0.95, 0.82, 0.84, -0.24, 0.8, 0.16, -0.86, -1.0, -0.75, 0.9, 0.97, 0.66, 0.89, -0.69, -0.29, -0.65, 0.16, 0.02, 0.55, 1.0, -0.41, 0.28, 0.46, 0.99, -0.72, 0.91, 0.92, 0.7, -0.64, 0.23, -0.99, -0.16, -0.94, -0.98, 0.49, -0.62, 0.09, 0.15, -0.9, 0.28, 1.0, -0.19, 0.61, -0.3, -1.0, 0.28, -0.88, 0.95, 0.9, 0.89, 0.1, 0.5, 0.58, -0.4, -0.07, 0.16, -0.26, -0.46, -0.6, 0.23, -0.95, -0.78, 0.93, 0.78, -0.27, -0.51, -0.05, -0.02, 0.76, 0.13, -0.31, -0.82, 0.76, 0.36, -0.69, 1.0, -0.63, -0.98, 0.91, 0.8, 0.66, -0.56, 0.44, -1.0, 0.66, -0.22, -0.98, 0.28, 0.6, -0.23, 0.32, 0.67, -0.45, -0.53, -0.33, -0.9, -0.13, -0.44, 0.08, -0.26, -0.25, -0.47, 0.46, -0.53, -0.33, 0.54, 0.27, 0.6, 0.46, -0.34, 0.54, -0.93, 0.68, -0.44, -0.98, -0.71, -0.98, 0.71, -0.51, -0.34, 0.92, -0.41, 0.48, -0.06, -0.92, -1.0, -0.74, -0.78, -0.22, -0.38, -0.96, -0.95, 0.53, 0.94, 0.24, 1.0, -0.42, 0.92, -0.43, -0.68, 0.66, -0.37, 0.87, 0.09, -0.4, 0.23, -0.55, 0.57, -0.83, -0.29, -0.84, -0.9, -0.18, 0.95, -0.68, -0.94, -0.19, -0.19, -0.49, 0.83, 0.81, 0.31, -0.51, 0.33, -0.01, 0.62, -0.83, -0.14, 0.34, -0.41, -0.92, -0.97, -0.52, 0.47, 0.98, 0.68, 0.38, 0.8, -0.36, 0.8, -0.95, 0.98, -0.13, 0.32, -0.7, 0.41, -0.83, 0.06, 0.77, -0.77, -0.76, -0.14, -0.5, -0.35, -0.89, 0.22, -0.38, -0.44, -0.13, 0.9, 0.94, 0.7, 0.46, 0.75, -0.81, -0.42, 0.25, 0.14, 0.17, 0.99, -0.82, -0.26, -0.9, -0.97, -0.06, -0.9, -0.28, -0.62, 0.77, -0.56, 0.56, 0.38, -0.9, -0.68, 0.35, -0.46, 0.36, -0.23, 0.8, 0.95, -0.58, 0.11, 0.96, -0.92, -0.83, 0.59, -0.35, 0.71, -0.64, 0.96, 0.92, 0.56, -0.89, -0.73, -0.63, -0.56, -0.29, -0.1, 0.92, 0.66, 0.47, 0.34, -0.7, 0.98, -0.76, -0.96, -0.87, -0.2, -0.99, 0.92, 0.3, 0.79, -0.53, -0.7, -0.96, 0.62, 0.01, 0.95, -0.5, -0.78, -0.7, -0.93, -0.03, -0.18, -0.63, -0.13, -0.92, 0.5, 0.64, 0.46, -0.89, 1.0, 1.0, 0.94, 0.81, 0.69, -1.0, -0.81, 1.0, -0.98, -1.0, -0.93, -0.71, 0.35, -1.0, -0.29, -0.08, -0.89, 0.77, 0.96, 0.95, -1.0, 0.63, 0.91, -0.72, 0.96, -0.59, 0.96, 0.52, 0.71, -0.19, 0.37, -0.94, -0.78, -0.77, -0.81, 1.0, 0.2, -0.77, -0.81, 0.46, -0.12, -0.08, -0.96, -0.37, 0.62, 0.78, 0.24, 0.24, -0.46, 0.42, 0.35, -0.09, 0.73, -0.9, -0.29, -0.58, 0.12, -0.67, -0.97, 0.95, -0.32, 0.93, 1.0, 0.76, -0.81, 0.8, 0.41, -0.74, 1.0, 0.84, -0.97, -0.59, 0.8, -0.61, -0.69, 1.0, -0.37, -0.79, -0.68, 0.98, -0.99, 1.0, -0.78, -0.95, 0.96, 0.92, -0.71, -0.71, 0.16, -0.61, 0.31, -0.87, 0.6, 0.58, -0.18, 0.87, -0.57, -0.67, 0.43, -0.66, -0.18, 0.95, 0.48, -0.33, -0.13, -0.35, -0.7, -0.95, 0.65, 1.0, -0.42, 0.91, -0.33, 0.0, 0.17, 0.67, 0.62, -0.38, -0.68, 0.88, -0.93, -0.99, 0.64, 0.12, -0.11, 1.0, 0.45, 0.32, 0.43, 0.98, -0.18, 0.19, 0.9, 0.98, -0.35, 0.7, 0.69, -0.92, -0.17, -0.62, 0.08, -0.91, 0.17, -0.95, 0.97, 0.96, 0.57, 0.31, 0.85, 1.0, -0.94, 0.38, 0.22, 0.55, -1.0, -0.62, -0.45, -0.16, -0.85, -0.34, 0.29, -0.96, 0.89, 0.76, -0.96, -0.98, -0.4, 0.76, 0.12, -0.99, -0.61, -0.56, 0.76, -0.36, -0.93, -0.57, -0.34, 0.43, -0.22, 0.65, 0.86, 0.86, -0.94, -0.6, -0.08, -0.81, 0.8, -0.63, -0.92, -0.23, 1.0, -0.32, 0.88, 0.58, 0.69, -0.29, 0.37, 0.95, 0.33, -0.55, -0.9, 0.33, -0.48, 0.72, 0.79, 0.65, 0.84, 0.93, 0.15, -0.19, -0.07, 1.0, -0.12, 0.13, -0.29, -0.33, -0.36, 0.25, 1.0, 0.31, 0.83, -0.99, -0.9, -0.8, 1.0, 0.82, -0.52, 0.69, 0.65, -0.14, 0.62, -0.28, -0.27, 0.16, 0.21, 0.92, -0.6, -0.97, -0.75, 0.51, -0.96, 1.0, -0.54, -0.25, -0.42, -0.58, -0.46, 0.0, -0.97, -0.23, 0.37, 0.93, 0.32, -0.7, -0.91, 0.91, 0.89, -0.91, -0.93, 0.94, -0.96, 0.7, 1.0, 0.49, 0.58, 0.28, -0.33, 0.52, -0.53, 0.61, -0.93, -0.34, -0.23, 0.4, -0.07, -0.55, 0.82, 0.27, -0.68, -0.56, -0.06, 0.51, 0.82, -0.33, -0.11, 0.16, -0.18, -0.88, -0.36, -0.56, -1.0, 0.71, -1.0, 0.67, 0.19, -0.18, 0.86, 0.56, 0.79, -0.71, -0.92, 0.07, 0.81, -0.41, -0.72, -0.69, 0.41, -0.06, 0.14, -0.69, 0.79, -0.25, 1.0, 0.18, -0.71, -0.9, 0.23, -0.17, 1.0, -0.79, -0.96, 0.31, -0.79, -0.77, 0.55, 0.15, -0.75, -0.96, 0.91, 0.69, -0.72, 0.66, -0.32, -0.64, 0.14, 0.94, 0.98, 0.74, 0.79, 0.04, -0.28, 0.96, 0.18, 0.06, 0.13, 1.0, 0.44, -0.88, -0.09, -0.97, -0.27, -0.88, 0.37, 0.34, 0.92, -0.33, 0.93, -0.91, 0.04, -0.83, -0.69, 0.45, -0.92, -0.97, -0.98, 0.77, -0.25, -0.1, 0.2, 0.04, 0.53, 0.46, -1.0, 0.93, 0.41, 0.9, 0.95, 0.75, 0.71, 0.34, -0.97, -0.93, -0.43, -0.25, 0.64, 0.66, 0.76, 0.42, -0.51, -0.33, -0.72, -0.7, -0.99, 0.56, -0.68, -0.81, 0.95, -0.26, -0.13, -0.19, -0.92, 0.77, 0.57, -0.0, 0.12, 0.46, 0.78, 0.93, 0.97, -0.84, 0.69, -0.76, 0.52, 0.91, -0.93, 0.32, 0.66, -0.41, 0.34, -0.28, -0.89, 0.94, -0.22, 0.59, -0.5, 0.13, -0.38, -0.27, -0.79, -0.7, 0.74, 0.35, 0.84, 0.91, -0.05, -0.77, -0.35, -0.69, -0.9, 0.84, -0.09, 0.03, 0.85, -0.06, 0.97, 0.46, -0.52, -0.18, -0.73, 0.66, -0.79, -0.67, -0.62, 0.75, 0.43, 1.0, -0.82, -0.92, -0.65, -0.47, 0.41, -0.7, -1.0, 0.35, -0.76, 0.72, -0.8, 0.87, -0.82, -0.96, -0.3, 0.64, 0.82, -0.43, -0.78, 0.66, -0.68, 0.98, 0.79, -0.74, 0.21, 0.72, -0.88, -0.68, 0.87]
texts_feat_std = [0.04, 0.06, 0.05, 0.05, 0.08, 0.08, 0.07, 0.09, 0.09, 0.0, 0.11, 0.09, 0.01, 0.11, 0.03, 0.11, 0.23, 0.03, 0.05, 0.21, 0.09, 0.0, 0.19, 0.06, 0.07, 0.01, 0.07, 0.02, 0.02, 0.03, 0.07, 0.05, 0.0, 0.06, 0.05, 0.01, 0.07, 0.06, 0.08, 0.06, 0.02, 0.05, 0.0, 0.22, 0.11, 0.05, 0.0, 0.07, 0.02, 0.04, 0.07, 0.1, 0.06, 0.08, 0.06, 0.17, 0.07, 0.07, 0.05, 0.07, 0.05, 0.06, 0.04, 0.05, 0.06, 0.11, 0.09, 0.07, 0.06, 0.06, 0.06, 0.08, 0.11, 0.02, 0.14, 0.06, 0.03, 0.0, 0.12, 0.01, 0.07, 0.11, 0.03, 0.2, 0.17, 0.0, 0.08, 0.07, 0.01, 0.07, 0.11, 0.05, 0.22, 0.04, 0.13, 0.13, 0.06, 0.06, 0.07, 0.14, 0.07, 0.06, 0.09, 0.05, 0.06, 0.07, 0.11, 0.09, 0.15, 0.05, 0.04, 0.05, 0.08, 0.02, 0.04, 0.05, 0.0, 0.03, 0.0, 0.05, 0.13, 0.1, 0.02, 0.19, 0.06, 0.07, 0.07, 0.0, 0.15, 0.08, 0.13, 0.08, 0.01, 0.01, 0.05, 0.02, 0.07, 0.0, 0.08, 0.02, 0.17, 0.14, 0.19, 0.07, 0.06, 0.12, 0.09, 0.04, 0.17, 0.12, 0.07, 0.06, 0.08, 0.02, 0.07, 0.01, 0.15, 0.05, 0.24, 0.08, 0.08, 0.03, 0.08, 0.06, 0.11, 0.05, 0.17, 0.06, 0.04, 0.19, 0.07, 0.06, 0.05, 0.01, 0.06, 0.06, 0.0, 0.06, 0.05, 0.12, 0.04, 0.09, 0.01, 0.0, 0.06, 0.05, 0.12, 0.16, 0.03, 0.16, 0.05, 0.11, 0.05, 0.07, 0.05, 0.06, 0.08, 0.1, 0.07, 0.04, 0.06, 0.02, 0.03, 0.06, 0.22, 0.06, 0.05, 0.06, 0.07, 0.06, 0.07, 0.0, 0.07, 0.06, 0.02, 0.01, 0.06, 0.02, 0.07, 0.06, 0.06, 0.17, 0.16, 0.05, 0.05, 0.05, 0.08, 0.06, 0.06, 0.05, 0.08, 0.04, 0.06, 0.2, 0.01, 0.07, 0.02, 0.09, 0.07, 0.07, 0.04, 0.02, 0.05, 0.15, 0.03, 0.12, 0.11, 0.19, 0.07, 0.2, 0.06, 0.04, 0.06, 0.16, 0.07, 0.02, 0.1, 0.01, 0.06, 0.13, 0.0, 0.06, 0.06, 0.15, 0.07, 0.05, 0.01, 0.11, 0.05, 0.03, 0.14, 0.07, 0.11, 0.02, 0.07, 0.08, 0.2, 0.06, 0.02, 0.05, 0.03, 0.06, 0.08, 0.0, 0.0, 0.01, 0.03, 0.1, 0.0, 0.06, 0.0, 0.02, 0.0, 0.01, 0.09, 0.08, 0.0, 0.06, 0.08, 0.03, 0.12, 0.01, 0.03, 0.0, 0.11, 0.02, 0.04, 0.04, 0.09, 0.01, 0.09, 0.05, 0.07, 0.05, 0.04, 0.05, 0.11, 0.12, 0.0, 0.05, 0.04, 0.04, 0.15, 0.08, 0.15, 0.01, 0.06, 0.18, 0.07, 0.07, 0.06, 0.09, 0.06, 0.16, 0.08, 0.03, 0.03, 0.12, 0.18, 0.21, 0.15, 0.01, 0.01, 0.07, 0.06, 0.0, 0.1, 0.05, 0.09, 0.06, 0.09, 0.0, 0.09, 0.01, 0.05, 0.08, 0.07, 0.06, 0.0, 0.04, 0.12, 0.14, 0.01, 0.0, 0.0, 0.06, 0.01, 0.01, 0.02, 0.12, 0.06, 0.09, 0.14, 0.07, 0.05, 0.15, 0.06, 0.06, 0.02, 0.11, 0.04, 0.05, 0.16, 0.19, 0.04, 0.05, 0.06, 0.07, 0.06, 0.12, 0.01, 0.16, 0.0, 0.13, 0.06, 0.13, 0.05, 0.12, 0.05, 0.06, 0.05, 0.08, 0.08, 0.04, 0.0, 0.06, 0.07, 0.06, 0.0, 0.12, 0.06, 0.15, 0.02, 0.06, 0.1, 0.08, 0.0, 0.07, 0.04, 0.09, 0.07, 0.07, 0.05, 0.06, 0.03, 0.07, 0.01, 0.01, 0.04, 0.05, 0.06, 0.09, 0.0, 0.04, 0.08, 0.2, 0.12, 0.0, 0.09, 0.09, 0.06, 0.1, 0.05, 0.06, 0.01, 0.07, 0.11, 0.02, 0.0, 0.21, 0.09, 0.06, 0.01, 0.1, 0.04, 0.11, 0.08, 0.02, 0.18, 0.05, 0.07, 0.06, 0.04, 0.08, 0.04, 0.05, 0.16, 0.09, 0.04, 0.06, 0.07, 0.07, 0.05, 0.0, 0.1, 0.08, 0.08, 0.07, 0.07, 0.06, 0.04, 0.07, 0.2, 0.08, 0.24, 0.05, 0.05, 0.13, 0.14, 0.03, 0.04, 0.09, 0.11, 0.07, 0.0, 0.09, 0.09, 0.09, 0.07, 0.08, 0.14, 0.0, 0.05, 0.09, 0.0, 0.06, 0.05, 0.0, 0.02, 0.14, 0.07, 0.13, 0.07, 0.11, 0.06, 0.07, 0.06, 0.06, 0.02, 0.07, 0.01, 0.04, 0.08, 0.01, 0.0, 0.06, 0.06, 0.06, 0.2, 0.2, 0.06, 0.01, 0.06, 0.09, 0.02, 0.07, 0.04, 0.02, 0.07, 0.06, 0.07, 0.02, 0.02, 0.01, 0.08, 0.0, 0.06, 0.19, 0.07, 0.09, 0.06, 0.24, 0.07, 0.02, 0.08, 0.06, 0.06, 0.05, 0.2, 0.03, 0.05, 0.03, 0.06, 0.06, 0.06, 0.04, 0.05, 0.06, 0.05, 0.07, 0.04, 0.07, 0.07, 0.0, 0.05, 0.0, 0.14, 0.2, 0.06, 0.03, 0.08, 0.08, 0.06, 0.06, 0.22, 0.03, 0.07, 0.14, 0.05, 0.07, 0.07, 0.06, 0.14, 0.03, 0.06, 0.0, 0.07, 0.08, 0.05, 0.06, 0.06, 0.0, 0.07, 0.01, 0.05, 0.06, 0.03, 0.06, 0.07, 0.07, 0.04, 0.03, 0.15, 0.03, 0.08, 0.06, 0.07, 0.07, 0.05, 0.01, 0.07, 0.05, 0.22, 0.12, 0.01, 0.07, 0.16, 0.05, 0.0, 0.06, 0.02, 0.18, 0.01, 0.08, 0.03, 0.06, 0.08, 0.02, 0.06, 0.02, 0.07, 0.07, 0.1, 0.14, 0.06, 0.02, 0.01, 0.0, 0.06, 0.06, 0.07, 0.05, 0.06, 0.07, 0.07, 0.0, 0.01, 0.06, 0.07, 0.01, 0.1, 0.08, 0.05, 0.01, 0.04, 0.08, 0.04, 0.06, 0.06, 0.05, 0.07, 0.04, 0.08, 0.16, 0.1, 0.0, 0.05, 0.14, 0.09, 0.01, 0.12, 0.06, 0.19, 0.05, 0.1, 0.1, 0.1, 0.05, 0.05, 0.06, 0.02, 0.01, 0.09, 0.07, 0.15, 0.06, 0.04, 0.01, 0.08, 0.13, 0.09, 0.06, 0.06, 0.04, 0.02, 0.07, 0.09, 0.05, 0.08, 0.05, 0.04, 0.04, 0.09, 0.03, 0.15, 0.03, 0.04, 0.06, 0.06, 0.07, 0.15, 0.02, 0.05, 0.07, 0.21, 0.09, 0.06, 0.01, 0.17, 0.07, 0.07, 0.05, 0.08, 0.06, 0.05, 0.06, 0.05, 0.06, 0.0, 0.11, 0.06, 0.11, 0.1, 0.04, 0.07, 0.0, 0.07, 0.13, 0.1, 0.11, 0.09, 0.11, 0.01, 0.07, 0.08, 0.11, 0.05, 0.09, 0.04, 0.17, 0.02, 0.05, 0.16, 0.15, 0.04, 0.07, 0.03, 0.04]

texts_feat_mean = torch.tensor(texts_feat_mean)
texts_feat_std = torch.tensor(texts_feat_std)
texts_feat_std = torch.clamp_min(texts_feat_std, 1e-2)
texts_feat = (texts_feat - texts_feat_mean) / texts_feat_std
texts_feat = texts_feat.cpu().numpy().tolist()

texts_feat = [{
    'text'      : t,
    'text_feat' : f,
} for t,f in zip(texts, texts_feat)]

print(f'Save to ==> {text_feat_path}')
with open(text_feat_path,'w') as f:
    json.dump(texts_feat, f)