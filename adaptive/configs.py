import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
gpus = [1,2,3]
data_path='/mnt/local/Baselines_Bugs/baseline_bert/data'

###
dataset_file='desc_msg.csv'
# dataset_file='bert_data.csv'
mapping_file='mapped_desc_msg.csv'
desc_file='bert_desc.csv'
msg_file='bert_msg.csv'
batch_size= 2500#3000    #1536
save_path='/mnt/local/Baselines_Bugs/DrQA/adaptive/output'
os.makedirs(save_path,exist_ok=True)
debug=False
device = torch.device("cuda" if torch.cuda.is_available() and not debug else 'cpu')
