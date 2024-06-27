import torch
from torch.utils.data import Dataset
from utils import *
import numpy as np

class NssDataset(Dataset):
    def __init__(self, data=""): 
        self.file_list = read_txt(data)
        self.data_iter = np.load(self.file_list[0])["gt_imps"].shape[0]
        self.max_index = len(self.file_list) * self.data_iter
        
    def __len__(self):
        return self.max_index 
    
    def __getitem__(self, index):
        
        file_index = index // self.data_iter
        data_index = index % self.data_iter
        data = np.load(self.file_list[file_index])
        curr_paths = torch.from_numpy(data['curr_paths'][data_index]).float()
        agent_paths = torch.from_numpy(data['agent_paths'][data_index]).long()
        agent_paths_mask = torch.from_numpy(data['agent_paths_mask'][data_index]).bool()
        shortest_path = torch.from_numpy(data['shortest_path']).float()
        obstacle = torch.from_numpy(data['obstacle']).float()
        subagents_idx = torch.from_numpy(data['subagents_idx'][data_index]).long()
        subset_mask = torch.from_numpy(data['subset_mask'][data_index]).bool()
        gt_imps = torch.from_numpy(data['gt_imps'][data_index]).float()

        return (curr_paths, agent_paths, agent_paths_mask, shortest_path, obstacle, subagents_idx, subset_mask), gt_imps
        
