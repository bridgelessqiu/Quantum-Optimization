import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import numpy as np 
import os
import json
from tqdm import tqdm

class CircuitDataset(Dataset):
    r""" 
    Create / load a custom dataset of circuit dependency graphs.

    Args:
        root (required): Root dir where the data are stored
            Note:
            root/raw and root/processed contains the raw data
            and generated training/testing data.
            
        filename (required): The name of quantum circuit file
            Note:
            Do not include file extensions.
            
    Examples:
        >> dataset = CircuitDataset(root = root_name, filename = file_name)
        >> len(dataset)
        >> dataset[0].y
    """
    
    def __init__(self, root, filename, transform=None, pre_transform=None, pre_filter=None):
        self.filename = filename
        self.raw_files = [self.filename + '_attr.json', self.filename + '_dependency.edges']
        self.max_num_edge_remove = 2
        self.num_instances = 10
        self.num_edges = 0
        
        with open(os.path.join(root + 'raw', self.raw_files[1]), 'r') as f:
            for edge in f:
                if edge.strip():
                    self.num_edges += 1
        
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        """
        Note:If the raw data files are not found in the raw dir under root, 
            then the download function is executed.
        """
        return self.raw_files

    @property
    def processed_file_names(self):
        """
        Note: If the processed files are not found in the processd dir,
            then process function is executed.
        """
        total = self.num_instances
        for i in range(1, self.max_num_edge_remove + 1):
            total += (self.num_edges - i + 1) * self.num_instances
            
        return [f'{self.filename}_{ind}.pt' for ind in range(total)]

    def download(self):
        """
        Note: Raise an error since our data is not available to
            download for now.
        """
        raise FileNotFoundError(f"Raw data not found")

    def process(self):
        """
        Generate all datapoints
        """
        attr_data = os.path.join(self.raw_dir, self.raw_files[0])
        edge_data = os.path.join(self.raw_dir, self.raw_files[1])
        
        # -- Edges --
        edges = []
        with open(edge_data, 'r') as f:
            for edge in f:
                src, dst = map(int, edge.strip().split())
                edges.append([src, dst])
        edges = torch.tensor(edges, dtype = torch.long).t().contiguous()

        # -- Features --
        node_features = []
        with open(attr_data, 'r') as f:
            json_data = json.load(f)
    
        for node_id, attrs in json_data.items():
            gate = 1 if attrs['gate'] == 'cx' else 1
            q1 = float(attrs["qubit_1"])
            q2 = float(attrs["qubit_2"])
            node_features.append([gate, q1, q2]) 
        node_features = torch.tensor(node_features, dtype=torch.float)

        # -- Generate datapoints --
        ind = 0
        graph_class = 0
        for num_remove in range(self.max_num_edge_remove + 1):
            if num_remove == 0:
                label = torch.tensor([graph_class])
                ind = self.generate_data(edges, node_features, label, ind)
                graph_class += 1
                
            else:
                for i in range(edges.size(1) - num_remove + 1):
                    updated_edges = torch.cat((edges[:, :i], edges[:, i + num_remove : edges.size(1)]), dim = 1)
                    label = torch.tensor([graph_class])
                    ind = self.generate_data(updated_edges, node_features, label, ind)
                    graph_class += 1

    def generate_data(self, edges, node_features, label, ind):
        """
        Generate self.num_instances data points.
        """
        for i in range(self.num_instances):
            data = Data(x = node_features,
                        edge_index = edges,
                        y = label
                       )
            torch.save(data, os.path.join(self.processed_dir, f'{self.filename}_{ind}.pt'))
            ind += 1

        return ind
    
    def len(self):
        return len(self.processed_file_names)

    def get(self, ind):
        data = torch.load(os.path.join(self.processed_dir, f'{self.filename}_{ind}.pt')) 
        return data