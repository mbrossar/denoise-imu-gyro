import torch
import os
import pickle
import yaml


def pload(*f_names):
    """Pickle load"""
    f_name = os.path.join(*f_names)
    with open(f_name, "rb") as f:
        pickle_dict = pickle.load(f)
    return pickle_dict

def pdump(pickle_dict, *f_names):
    """Pickle dump"""
    f_name = os.path.join(*f_names)
    with open(f_name, "wb") as f:
        pickle.dump(pickle_dict, f)

def mkdir(*paths):
    '''Create a directory if not existing.'''
    path = os.path.join(*paths)
    if not os.path.exists(path):
        os.mkdir(path)

def yload(*f_names):
    """YAML load"""
    f_name = os.path.join(*f_names)
    with open(f_name, 'r') as f:
        yaml_dict = yaml.load(f)
    return yaml_dict

def ydump(yaml_dict, *f_names):
    """YAML dump"""
    f_name = os.path.join(*f_names)
    with open(f_name, 'w') as f:
        yaml.dump(yaml_dict, f, default_flow_style=False)

def bmv(mat, vec):
    """batch matrix vector product"""
    return torch.einsum('bij, bj -> bi', mat, vec)

def bbmv(mat, vec):
    """double batch matrix vector product"""
    return torch.einsum('baij, baj -> bai', mat, vec)

def bmtv(mat, vec):
    """batch matrix transpose vector product"""
    return torch.einsum('bji, bj -> bi', mat, vec)

def bmtm(mat1, mat2):
    """batch matrix transpose matrix product"""
    return torch.einsum("bji, bjk -> bik", mat1, mat2)

def bmmt(mat1, mat2):
    """batch matrix matrix transpose product"""
    return torch.einsum("bij, bkj -> bik", mat1, mat2)
