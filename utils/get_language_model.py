import pandas as pd
import os
from math import ceil
import numpy as np

def tensorize_ff(dims, kernel_size):
    M, N, K, *others = dims
    if kernel_size > M or kernel_size > N:
        print(f'[Error] Tensorized kernel size [{kernel_size}] to large. Please pick kernel_size < [{min(M, N)}]')
    input = [K, N]
    weight = [M, K]
    output = [M, N]
    input1 = [kernel_size, K//kernel_size, N]
    weight1 = [kernel_size, kernel_size]
    output1 = [kernel_size, K//kernel_size, N]
    weight2 = [K//kernel_size, M//kernel_size]
    output2 = [kernel_size, M//kernel_size, N]
    layer1 = [kernel_size, N * K//kernel_size, kernel_size, *others]
    layer2 = [M//kernel_size, N * kernel_size, K//kernel_size, *others]
    return layer1, layer2

def tensorized_ff1_ff2(layers, kernel_size):
    ff2 = layers.pop()
    ff1 = layers.pop()
    ff1_1, ff1_2 = tensorize_ff(ff1, kernel_size)
    ff2_1, ff2_2 = tensorize_ff(ff2, kernel_size)
    layers.extend([ff1_1, ff1_2, ff2_1, ff2_2])
    return layers
def get_lanugage_model(H, M, N, D, Df):
    key =           [D, N, D, 1, 1, 1, 3]
    value =         [D, N, D, 1, 1, 1, 3]
    query =         [D, M, D, 1, 1, 1, 3]
    logit =         [H, M, N, D//H, 1, 1, 4]
    attend =        [H, M, N, D//H, 1, 1, 5]
    output =        [D, M, D, 1, 1, 1, 3]
    ffo =           [Df, M, D, 1, 1, 1, 3]
    ffi =           [D, N, Df, 1, 1, 1, 3]
    layers = [key, value, query, logit, attend, output, ffo, ffi]
    return layers

def get_lanugage_model_low_rank(H, M, N, D, Df, rank):
    key =           [D, N, D, 1, 1, 1, 3]
    value =         [D, N, D, 1, 1, 1, 3]
    query =         [D, M, D, 1, 1, 1, 3]
    key_proj =      [rank, D, N, 1, 1, 1, 3]
    query_proj =    [rank, D, N, 1, 1, 1, 3]
    logit =         [H, M, rank, D//H, 1, 1, 4]
    attend =        [H, M, rank, D//H, 1, 1, 5]
    output =        [D, M, D, 1, 1, 1, 3]
    ffo =           [Df, M, D, 1, 1, 1, 3]
    ffi =           [D, N, Df, 1, 1, 1, 3]
    layers = [key, value, query, key_proj, query_proj, logit, attend, output, ffo, ffi]
    return layers

def get_lanugage_model_kernel(H, M, N, D, Df, m_ratio):
    key =           [D, N, D, 1, 1, 1, 3]
    value =         [D, N, D, 1, 1, 1, 3]
    query =         [D, M, D, 1, 1, 1, 3]
    key_proj =      [ceil(m_ratio), N, D, 1, 1, 1, 3]
    query_proj =    [ceil(m_ratio), N, D, 1, 1, 1, 3]
    kv =         [H, m_ratio *D//H, N, D//H, 1, 1, 5]
    kqv =        [H, M, m_ratio *D//H, D//H, 1, 1, 5]
    output =        [D, M, D, 1, 1, 1, 3]
    ffo =           [Df, M, D, 1, 1, 1, 3]
    ffi =           [D, N, Df, 1, 1, 1, 3]
    layers = [key, value, query, key_proj, query_proj, kv, kqv, output, ffo, ffi]
    return layers


def get_configs(name):
    if name == 'BERT':
        D = 768
        H = 12
        Df = 4*D
    elif name == 'TrXL':
        D = 1024
        H = 16
        Df = 4*D
    elif name == 'XLM':
        D = 2048
        H = 16
        Df = 4*D
    return H, D, Df
def create_sparsity_file( num_layers, name='BERT', method='vanilla',data_path='./',  density=(1,1,1), spattn_density=1/16, custom_sparsity=False):
    sparsity_file_path = os.path.join(data_path,"sparsity")
    if custom_sparsity and os.path.exists(os.path.join(sparsity_file_path, name + '.csv')):
        return
    densities = np.ones((num_layers, 3), dtype=float) * np.array(density)
    if method == 'sparse':
        densities[3][2] = spattn_density  # logit output
        densities[4][0] = spattn_density
    df = pd.DataFrame(densities,columns=['I', 'W', 'O'])
    if os.path.exists(sparsity_file_path):
        df.to_csv(os.path.join(sparsity_file_path, name + '.csv'),  header=True, index=None)
    return df

def create_model(seq_len, name='BERT',  data_path='./', method='vanilla', low_rank_ratio=1/8, m_ratio=4, to_tensorized=False,
                 tensorized_kernel=128):

    model_path = os.path.join(data_path,"model")
    H, D, Df = get_configs(name)
    M = seq_len
    N = seq_len
    if method == 'vanilla':
        layers = get_lanugage_model(H, M, N, D, Df)
    elif method == 'lowrank':
        rank = ceil(N*low_rank_ratio)
        layers = get_lanugage_model_low_rank(H, M, N, D, Df, rank)
    elif method == 'kernel':
        layers = get_lanugage_model_kernel(H, M, N, D, Df, m_ratio)
    elif method == 'sparse':
        layers = get_lanugage_model(H, M, N, D, Df)
    if to_tensorized:
        layers = tensorized_ff1_ff2(layers, tensorized_kernel)
    name = name + f'_{method}'
    df = pd.DataFrame(layers, columns=['M', 'N', 'D', 'H', 'Z', 'Z', 'T'])
    if os.path.exists(model_path):
        df.to_csv(os.path.join(model_path, name + '.csv'),  header=True, index=None)
    return df


if __name__ == '__main__':
    model = 'BERT'
    model_path = os.path.join('../',"data/model/language")
    model_df = create_model(256, name=model, data_path=model_path)
    create_sparsity_file(len(model_df), 'BERT_vanilla')