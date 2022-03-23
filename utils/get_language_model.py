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

def create_model(seq_len, name='BERT', data_path='./', method='vanilla', low_rank_ratio=1/8, m_ratio=4, spattn_density=1/16, density=(1.0,1.0,1.0), special_layer_only=False, to_tensorized=False, tensorized_kernel=128):

    model_path = os.path.join(data_path,"model")
    sparsity_file_path = os.path.join(data_path,"sparsity")
    H, D, Df = get_configs(name)
    M = seq_len
    N = seq_len
    if method == 'vanilla':
        layers = get_lanugage_model(H, M, N, D, Df)
        special_layers = [3, 4]
    elif method == 'lowrank':
        rank = ceil(N*low_rank_ratio)
        layers = get_lanugage_model_low_rank(H, M, N, D, Df, rank)
        special_layers = [3, 4, 5, 6]
    elif method == 'kernel':
        layers = get_lanugage_model_kernel(H, M, N, D, Df, m_ratio)
        special_layers = [3, 4, 5, 6]
    elif method == 'sparse':
        layers = get_lanugage_model(H, M, N, D, Df)
        special_layers = [3, 4]
    if to_tensorized:
        layers = tensorized_ff1_ff2(layers, tensorized_kernel)



    name = name + f'_{method}'
    if density:
        densities = np.ones((len(layers), 3), dtype=float) * np.array(density)
        densities[special_layers] = 1.0
        if method == 'sparse':
            densities[3][2] = spattn_density  # logit output
            densities[4][0] = spattn_density
        if special_layer_only:
            densities = densities[special_layers]
        df = pd.DataFrame(densities,columns=['I', 'W', 'O'])
        df.to_csv(os.path.join(sparsity_file_path, name + '.csv'),  header=True, index=None)

    if special_layer_only:
        layers = np.array(layers)[special_layers]
    df = pd.DataFrame(layers, columns=['M', 'N', 'D', 'H', 'Z', 'Z', 'T'])
    df.to_csv(os.path.join(model_path, name + '.csv'),  header=True, index=None)


if __name__ == '__main__':
    model = 'BERT'
    model_path = os.path.join('../',"data/model/language")
    create_model(256, name=model, model_path=model_path)