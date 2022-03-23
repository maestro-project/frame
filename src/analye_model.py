import os, sys
script_dir = os.path.dirname(__file__)
module_path = script_dir
for _ in range(5):
    if os.path.basename(module_path) =='roofline_dnn':
        break
    module_path = os.path.abspath(os.path.join(module_path, '../'))
    if module_path not in sys.path:
        sys.path.insert(0,module_path)

from src.unit import Unit
from src.operators import *
import src.operators as operators
from src.operator_base import op_type_dicts
from src.system import System
import pandas as pd
import os
from utils.get_language_model import *

def get_attn_index(df):
    ret = []
    for idx in range(len(df)):
        if df.loc[idx, 'Op Type'] == 'Attend' or df.loc[idx, 'Op Type']  == 'Logit':
           ret.append(idx)
    return ret

def get_summary_table(df):
    total_cycles = np.sum(df['Cycles'])
    total_latencies = np.sum(df['Latency (msec)'])
    attn_idx = get_attn_index(df)
    total_parameters = np.sum(df['Input_w (MB)']) - sum([df.loc[i, 'Input_w (MB)'] for i in attn_idx])
    max_memory_footprint = max([df.loc[i, 'Input_a (MB)'] + df.loc[i, 'Input_w (MB)'] + df.loc[i, 'Output (MB)'] for i in range(len(df))])
    total_energy = np.sum(df['Total energy (uJ)'])
    total_original_energy = np.sum(df['MXU energy (uJ)'])
    saved_energy_rate = (total_original_energy-total_energy)/total_original_energy
    ret = {
        'Latency (msec)': [total_latencies],
        'Cycles': [total_cycles],
        'Parameters  (MB)': [total_parameters],
        'On-chip Memory Footprint (MB)': [max_memory_footprint],
        'Energy  (uJ)': [total_energy],
        'Saved energy (%)':[saved_energy_rate*100],
    }
    return pd.DataFrame.from_dict(ret)

def analysis_model(model_dims, system, unit, densities):
    roofline_list = []
    for i, (dim, density) in enumerate(zip(model_dims, densities)):
        type = op_type_dicts[dim[-1]]
        operator = getattr(operators, type)
        operator_instance = operator(dim=dim, density=density)
        roofline = operator_instance.get_roofline(system=system, unit=unit)
        if i==0:
            column = roofline.keys()
        roofline_list.append([roofline[c] for c in column])

    # pd.set_option("precision", 3)
    # pd.set_option('display.float_format', lambda x: '%.3f' % x)
    df = pd.DataFrame(np.array(roofline_list), columns=column)

    # df.style.format('{:.2f}')
    # df.to_csv('output/trial.csv')
    return df


def get_model_df(model, system, unit, batch_size=1, data_path='./', sparse=True):
    m_file_path = os.path.join(data_path,"model")
    sparsity_file_path = os.path.join(data_path,"sparsity")
    m_file = os.path.join(m_file_path, model + ".csv")
    density_file = os.path.join(sparsity_file_path, model + ".csv")
    df = pd.read_csv(m_file)
    model_defs = df.to_numpy()
    batch_sizes = np.ones((len(model_defs), 1)) * batch_size
    model_defs = np.append(batch_sizes, model_defs, axis=1).astype(int)

    densities = np.ones((len(model_defs), 3), dtype=float)
    if sparse:
        try:
            df = pd.read_csv(density_file)
            density_defs = df.to_numpy()
            densities[:len(density_defs),:] = density_defs
        except:
            print('[INFO]Use default dense analysis.')



    model_df  = analysis_model(model_defs, system, unit, densities)
    return model_df



if __name__ == '__main__':

    # model = 'example'
    # data_path = os.path.join(module_path,"data/")
    #

    method = 'sparse'
    low_rank_ratio = 1/8
    m_ratio = 4
    spattn_density = 0.1
    seq_len = 256
    batch_size = 4
    model = 'BERT'
    unit = Unit()
    system = System(unit, mxu_shape = [4, 128, 128], compress_mem=True, skip_compute=True, skip_compute_on_noopt_output=True)
    data_path = os.path.join(module_path,"data")
    model_path = os.path.join(data_path,"model")
    create_model(seq_len, name=model, data_path=data_path, density=(1,1,1), low_rank_ratio=low_rank_ratio,
                 m_ratio=m_ratio, spattn_density=spattn_density, method=method, special_layer_only=False,
                 to_tensorized=True)
    model_name = model + f'_{method}'

    model_df = get_model_df(model_name, system, unit, batch_size, data_path)
    get_summary_table(model_df)
    print(model_df)

