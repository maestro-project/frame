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
    # total_energy = np.sum(df['Total energy (uJ)'])
    # total_original_energy = np.sum(df['MXU energy (uJ)'])
    # saved_energy_rate = (total_original_energy-total_energy)/total_original_energy
    ret = {
        'Latency (msec)': [total_latencies],
        'Cycles': [total_cycles],
        'Parameters  (MB)': [total_parameters],
        'On-chip Memory Footprint (MB)': [max_memory_footprint],
        # 'Energy  (uJ)': [total_energy],
        # 'Saved energy (%)':[saved_energy_rate*100],
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

def analyze_model( use_attn_model=True,  custom_model='alexnet', attn_model='XLM', attn_method='vanilla', batch_size=1,
                   low_rank_ratio=0.1, m_ratio=4, custom_sparsity=False,density_input=1, density_weight=1, density_output=1,
                   spattn_density=0.1, seq_len=512, onchip_mem_bw=9000, offchip_mem_bw=900, on_chip_mem_size=float('Inf'),
                   off_chip_mem_size=float('Inf'), compute_efficiency=1, memory_efficiency=1, use_flops=True, flops=123.20768,
                   mxu_instance=4, mxu_height=128, mxu_width=128, frequency=940, bits='bf16', skip_compute_on_noopt_output=True,
                   compress_mem=False, skip_compute=False):
    unit = Unit()
    mxu_shape = [mxu_instance, mxu_height, mxu_width] if not use_flops else None
    data_path = os.path.abspath(os.path.join(module_path,"data"))
    model_path = os.path.join(data_path,"model")
    if use_attn_model:
        model = attn_model
        model_df = create_model(seq_len, name=model, data_path=data_path, low_rank_ratio=low_rank_ratio,
                          m_ratio=m_ratio, method=attn_method,)
        model = model + f'_{attn_method}'
    else:
        model = custom_model
        model_df = read_model(model, data_path=data_path)
    sparsity_df = create_sparsity_file(name=model, method=attn_method, data_path=data_path,  density=(density_input,density_weight,density_output), spattn_density=spattn_density, custom_sparsity=custom_sparsity)
    system = System(unit, mxu_shape = mxu_shape, compress_mem=compress_mem, skip_compute=skip_compute, onchip_mem_bw=onchip_mem_bw,
                    offchip_mem_bw=offchip_mem_bw, on_chip_mem_size=on_chip_mem_size,off_chip_mem_size=off_chip_mem_size,
                    compute_efficiency=compute_efficiency, memory_efficiency=memory_efficiency, flops=flops,
                    frequency=frequency, bits=bits, skip_compute_on_noopt_output=skip_compute_on_noopt_output)
    model_df = get_model_df(model, system, unit, batch_size, data_path, sparsity_df=sparsity_df, model_df=model_df )

    return model_df, (system, unit)

def read_model(model,  data_path='./',):
    m_file_path = os.path.join(data_path,"model")
    m_file = os.path.join(m_file_path, model + ".csv")
    df = pd.read_csv(m_file)
    return df

def get_model_df(model, system, unit, batch_size=1, data_path='./', sparsity_df=None, model_df=None):
    m_file_path = os.path.join(data_path,"model")
    sparsity_file_path = os.path.join(data_path,"sparsity")
    m_file = os.path.join(m_file_path, model + ".csv")
    density_file = os.path.join(sparsity_file_path, model + ".csv")
    try:
        df = pd.read_csv(m_file)
    except:
        df = model_df
    model_defs = df.to_numpy()
    batch_sizes = np.ones((len(model_defs), 1)) * batch_size
    model_defs = np.append(batch_sizes, model_defs, axis=1).astype(int)

    densities = np.ones((len(model_defs), 3), dtype=float)
    try:
        try:
            df = pd.read_csv(density_file)
        except:
            df = sparsity_df
        density_defs = df.to_numpy()
        densities[:len(density_defs),:] = density_defs
    except:
        print('[INFO]Use default dense analysis.')



    model_df  = analysis_model(model_defs, system, unit, densities)
    return model_df



if __name__ == '__main__':



    model_df = analyze_model()
    print(model_df)

