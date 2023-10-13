import os, sys
script_dir = os.path.dirname(__file__)
module_path = script_dir
for _ in range(5):
    if os.path.basename(module_path).lower() =='frame':
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
from src.scale_sim_utils.scalesim_api import *
from src.scale_sim_utils.topology_generator import *
from src.scale_sim_utils.system_config_generator import *

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
    total_data = np.sum(df['Input_a (MB)'] + df['Input_w (MB)'] + df['Output (MB)']) 
    total_MACS = np.sum(df['Num ops (MFLOP)'])
    total_weights = 0; 
    for i in range(len(df)):
        if (df.loc[i, 'Op Type'] != 'Logit' or df.loc[i, 'Op Type'] != 'Attend'):
           total_weights = total_weights + df.loc[i,'Input_w (MB)'] 
    max_memory_footprint = max([df.loc[i, 'Input_a (MB)'] + df.loc[i, 'Input_w (MB)'] + df.loc[i, 'Output (MB)'] for i in range(len(df))])
    # total_energy = np.sum(df['Total energy (uJ)'])
    # total_original_energy = np.sum(df['MXU energy (uJ)'])
    # saved_energy_rate = (total_original_energy-total_energy)/total_original_energy
    ret = {
        'Latency (msec)': [total_latencies],
        'Cycles': [total_cycles],
        'MACs (MFLOPS)': [total_MACS],
        'Total Data (MB)': [total_data],
        'Total Weights (MB)': [total_weights],
        'Parameters  (MB)': [total_parameters],
        'On-chip Memory Footprint (MB)': [max_memory_footprint],
        # 'Energy  (uJ)': [total_energy],
        # 'Saved energy (%)':[saved_energy_rate*100],
    }
    return pd.DataFrame.from_dict(ret)

def analysis_model(model_dims, system, unit, densities, FLAT_enabled=False):
    roofline_list = []
    for i, (dim, density) in enumerate(zip(model_dims, densities)):
        type = op_type_dicts[dim[-1]]
        operator = getattr(operators, type)
        operator_instance = operator(dim=dim, density=density)
        if (FLAT_enabled):
            if(type == 'Logit'):
                operator_instance.set_mem_pin(output='on')
            elif(type == 'Attend'):
                operator_instance.set_mem_pin(input_a='on')
        roofline = operator_instance.get_roofline(system=system, unit=unit)
        if i==0:
            column = roofline.keys()
        roofline_list.append([roofline[c] for c in column])

    # pd.set_option("precision", 3)
    # pd.set_option('display.float_format', lambda x: '%.3f' % x)
    df = pd.DataFrame(np.array(roofline_list,dtype=object), columns=column, dtype=object)

    # df.style.format('{:.2f}')
    # df.to_csv('output/trial.csv')
    return df

def analyze_model( use_attn_model=True, head=16, hidden_size=1024, ff_hidden_size=4096, custom_model='alexnet',  attn_method='vanilla', batch_size=1,
                   low_rank_ratio=0.1, m_ratio=4, custom_sparsity=False,density_input=1, density_weight=1, density_output=1,
                   spattn_density=0.1, seq_len=512, FLAT_enabled = False, onchip_mem_bw=9000, offchip_mem_bw=900, on_chip_mem_size=float('Inf'),
                   off_chip_mem_size=float('Inf'), compute_efficiency=1, memory_efficiency=1, use_flops=True, flops=123.20768,
                   mxu_instance=4, mxu_height=128, mxu_width=128, frequency=940, bits='bf16', skip_compute_on_noopt_output=True,
                   compress_mem=False, skip_compute=False):
    unit = Unit()
    mxu_shape = [mxu_instance, mxu_height, mxu_width] if not use_flops else None
    data_path = os.path.abspath(os.path.join(module_path,"data"))
    model_path = os.path.join(data_path,"model")
    if use_attn_model:
        model = 'custom_attn'
        model_df = create_model(seq_len, name=model, data_path=data_path, low_rank_ratio=low_rank_ratio,
                          m_ratio=m_ratio, method=attn_method, attn_config={'H': head, 'D': hidden_size, 'Df': ff_hidden_size})
        model = model + f'_{attn_method}'
        print('Since using Attn Model, Ignoring the custom model argument')
    else:
        model = custom_model
        model_df = read_model(model, data_path=data_path)
    sparsity_df = create_sparsity_file(num_layers=len(model_df), name=model, method=attn_method, data_path=data_path,  density=(density_input,density_weight,density_output), spattn_density=spattn_density, custom_sparsity=custom_sparsity)
    system = System(unit, mxu_shape = mxu_shape, compress_mem=compress_mem, skip_compute=skip_compute, onchip_mem_bw=onchip_mem_bw,
                    offchip_mem_bw=offchip_mem_bw, on_chip_mem_size=on_chip_mem_size,off_chip_mem_size=off_chip_mem_size,
                    compute_efficiency=compute_efficiency, memory_efficiency=memory_efficiency, flops=flops,
                    frequency=frequency, bits=bits, skip_compute_on_noopt_output=skip_compute_on_noopt_output)
    model_df = get_model_df(model, system, unit, batch_size, data_path, sparsity_df=sparsity_df, model_df=model_df , sparse=True, FLAT_enabled= FLAT_enabled)

    return model_df, (system, unit)

def read_model(model,  data_path='./',):
    m_file_path = os.path.join(data_path,"model")
    m_file = os.path.join(m_file_path, model + ".csv")
    df = pd.read_csv(m_file)
    return df

def get_model_df(model, system, unit, batch_size=1, data_path='./',
                sparsity_df=None, model_df=None, sparse= False, FLAT_enabled=False,
                analysis_mode="Scale-sim"):
    m_file_path = os.path.join(data_path,"model")
    sparsity_file_path = os.path.join(data_path,"sparsity")
    m_file = os.path.join(m_file_path, model + ".csv")
    density_file = os.path.join(sparsity_file_path, model + ".csv")
    if model is not None:
        assert os.path.exists(m_file), f"File {m_file} should exists."
    try:
        df = pd.read_csv(m_file)
    except:
        df = model_df
    model_defs = df.to_numpy()
    batch_sizes = np.ones((len(model_defs), 1)) * batch_size
    #TODO: Change here to have the layer name in the csv file.
    model_defs = np.append(batch_sizes, model_defs, axis=1).astype(int)

    densities = np.ones((len(model_defs), 3), dtype=float)
    try:
        if(sparse):
            try:
                df = pd.read_csv(density_file)
            except:
                df = sparsity_df
            density_defs = df.to_numpy()
            densities[:len(density_defs),:] = density_defs
    except:
        print('[INFO]Use default dense analysis.')
    
    model_df  = analysis_model(model_defs, system, unit, densities, FLAT_enabled)
    if 'scale' in analysis_mode.lower():
        _, batch_list =change_F_to_S(m_file, output_file='./test.csv')

        scalesim_results = run_scale_sim(topology_filename="./test.csv", 
                system_config="/Users/abhimanyu/Work/Dive/Scale-sim/configs/scale.cfg", verbose=True)
        os.remove('./test.csv')
        scalesim_cycles = [a * b * batch_size for a, b in zip(batch_list, scalesim_results)]

        

        model_df['Scale Sim Cycles'] = scalesim_cycles
        model_df['Scale Sim Thrpt (Tflops)'] = model_df['Throughput (Tflops)'] * model_df['Cycles'] / model_df['Scale Sim Cycles']
    return model_df



if __name__ == '__main__':



    model_df = analyze_model()
    print(model_df)

