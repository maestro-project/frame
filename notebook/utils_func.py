from __future__ import print_function
import os, sys
script_dir = os.getcwd()
module_path = script_dir
for _ in range(5):
    module_path = os.path.abspath(os.path.join(module_path, '../'))
    if module_path not in sys.path:
        sys.path.insert(0,module_path)
    if os.path.basename(module_path) =='roofline_dnn':
        break
from src.unit import Unit
from src.operators import *
import src.operators
from src.operator_base import op_type_dicts
from src.system import System
import pandas as pd
from src.analye_model import *
import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual, HBox, Label
import ipywidgets as widgets
pd.set_option('display.float_format', lambda x: '%.2f' % x)

def plot_model_func( use_attn_model=True,  custom_model='alexnet', attn_model='XLA', attn_method='vanilla', batch_size=1,
                     low_rank_ratio=0.1, m_ratio=4, custom_sparsity=False,density_input=1, density_weight=1, density_output=1,
                     spattn_density=0.1, seq_len=512, onchip_mem_bw=9000, offchip_mem_bw=900, on_chip_mem_size=float('Inf'),
                     off_chip_mem_size=float('Inf'), compute_efficiency=1, memory_efficiency=1, use_flops=True, flops=123.20768,
                     mxu_instance=4, mxu_height=128, mxu_width=128, frequency=940, bits='bf16', skip_compute_on_noopt_output=True,
                     compress_mem=False, skip_compute=False):
    args = locals()
    model_df, (system, unit) = analyze_model(**args)
    if use_attn_model:
        model = attn_model
        model = model + f'_{attn_method}'
    else:
        model = custom_model
    summary_df = get_summary_table(model_df)
    print(f'=========={model} Layer-by-Layer Performance=======')
    display(model_df)
    print(f'============Model-wise Summary=====================')
    display(summary_df)
    print(f'==============Layer-wise Roofline==================')
    dot_roofline(model_df, system, unit)

plot_model = interactive(plot_model_func, attn_model=['XLM','BERT', 'TrXL'],
                         attn_method=['vanilla', 'sparse', 'lowrank', 'kernel'],
                         batch_size=widgets.IntSlider(min=1, max=2**12, step=1, value=1,style = {'description_width': 'initial'}),
                         mxu_instance=widgets.IntSlider(min=1, max=1024, step=1, value=4,style = {'description_width': 'initial'}),
                         mxu_height=widgets.IntSlider(min=1, max=1024, step=1, value=128,style = {'description_width': 'initial'}),
                         mxu_width=widgets.IntSlider(min=1, max=1024, step=1, value=128,style = {'description_width': 'initial'}),
                         low_rank_ratio=widgets.FloatSlider(min=1e-5, max=1.0, step=1e-5,value=1,style = {'description_width': 'initial'}),
                         m_ratio=widgets.IntSlider(min=1, max=16, step=1, value=4,style = {'description_width': 'initial'}),
                         spattn_density=widgets.FloatSlider(min=1e-5, max=1.0, step=1e-5, value=1,style = {'description_width': 'initial'}),
                         seq_len=widgets.IntSlider(min=1, max=2**16, step=1, value=512,style = {'description_width': 'initial'}),
                         density_input=widgets.FloatSlider(min=1e-5, max=1.0, step=1e-5,value=1,style = {'description_width': 'initial'}),
                         density_weight=widgets.FloatSlider(min=1e-5, max=1.0, step=1e-5,value=1,style = {'description_width': 'initial'}),
                         density_output=widgets.FloatSlider(min=1e-5, max=1.0, step=1e-5,value=1,style = {'description_width': 'initial'}),
                         onchip_mem_bw=widgets.IntSlider(min=1, max=1e5, step=1, value=20000,style = {'description_width': 'initial'}),
                         offchip_mem_bw=widgets.IntSlider(min=1, max=1e5, step=1, value=900,style = {'description_width': 'initial'}),
                         compute_efficiency=widgets.FloatSlider(min=1e-5, max=1.0, step=1e-5,value=1,style = {'description_width': 'initial'}),
                         memory_efficiency=widgets.FloatSlider(min=1e-5, max=1.0, step=1e-5,value=1,style = {'description_width': 'initial'}),
                         flops=widgets.IntSlider(min=1, max=1e3, step=1, value=123,style = {'description_width': 'initial'}),
                         frequency=widgets.IntSlider(min=1, max=1e5, step=1, value=940,style = {'description_width': 'initial'}),
                         bits=['int8','bf16','f32'],
                         compress_mem=True,
                         skip_compute=True,
                         skip_compute_on_noopt_output=widgets.Checkbox(value=True, style = {'description_width': 'initial'}),
                         use_flops=True,
                         use_attn_model=True,
                         custom_sparsity=False,
                         custom_model=widgets.Dropdown(options=['custom', 'alexnet', 'densenet', 'googlenet', 'mnasnet', 'mobilenet_v2', 'resnet_18', 'resnet_50', 'resnext50_32x4d','shufflenet_v2', 'squeezenet', 'vgg16', 'wide_resnet50'], value='alexnet', style={'description_width': 'initial'},),
                         )


def plot_roofline_background(system, unit, max_x):
    op_intensity = system.flops/system.offchip_mem_bw
    flops = unit.raw_to_unit(system.flops, type='C')
    max_x = max(max_x, op_intensity*2.5)
    turning_points = [[0, 0], [op_intensity, flops], [max_x, flops]]
    turning_points = np.array(turning_points)
    plt.plot(turning_points[:,0], turning_points[:,1], c='grey')

    op_intensity = system.flops/system.onchip_mem_bw
    flops = unit.raw_to_unit(system.flops, type='C')
    turning_points = [[0, 0], [op_intensity, flops], [max_x, flops]]
    turning_points = np.array(turning_points)
    plt.plot(turning_points[:,0], turning_points[:,1], '--', c='grey')

    plt.xlabel('Op Intensity (FLOPs/Byte)')
    plt.ylabel(f'{unit.unit_compute.upper()}')

def dot_roofline(df, system, unit):
    max_x = max(df['Op Intensity'])
    plot_roofline_background(system, unit, max_x)
    for i in range(len(df)):
        op_intensity = df.loc[i, 'Op Intensity']
        thrpt = df.loc[i, 'Throughput (Tflops)']
        plt.scatter(op_intensity, thrpt)
