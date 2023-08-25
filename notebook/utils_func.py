from __future__ import print_function
import os, sys
script_dir = os.getcwd()
module_path = script_dir
for _ in range(5):
    module_path = os.path.abspath(os.path.join(module_path, '../'))
    if module_path not in sys.path:
        sys.path.insert(0,module_path)
    if os.path.basename(module_path) =='frame':
        break
from src.unit import Unit
from src.operators import *
import src.operators
from src.operator_base import op_type_dicts
from src.system import System
from utils.display_and_plots import *

import pandas as pd
from src.analye_model import *
import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual, HBox, Label
import ipywidgets as widgets
pd.set_option('display.float_format', lambda x: '%.2f' % x)

def plot_model_func( use_attn_model=True, head=16, hidden_size=1024, ff_hidden_size=4096, custom_model='alexnet',attn_method='vanilla', batch_size=1,
                     low_rank_ratio=0.1, m_ratio=4, custom_sparsity=False,density_input=1, density_weight=1, density_output=1,
                     spattn_density=0.1, seq_len=512, onchip_mem_bw=9000, offchip_mem_bw=900, on_chip_mem_size=float('Inf'),
                     off_chip_mem_size=float('Inf'), compute_efficiency=1, memory_efficiency=1, use_flops=True, flops=123.20768,
                     mxu_instance=4, mxu_height=128, mxu_width=128, frequency=940, bits='bf16', skip_compute_on_noopt_output=True,
                     compress_mem=False, skip_compute=False):
    args = locals()
    model_df, (system, unit) = analyze_model(**args)
    if use_attn_model:
        model = 'custom_attn'
        model = model + f'_{attn_method}'
        print('Since using Attn Model, Ignoring the custom model argument')
    else:
        model = custom_model
    summary_df = get_summary_table(model_df)
    print(f'=========={model} Layer-by-Layer Performance=======')
    display_df(model_df)
    print(f'============Model-wise Summary=====================')
    display(summary_df)
    print(f'==============Layer-wise Roofline==================')
    dot_roofline(model_df, system, unit)

plot_model = interactive(plot_model_func,
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
                         custom_mode=widgets.Dropdown(options=['custom', 'alexnet', 'densenet', 'googlenet', 'mnasnet', 'mobilenet_v2', 'resnet_18', 'resnet_50', 'resnext50_32x4d','shufflenet_v2', 'squeezenet', 'vgg16', 'wide_resnet50'], value='alexnet', style={'description_width': 'initial'},),
                         head=widgets.IntSlider(min=1, max=36, step=1, value=16,style = {'description_width': 'initial'}),
                         hidden_size=widgets.IntSlider(min=1, max=2**14, step=1, value=1024,style = {'description_width': 'initial'}),
                         ff_hidden_size=widgets.IntSlider(min=1, max=2**16, step=1, value=4096,style = {'description_width': 'initial'}),
                         )


