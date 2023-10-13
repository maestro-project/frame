import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

def dot_roofline(df, system, unit = None):
    max_x = max(df['Op Intensity'])
    plot_roofline_background(system, unit, max_x)
    for i in range(len(df)):
        op_intensity = df.loc[i, 'Op Intensity']
        thrpt = df.loc[i, 'Throughput (Tflops)']
        plt.scatter(op_intensity, thrpt, marker='o', label='Frame-'+str(i))
        if 'Scale Sim Thrpt (Tflops)' in df.columns:
           ss_thrpt = df.loc[i, 'Scale Sim Thrpt (Tflops)']
           plt.scatter(op_intensity, ss_thrpt, marker='^', label='Scale-Sim-'+str(i)) 
    plt.legend(loc='lower right')

def color_bound_type(value):
    if value == 'M':
        color = 'red'
    elif value == 'C':
        color = 'green'
    else:
        return
    return 'color: %s' % color

def highlight_max_cycles(s):
    '''
    highlight the maximum in a Series green.
    '''
    is_max = s == s.max()
    return ['background-color: green' if v else '' for v in is_max]

def display_df(df):
    ## Adding % of total time for each operation
    try:
        total_cycles = np.sum(df['Cycles'])
        for i in range(len(df)):
            df.loc[i, '% of total time'] = 100*df.loc[i,'Cycles']/total_cycles
    except:
        pass
    ## reducing display precision
    pd.set_option("display.precision", 3)

    ## Reordering columns
    first_cols = ['Op Type','Dimension','Op Intensity','Num ops (MFLOP)','Input_a (MB)','Input_w (MB)','Output (MB)','Total Data (MB)','Bound','Compute Cycles','Memory Cycles','C/M ratio','Cycles', '% of total time','Throughput (Tflops)']
    last_cols = [col for col in df.columns if col not in first_cols]
    df = df.loc[:,first_cols+last_cols]

    ## Applying colors and gradients to colmns
    df = df.style.background_gradient(cmap='Blues',axis=0,subset=["Cycles","Total Data (MB)"])\
        .background_gradient(cmap='Spectral_r',axis=1,subset=['Input_a (MB)','Input_w (MB)','Output (MB)'])\
        .background_gradient(cmap='Oranges',axis=0,subset=["Op Intensity"])\
        .applymap(color_bound_type, subset=['Bound'])\
        .apply(highlight_max_cycles,axis=1,subset=pd.IndexSlice[:, ['Compute Cycles','Memory Cycles']])
    


    display(df)
    # return df
