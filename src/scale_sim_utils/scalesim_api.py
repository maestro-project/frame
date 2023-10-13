
from scalesim.scale_sim import scalesim
import os, time


#TODO: Change it later to help user locally change the data by generating config + topo files
def run_scale_sim(topology_filename, system_config='/configs/usr_config.cfg', verbose=False):


    #TODO: Change to GEMM later
    gemm_input = False
    log_path = os.getcwd() + "/scale_sim_logs/"

    #TODO: Turn off the verbose
    s = scalesim(save_disk_space=True, verbose=verbose,
                 config=system_config,
                 topology=topology_filename,
                 input_type_gemm=gemm_input
                 )
    s.run_scale(top_path=log_path)



    # total_cycles = 0
    layer_wise_cycles = []
    for layer_obj in s.runner.single_layer_sim_object_list:
        layer_wise_cycles.append(int(layer_obj.total_cycles))
        

    return layer_wise_cycles

