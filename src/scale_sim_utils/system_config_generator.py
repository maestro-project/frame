from src.unit import Unit
def create_scale_sim_config_file(system, output_file_name='./system.cfg'):
    shape = system.mxu_shape
    assert len(shape) > 2, f"For scale sim please provide mxu shape as [ArrayHeight, ArrayWidth], shape={shape}"
    unit = Unit()
    offchip_mem_bw = int(unit.raw_to_unit(system.offchip_mem_bw, type='BW'))
    if system.on_chip_mem_size == float('Inf'):
        on_chip_memory = 10240
    else:
        on_chip_memory = int(unit.raw_to_unit(system.on_chip_mem_size, type='M')/3)

    file = open(output_file_name, "w")
    file.write("[general]\n")
    file.write("run_name = scale_compute_example\n")
    file.write("\n")
    file.write("[architecture_presets]\n")
    file.write(f"ArrayHeight:    {shape[-2]}\n")
    file.write(f"ArrayWidth:     {shape[-1]}\n")
    file.write(f"IfmapSramSzkB:   {on_chip_memory} \n")
    file.write(f"FilterSramSzkB:  {on_chip_memory} \n")
    file.write(f"OfmapSramSzkB:   {on_chip_memory} \n")
    file.write("IfmapOffset:    0\n")
    file.write("FilterOffset:   10000000\n")
    file.write("OfmapOffset:    20000000\n")
    file.write(f"Bandwidth : {offchip_mem_bw}\n")
    file.write("Dataflow : os\n")
    file.write("MemoryBanks:   1\n")
    file.write("\n")
    file.write("[run_presets]\n")
    file.write("InterfaceBandwidth: USER\n ")
    file.close()