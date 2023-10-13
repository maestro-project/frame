import numpy as np
import math
from src.unit import Unit
class System(object):
    compute_multiplier = {'int8': 1, 'bf16': 1, 'f32': 2, 'int4': 1, 'int2':1}
    mem_multiplier = {'int8': 1, 'bf16': 2, 'f32': 4, 'int4':0.5, 'int2':0.25}
    def __init__(self, unit=None, onchip_mem_bw=18000, offchip_mem_bw=900, external_mem_bw=100, 
                 on_chip_mem_size=float('Inf'), off_chip_mem_size=float('Inf'),
                 compute_efficiency=1, memory_efficiency=1, flops=123, mxu_shape=None,
                 frequency=940, bits='bf16', compress_mem=True, skip_compute=True, skip_compute_on_noopt_output=True,
                 pg_gran=None, pe_min_density_support=0.5,accelerator_type="structured",unstructured_efficiency=0.75,
                 model_on_chip_mem_implications=False, num_cores = 1):
        
        if unit is None:
            self.unit = Unit()
        else:
            self.unit = unit
        ## 9000GBps / 940MHz =  9500B per cycle. -> Off chip
        self.onchip_mem_bw = self.unit.unit_to_raw(onchip_mem_bw, type='BW')
        self.offchip_mem_bw = self.unit.unit_to_raw(offchip_mem_bw, type='BW')
        self.external_mem_bw = self.unit.unit_to_raw(external_mem_bw, type='BW')
        self.on_chip_mem_size = self.unit.unit_to_raw(on_chip_mem_size, type='M')
        self.on_chip_mem_left_size = self.unit.unit_to_raw(on_chip_mem_size, type='M')
        self.off_chip_mem_size = self.unit.unit_to_raw(off_chip_mem_size, type='M')
        self.compute_efficiency = compute_efficiency
        self.memory_efficiency = memory_efficiency
        self.mxu_shape = mxu_shape
        self.num_cores = num_cores
        ## This means when skip compute is true, what is the maximum sparsity the hardware can support 
        ## Currently i am assuming the same for input,weights and output, Please change this in future if needed. 
        ## If sparse PE efficiency = 0.5, then it supports 50% sparsity, default is 0.25 so, then it supports 1:4 sparsity.
        ## The minimum value of this can 0, that will mean it supports any random sparsity.
        if(accelerator_type=="unstructured"):
            self.pe_min_density_support = 0.000001
            self.treat_as_dense = False
            # print("Since it is an unstructured accelerator, setting pe minimum density is set to 0.")
        else:
            self.pe_min_density_support = pe_min_density_support 
            self.treat_as_dense = True

        self.compute_type="ideal"
        self.accelerator_type = accelerator_type            ## can be structured or unstructured
        self.unstructured_efficiency = unstructured_efficiency 
        ## 4*128*128*940*2 = 123 TFLOPS
        ## 4*128*128 -> PE arrays
        ## 940 -> Freq
        ## 2 -> MAC = multiple + accumulate
        self.flops = self.unit.unit_to_raw(flops, type='C')
        # flops : # of floating point operations, flops/2 : # of (bf16) operations
        self.op_per_sec = self.flops/2
        self.onchip_mem_bw_FC_array =self.onchip_mem_bw/self.op_per_sec 
        self.onchip_mem_bw_sys_array =self.onchip_mem_bw*128/(self.flops)
        self.frequency = self.unit.unit_to_raw(frequency, type='F')
        self.bits = bits
        self.compress_mem = compress_mem
        self.model_on_chip_mem_implications = model_on_chip_mem_implications 

        self.skip_compute = skip_compute
        self.skip_compute_on_noopt_output = skip_compute_on_noopt_output
        
        
        
        
        ##TODO : Get energy values for each values below
        self.power_per_4_128_128_mxu = 200 if not mxu_shape else 200 / (4*128*128) *np.prod(mxu_shape)
        self.energy_per_4_128_128_mxu = self.power_per_4_128_128_mxu / self.frequency
        self.power_gating_granularity = [1,1,1] if not pg_gran else pg_gran


        ## Energy Value Ref Tab 2 in TPUv4: 10 Lessons paper ISCA 2022. 
        self.energy_per_mac = self.unit.unit_to_raw(0.32 + 0.2, type='E')         ## Mult + Acc + Regs inside the PE (scales down from 65nm by 1.2 so 2/1.2 for 8B read)
        self.energy_per_onchip_access = self.unit.unit_to_raw(8.5, type='E')
        self.energy_per_offchip_access = self.unit.unit_to_raw(37.5, type='E')
        self.energy_per_data_byte_onchip_to_compute = self.unit.unit_to_raw(2, type='E')     ## 8.5pJ for 8B from 32kB. + energy to move through CBs
        self.energy_per_data_byte_core_to_core = self.unit.unit_to_raw(2+2, type='E')         ## Assuming a number similar to 4MB SRAM access so 
        self.energy_per_data_byte_offchip_to_onchip = self.unit.unit_to_raw(37.5, type='E')   ## Assume 300 from HBM, so 300/8 = 

        if mxu_shape:
            self.op_per_sec = np.prod(mxu_shape) * self.frequency
            self.flops = self.op_per_sec * 2
    
    def __str__(self):
        unit = Unit()
        a = f"Accelerator OPS: {unit.raw_to_unit(self.flops,type='C')} TOPS , Freq = {unit.raw_to_unit(self.frequency,type='F')} GHz, Num Cores = {self.num_cores} \n"
        b = f"On-Chip mem size: {unit.raw_to_unit(self.on_chip_mem_size, type='M')} MB , Off-chip mem size:{unit.raw_to_unit(self.off_chip_mem_size, type='M')} MB\n"
        c = f"On-Chip mem BW: {unit.raw_to_unit(self.onchip_mem_bw, type='BW')} GB/s , Off-chip mem BW:{unit.raw_to_unit(self.offchip_mem_bw, type='BW')} GB/s, External-mem BW:{unit.raw_to_unit(self.external_mem_bw, type='BW')} GB/s,\n"
        d = f"Compute type: {self.compute_type} , Realistic mem type: {self.model_on_chip_mem_implications}\n"
        e = f"Sparsity Params: Acc. type: {self.accelerator_type} , Skip compute: {self.skip_compute} , Compress mem: {self.compress_mem}"
        return a+b+c+d+e
    
    def get_params(self):
        unit = Unit()
        a = f"Accelerator OPS: {unit.raw_to_unit(self.flops,type='C')} TOPS , Freq = {unit.raw_to_unit(self.frequency,type='F')} GHz, Num Cores = {self.num_cores}"
        b = f" Off-chip mem size:{unit.raw_to_unit(self.off_chip_mem_size, type='M')/1024} GB "
        c = f" Off-chip mem BW:{unit.raw_to_unit(self.offchip_mem_bw, type='BW')} GB/s, External-mem BW:{unit.raw_to_unit(self.external_mem_bw, type='BW')} GB/s"
        return a+b+c

    def set_pe_min_density_support(self,pe_min_density_support):
        if(self.accelerator_type=="structured" or pe_min_density_support==1):
            self.treat_as_dense = True
            self.pe_min_density_support = pe_min_density_support
        elif(self.accelerator_type=="unstructured"):
            self.pe_min_density_support = 0.000001
            self.treat_as_dense = False

    def set_onchip_mem_bw(self,onchip_mem_bw):
        self.onchip_mem_bw = self.unit.unit_to_raw(onchip_mem_bw, type='BW')

    def set_offchip_mem_bw(self,offchip_mem_bw):
        self.offchip_mem_bw = self.unit.unit_to_raw(offchip_mem_bw, type='BW')

    def get_offchip_mem_bw(self):
        return self.unit.raw_to_unit(self.offchip_mem_bw,type='BW')
    
    def claim_onchip_mem(self, data_sz):
        if data_sz > self.on_chip_mem_left_size:
            raise ValueError(f'Not enough on-chip memory: Need {data_sz}, only has {self.on_chip_mem_size}')
        self.on_chip_mem_left_size -= data_sz
        return self.on_chip_mem_left_size

    def release_onchip_mem(self, data_sz):
        self.on_chip_mem_left_size = max(self.on_chip_mem_size, data_sz + self.on_chip_mem_left_size)
        return self.on_chip_mem_left_size

    def get_bit_multiplier(self, type='C'):
        if type == 'C':
            return self.compute_multiplier[self.bits]
        elif type == 'M':
            return self.mem_multiplier[self.bits]