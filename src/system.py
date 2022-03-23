import numpy as np
import math
class System(object):
    compute_multiplier = {'int8': 1, 'bf16': 1, 'f32': 2}
    mem_multiplier = {'int8': 1, 'bf16': 2, 'f32': 4}
    def __init__(self, unit, onchip_mem_bw=9000, offchip_mem_bw=900, on_chip_mem_size=float('Inf'),
                 off_chip_mem_size=float('Inf'), compute_efficiency=1, memory_efficiency=1, flops=123.20768, mxu_shape=None,
                 frequency=940, bits='bf16', compress_mem=True, skip_compute=True, skip_compute_on_noopt_output=True,
                 pg_gran=None):
        self.onchip_mem_bw = unit.unit_to_raw(onchip_mem_bw, type='BW')
        self.offchip_mem_bw = unit.unit_to_raw(offchip_mem_bw, type='BW')
        self.on_chip_mem_size = unit.unit_to_raw(on_chip_mem_size, type='M')
        self.on_chip_mem_left_size = unit.unit_to_raw(on_chip_mem_size, type='M')
        self.off_chip_mem_size = unit.unit_to_raw(off_chip_mem_size, type='M')
        self.compute_efficiency = compute_efficiency
        self.memory_efficiency = memory_efficiency
        self.mxu_shape = mxu_shape
        self.flops = unit.unit_to_raw(flops, type='C')
        # flops : # of floating point operations, flops/2 : # of (bf16) operations
        self.op_per_sec = self.flops/2
        self.frequency = unit.unit_to_raw(frequency, type='F')
        self.bits = bits
        self.compress_mem = compress_mem
        self.skip_compute = skip_compute
        self.skip_compute_on_noopt_output = skip_compute_on_noopt_output
        self.power_per_4_128_128_mxu = 200 if not mxu_shape else 200 / (4*128*128) *np.prod(mxu_shape)
        self.energy_per_4_128_128_mxu = self.power_per_4_128_128_mxu / self.frequency
        self.power_gating_granularity = [1,1,1] if not pg_gran else pg_gran
        self.energy_per_mac = unit.unit_to_raw(2.3, type='E')
        self.energy_per_onchip_access = unit.unit_to_raw(2.3, type='E')
        self.energy_per_offchip_access = unit.unit_to_raw(23, type='E')
        if mxu_shape:
            self.op_per_sec = np.prod(mxu_shape) * self.frequency
            self.flops = self.op_per_sec * 2

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