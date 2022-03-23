import numpy as np

class Unit(object):
    unit_dicts = {'K': 1e3, 'M': 1e6, 'G':1e9, 'T':1e12, 'm': 1e-3, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12}

    def __init__(self, unit_mem='MB', unit_compute='Tflops', unit_time='msec', unit_bw='GBsec', unit_freq='MHz',
                 unit_energy='pJ'):
        self.unit_mem = unit_mem
        self.unit_compute = unit_compute
        self.unit_time = unit_time
        self.unit_bw = unit_bw
        self.unit_freq = unit_freq
        self.unit_energy = unit_energy

    def get_unit_value(self, type):
        if type =='C':
            unit_value = self.unit_dicts[self.unit_compute[0]]
        elif type == 'M':
            unit_value = self.unit_dicts[self.unit_mem[0]]
        elif type == 'T':
            unit_value = self.unit_dicts[self.unit_time[0]]
        elif type == 'BW':
            unit_value = self.unit_dicts[self.unit_bw[0]]
        elif type == 'F':
            unit_value = self.unit_dicts[self.unit_freq[0]]
        elif type == 'E':
            unit_value = self.unit_dicts[self.unit_energy[0]]
        else:
            raise ValueError(f'Wrong unit type: {type}')
        return unit_value

    def raw_to_unit(self, data, type='C'):
        unit_value = self.get_unit_value(type=type)
        return data / unit_value

    def unit_to_raw(self, data, type='C'):
        unit_value = self.get_unit_value(type=type)
        return data * unit_value

