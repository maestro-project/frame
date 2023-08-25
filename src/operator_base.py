import numpy as np
from operator import mul
from math import ceil
op_type_dicts = {0: 'FC', 1: 'CONV2D', 2: 'DWCONV', 3: 'GEMM', 4: 'Logit', 5: 'Attend'}
class Operator(object):
    def __init__(self, dim, density=(1.0,1.0,1.0)):
        self.dim = dim
        self.density_a, self.density_w, self.density_o = density
        self.input_a, self.input_w, self.output = self.get_tensors()
        self.num_ops = self.get_num_ops()
        self.set_mem_pin(*self.get_default_mem_loc())

    def get_default_mem_loc(self):
        return ['off', 'off', 'off']

    def set_mem_pin(self, input_a=None, input_w=None, output=None):
        if input_a is not None:
            self.input_a_loc = input_a
        if input_w is not None:
            self.input_w_loc = input_w
        if output is not None:
            self.output_loc = output

    def set_tensor(self, input_a=None, input_w=None, output=None):
        if input_a is not None:
            self.input_a = input_a
        if input_w is not None:
            self.input_w = input_w
        if output is not None:
            self.output = output

    def get_density_list(self):
        return [self.density_a, self.density_w, self.density_o]

    def get_op_type(self, dim):
        return op_type_dicts[dim[-1]]

    def get_tensors(self):
        pass

    def get_size(self, tensor):
        return np.prod(tensor)

    def get_num_ops(self):
        pass

    def get_effective_dim_len(self):
        pass

    def get_num_data(self):
        return sum(self.get_sz_list())

    def get_effective_num_data(self, system):
        return sum(self.get_sz_list(system))

    def get_ideal_compute_time(self, system):
        return self.get_effective_num_ops(system) * system.get_bit_multiplier(type='C')/system.op_per_sec

    def get_ideal_memory_time(self, system):
        sz_list = self.get_sz_list(system)
        memory_time_onchip = 0
        memory_time_offchip = 0
        for tensor_sz in sz_list:
            memory_time_onchip += tensor_sz * system.get_bit_multiplier(type='M')/ system.onchip_mem_bw
            memory_time_offchip += tensor_sz * system.get_bit_multiplier(type='M')/ system.offchip_mem_bw
        return  memory_time_offchip, memory_time_onchip



    def get_compute_efficiency(self, mxu_shape, mxu_mapping):
        outer_iters = ceil(mxu_mapping[0]/mxu_shape[0])
        inner_iters = []
        for mxu_size, dim_size in zip(mxu_shape[1:], mxu_mapping[1:]):
            inner_iter_cur = ceil(dim_size/mxu_size)
            inner_iters.append(inner_iter_cur)
        iters = [outer_iters] + inner_iters
        num_iters = np.prod(iters)
        efficiency = np.prod(mxu_mapping) / (num_iters * np.prod(mxu_shape))
        return num_iters, efficiency



    def get_effective_mxu_mapping(self, system):
        left, upper, contract, outer = self.get_gemms()
        if system.skip_compute:
            contract = contract * self.density_w * self.density_a
            if contract < 1:
                print(f'[Warning] Contract dimension < 1, after sparsified')
            if system.skip_compute_on_noopt_output:
                left = left *self.density_o
        mxu_mapping = np.sort([left, upper, contract])[::-1]
        *mxu_mapping, streaming_dim = [outer] + [m for m in mxu_mapping]
        return mxu_mapping, streaming_dim

    def get_effective_mxu_shape(self, mxu_shape):
        effective_mxu_shape = np.sort(mxu_shape[-2:])[::-1]
        effective_mxu_shape = [mxu_shape[0]] + [m for m in effective_mxu_shape]
        return effective_mxu_shape

    def get_compute_time(self, system):
        if system.mxu_shape is not None:
            mxu_mapping, _ = self.get_effective_mxu_mapping(system)
            effective_mxu_shape = self.get_effective_mxu_shape(system.mxu_shape)
            _ , compute_efficiency = self.get_compute_efficiency(effective_mxu_shape, mxu_mapping)
        else:
            compute_efficiency = 1
        return self.get_effective_num_ops(system) * system.get_bit_multiplier(type='C')/system.op_per_sec / compute_efficiency, compute_efficiency

    def get_mxu_energy(self, system):
        if system.mxu_shape is not None:
            mxu_mapping, streaming_dim = self.get_effective_mxu_mapping(system)
            power_gating_mxu_shape = [ m//g for m, g in zip(system.mxu_shape, system.power_gating_granularity)]
            effective_mxu_shape = self.get_effective_mxu_shape(power_gating_mxu_shape)
            pg_num_iters, compute_efficiency = self.get_compute_efficiency(effective_mxu_shape, mxu_mapping)
            effective_mxu_shape = self.get_effective_mxu_shape(system.mxu_shape)
            origin_num_iters, compute_efficiency = self.get_compute_efficiency(effective_mxu_shape, mxu_mapping)
            energy_per_power_gated_mxu = system.energy_per_4_128_128_mxu / np.prod(system.power_gating_granularity)
            power_gated_energy = energy_per_power_gated_mxu * pg_num_iters * streaming_dim
            energy = system.energy_per_4_128_128_mxu * origin_num_iters * streaming_dim
        else:
            energy = system.power_per_4_128_128_mxu * self.get_effective_num_ops(system) / (system.op_per_sec)
            power_gated_energy = energy
        return energy, power_gated_energy



    # def get_compute_efficeincy(self, dim_size, mxu_size):
    #     iters = ceil(dim_size/mxu_size)
    #     efficiency = dim_size/ (iters * mxu_size)
    #     return efficiency
    #
    # def get_compute_time(self, system):
    #
    #     if system.mxu_shape is not None:
    #         left, upper, contract, outer = self.get_gemms()
    #         if system.skip_compute:
    #             contract = contract * self.density_w * self.density_a
    #             if contract < 1:
    #                 print(f'[Warning] Contract dimension < 1, after sparsified')
    #             if system.skip_compute_on_noopt_output:
    #                 left = left *self.density_o
    #         mxu_mapping = np.sort([left, upper, contract])[::-1][:2]
    #         effective_mxu_shape = np.sort(system.mxu_shape[-2:])[::-1]
    #         compute_efficiency = np.prod([self.get_compute_efficeincy(d, m) for d, m in zip(mxu_mapping, effective_mxu_shape)])
    #     else:
    #         compute_efficiency = 1.0
    #     return self.get_effective_num_ops(system) * system.get_bit_multiplier(type='C')/system.op_per_sec  / compute_efficiency

    def get_compute_energy(self, system):
        return self.get_effective_num_ops(system)  * system.energy_per_mac



    def get_effective_num_ops(self, system):
        if system.skip_compute:
            if system.skip_compute_on_noopt_output:
                return self.get_num_ops() * self.density_w * self.density_a *self.density_o
            else:
                return self.get_num_ops() * self.density_w * self.density_a
        else:
            return  self.get_num_ops()

    def get_index_bits_estimator(self, density):
        if density < 0.1:
            bits = 4
        elif density < 0.25:
            bits = 3
        elif density == 1:
            bits = 0
        else:
            bits = 2
        return bits


    def get_sz_list(self, system=None, index_mem=False):
        if system:
            if system.compress_mem:
                sz_list = [sz * density for sz, density in zip(self.get_sz_list(), self.get_density_list())]
                if not index_mem:
                    return sz_list
                else:
                    left, upper, contract, outer = self.get_gemms()
                    contract_w = max(1, contract*self.density_w)
                    contract_a = max(1, contract*self.density_a)
                    index_size_w = upper * contract_w * outer  * self.get_index_bits_estimator(self.density_w) / 8 * system.get_bit_multiplier('M')
                    index_size_a = left * contract_a * outer  * self.get_index_bits_estimator(self.density_a) / 8 * system.get_bit_multiplier('M')
                    sz_list[0] += index_size_a
                    sz_list[1] += index_size_w
                    return sz_list

        return list(map(self.get_size, [self.input_a, self.input_w, self.output]))

    def get_loc_list(self):
        return [self.input_a_loc, self.input_w_loc, self.output_loc]

    def get_memory_time(self, system):
        sz_list = self.get_sz_list(system)
        loc_list = self.get_loc_list()
        memory_time = 0
        for tensor_sz, loc in zip(sz_list, loc_list):
            if loc == 'off':
                bw = system.offchip_mem_bw
            elif loc == 'on':
                bw = system.onchip_mem_bw
            else:
                raise ValueError(f'Wrong bw allocation: {loc}.')
            memory_time += tensor_sz * system.get_bit_multiplier(type='M')/bw
        return memory_time

    def get_memory_energy(self, system):
        sz_list = self.get_sz_list(system)
        loc_list = self.get_loc_list()
        memorgy_energy = 0
        for tensor_sz, loc in zip(sz_list, loc_list):
            if loc == 'off':
                energy = system.energy_per_offchip_access
            elif loc == 'on':
                energy = system.energy_per_onchip_access + system.energy_per_offchip_access
            else:
                raise ValueError(f'Wrong bw allocation: {loc}.')
            memorgy_energy += tensor_sz * energy
        return memorgy_energy


    def get_onchip_occupancy(self):
        sz_list = self.get_sz_list()
        loc_list = self.get_loc_list()
        onchip_mem_occupancy = 0
        for tensor_sz, loc in zip(sz_list, loc_list):
            if loc == 'on':
                onchip_mem_occupancy += tensor_sz

        return onchip_mem_occupancy

    def get_roofline(self, system, unit):
        ideal_compute_time = self.get_ideal_compute_time(system=system)
        ideal_complete_offchip_time, ideal_complete_onchip_time = self.get_ideal_memory_time(system=system)
        # x2 for ops -> float ops
        num_ops = self.get_effective_num_ops(system) * 2
        num_data = self.get_effective_num_data(system) * system.get_bit_multiplier(type='M')
        op_intensity = num_ops/num_data

        ideal_exec_time_complete_offchip = max(ideal_compute_time, ideal_complete_offchip_time)
        ideal_exec_time_complete_onchip = max(ideal_compute_time, ideal_complete_onchip_time)

        ideal_thrpt_complete_offchip = num_ops/ideal_exec_time_complete_offchip
        ideal_thrpt_complete_onchip = num_ops/ideal_exec_time_complete_onchip

        compute_time, compute_efficiency = self.get_compute_time(system=system)
        mxu_energy, power_gated_mxu_energy = self.get_mxu_energy(system=system)
        compute_time /= system.compute_efficiency
        compute_efficiency /= system.compute_efficiency
        memory_time = self.get_memory_time(system=system) / system.memory_efficiency
        exec_time = max(compute_time, memory_time)
        thrpt = num_ops/exec_time
        com_to_mem_ratio = compute_time/memory_time
        boundedness = 'C' if com_to_mem_ratio > 1 else 'M'

        input_a_size, input_w_size, output_size = self.get_sz_list(system)

        compute_energy = self.get_compute_energy(system)
        memory_energy = self.get_memory_energy(system)
        total_energy = compute_energy + memory_energy
        saved_energy_rate = (mxu_energy-power_gated_mxu_energy)/mxu_energy
        ret = {
            'Op Type': self.get_op_type(self.dim),
            'Dimension': self.dim[:self.get_effective_dim_len()],
            'Bound': boundedness,
            'C/M ratio': com_to_mem_ratio,
            'Op Intensity': op_intensity,
            f'Latency ({unit.unit_time})': unit.raw_to_unit(exec_time, type='T'),
            f'Cycles': exec_time*system.frequency,
            f'C Effcy': compute_efficiency,
            f'Num ops ({unit.unit_flop})': unit.raw_to_unit(num_ops, type='O'),
            f'Input_a ({unit.unit_mem})': unit.raw_to_unit(input_a_size, type='M'),
            f'Input_w ({unit.unit_mem})': unit.raw_to_unit(input_w_size, type='M'),
            f'Output ({unit.unit_mem})': unit.raw_to_unit(output_size, type='M'),
            f'Total Data ({unit.unit_mem})': unit.raw_to_unit(sum(self.get_sz_list(system)), type='M'),
            f'Throughput ({unit.unit_compute})': unit.raw_to_unit(thrpt, type='C'),
            f'Roofline Throughput offchip ({unit.unit_compute})': unit.raw_to_unit(ideal_thrpt_complete_offchip, type='C'),
            f'Roofline Throughput onchip ({unit.unit_compute})': unit.raw_to_unit(ideal_thrpt_complete_onchip, type='C'),
            f'Compute Cycles': compute_time*system.frequency,
            f'Memory Cycles': memory_time*system.frequency,
            # f'MXU energy (uJ)': mxu_energy *1e6,
            # f'PG-MXU energy (uJ)': power_gated_mxu_energy *1e6,
            # f'Total energy (uJ)': power_gated_mxu_energy*1e6,
            # f'Saved energy (%)': saved_energy_rate * 100,
            # f'Compute energy (mJ)': compute_energy *1e3,
            # f'Mem energy (mJ)': memory_energy *1e3 ,
            # f'Total energy (mJ)': total_energy*1e3,
            # f'Onchip Memory Occupancy ({unit.unit_mem})':  unit.raw_to_unit(self.get_onchip_occupancy(), type='M'),
            # f'Left Onchip Memory ({unit.unit_mem})': unit.raw_to_unit(system.claim_onchip_mem(
            #     self.get_onchip_occupancy()), type='M'),
        }

        return ret










