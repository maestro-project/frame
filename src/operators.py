import numpy as np
from src.operator_base import Operator

class FC(Operator):
    def __init__(self, dim, density):
        super().__init__(dim=dim, density=density)

    def get_effective_dim_len(self):
        return 3

    def get_tensors(self):
        B, O, I = self.dim[:self.get_effective_dim_len()]
        input_a = (B, I)
        input_w = (O, I)
        output = (B, O)
        return input_a, input_w, output

    def get_gemms(self):
        B, O, I = self.dim[:self.get_effective_dim_len()]
        left = B
        upper = O
        contract = I
        outer = 1
        return left, upper, contract, outer

    def get_num_ops(self):
        B, O, I = self.dim[:self.get_effective_dim_len()]
        return np.prod([B, O, I])


class CONV2D(Operator):
    def __init__(self, dim, density):
        super().__init__(dim=dim, density=density)

    def get_effective_dim_len(self):
        return 7

    def get_tensors(self):
        B, K, C, Y, X, R, S = self.dim[:self.get_effective_dim_len()]
        input_a = (B, C, Y, X)
        input_w = (K, C, R, S)
        output = (B, K, Y, X)

        return input_a, input_w, output

    def get_gemms(self):
        B, K, C, Y, X, R, S = self.dim[:self.get_effective_dim_len()]
        left = B*Y*X
        upper = K
        contract = C*R*S
        outer = 1
        return left, upper, contract, outer


    def get_num_ops(self):
        B, K, C, Y, X, R, S = self.dim[:self.get_effective_dim_len()]
        ofmap_h = Y - R + 1
        ofmap_w = X - S + 1
        return np.prod([B, K, C, ofmap_h, ofmap_w, R, S])

class GEMM(Operator):
    def __init__(self, dim, density):
        super().__init__(dim=dim, density=density)

    def get_effective_dim_len(self):
        return 4

    def get_tensors(self):
        B, M, N, K = self.dim[:self.get_effective_dim_len()]
        input_a = (B, K, N)
        input_w = (M, K)
        output = (B, M, N)
        return input_a, input_w, output

    def get_gemms(self):
        B, M, N, K = self.dim[:self.get_effective_dim_len()]
        left = N
        upper = M
        contract = K
        outer = B
        return left, upper, contract, outer


    def get_num_ops(self):
        B, M, N, K = self.dim[:self.get_effective_dim_len()]
        return np.prod([B, M, N, K])


class Logit(Operator):
    def __init__(self, dim, density):
        super().__init__(dim=dim, density=density)

    def get_effective_dim_len(self):
        return 5

    def get_tensors(self):
        B, H, M, N, D = self.dim[:self.get_effective_dim_len()]
        input_a = (B, H, M, D)
        input_w = (B, H, N, D)
        output = (B, H, M, N)
        return input_a, input_w, output

    def get_gemms(self):
        B, H, M, N, D = self.dim[:self.get_effective_dim_len()]
        left = M
        upper = N
        contract = D
        outer =B*H
        return left, upper, contract, outer


    def get_num_ops(self):
        B, H, M, N, D = self.dim[:self.get_effective_dim_len()]
        return np.prod([B, H, M, N, D])

class Attend(Operator):
    def __init__(self, dim, density):
        super().__init__(dim=dim, density=density)

    def get_effective_dim_len(self):
        return 5

    def get_tensors(self):
        B, H, M, N, D = self.dim[:self.get_effective_dim_len()]
        input_a = (B, H, M, N)
        input_w = (B, H, N, D)
        output = (B, H, M, D)
        return input_a, input_w, output

    def get_gemms(self):
        B, H, M, N, D = self.dim[:self.get_effective_dim_len()]
        left = M
        upper = D
        contract = N
        outer = B*H
        return left, upper, contract, outer

    def get_num_ops(self):
        B, H, M, N, D = self.dim[:self.get_effective_dim_len()]
        return np.prod([B, H, M, N, D])


class DWCONV(Operator):
    def __init__(self, dim, density):
        super().__init__(dim=dim, density=density)

    def get_effective_dim_len(self):
        return 7

    def get_tensors(self):
        B, K, C, Y, X, R, S = self.dim[:self.get_effective_dim_len()]
        input_a = (B, C, Y, X)
        input_w = (C, R, S)
        output = (B, C, Y, X)
        return input_a, input_w, output

    def get_gemms(self):
        B, K, C, Y, X, R, S = self.dim[:self.get_effective_dim_len()]
        left = B*Y*X
        upper = 1
        contract = C*R*S
        outer = 1
        return left, upper, contract, outer

    def get_num_ops(self):
        B, K, C, Y, X, R, S = self.dim[:self.get_effective_dim_len()]
        return np.prod([B, C, Y, X, R, S])