import math
import torch

class Combiner:
    def __init__(self, model, s_prev, s_cur, cache_len, n = None, mode = 0) -> None:
        self.model = model
        if cache_len == 0 : cache_len = 1
        self.cache = [None] * cache_len
        self.cache_easy = []
        self.cache_len = cache_len
        self.n = math.floor(s_prev/s_cur) if (n == None or n == 0) else n
        self.mode = mode
        if mode == 0: self.dim = 2
        elif mode == 1 : self.dim = 0

    def check_full(self,index):
        check_cache = self.cache[index//self.n * self.n : index//self.n * self.n + self.n]
        for i in range(len(check_cache)):
            if check_cache[i] == None: return False, None
        return True, check_cache

    def combine(self, index, x):
        if x == None:
            return index // self.n, None
        
        self.cache[index] = x
        is_check, check_cache = self.check_full(index)
        if is_check == False:
            return index // self.n, None
        input = torch.cat(check_cache, dim=2)
        index_next = index // self.n
        return index_next, input

    def combine_easy(self, x):
        # only combine, don't care index
        assert(self.n == self.cache_len)
        ret = None
        self.cache_easy.append(x)
        if len(self.cache_easy) == self.cache_len:
            ret = torch.cat(self.cache_easy, self.dim)
            self.cache_easy = []
        return ret


    def combine_and_compute(self, index, x):
        if x == None :
            return index //self.n, None
        
        self.cache[index] = x
        is_check, check_cache = self.check_full(index)
        if is_check == False:
            return index //self.n, None
        input = torch.cat(check_cache, dim=2)
        # self.cache = [None] * self.cache_len
        out = self.model(input)
        index_next = index // self.n
        return index_next, out