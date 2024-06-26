import math
import torch
from utils.spliter import Spliter
from utils.combiner import Combiner

class Spliter_combiner:
    def __init__(self, model, input_size, n_spliter, n_combiner, s_prev, s_cur, cache_len, mode) -> None:
        if (n_combiner == 0) : n_combiner = 1
        if (n_spliter == 0): n_spliter =1


        self.spliter = Spliter(model, input_size, 256, n_spliter, mode)
        self.combiner = Combiner(model, s_prev , s_cur, cache_len, n_combiner, mode)
        self.model = model
        self.split_list = [None] * n_spliter
        self.combiner_list= [None] * (n_spliter // n_combiner)

    def combine_split_list(self):
        i = 0
        for split in self.split_list:
            out = self.combiner.combine_easy(split)
            if out != None:
                self.combiner_list[i] = out
                i=i+1