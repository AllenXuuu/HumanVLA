import torch


class ExperienceBuffer():
    def __init__(self, info_dict, device, default_dtype = torch.float32) -> None:
        self.device = device
        self.tensor_dict = {}
        for k,info in info_dict.items():
            shape = info['shape']
            dtype = info.get('dtype', default_dtype)
            self.tensor_dict[k] = torch.zeros(
                shape,dtype=dtype,device=device
            )

    def update(self, key, idx, val):
        assert val.dtype == self.tensor_dict[key].dtype, (key, val.dtype, self.tensor_dict[key].dtype)
        assert val.shape == self.tensor_dict[key][idx].shape, (key, val.shape, self.tensor_dict[key].shape)
        self.tensor_dict[key][idx] = val.detach()

    
    def __str__(self) -> str:
        size = sum([v.numel() * v.element_size() / 4 for k,v in self.tensor_dict.items()])
        items = ''
        for k,v in self.tensor_dict.items():
            items += f'\t{k:20s} shape={str(v.shape):30s} dtype={v.dtype}\n'
        return f'Experience Buffer: device={self.device} size={size}@FP32\n{items}'

    def export(self,):
        return self.tensor_dict


class ReplayBuffer():
    def __init__(self, info_dict, device, default_dtype = torch.float32) -> None:
        self.device = device
        self.store_device = torch.device('cpu')
        # self.store_device = device
        self.tensor_dict = {}
        for k,info in info_dict.items():
            shape = info['shape']
            dtype = info.get('dtype', default_dtype)
            self.tensor_dict[k] = torch.zeros(
                shape,dtype=dtype,device=self.store_device
            )
        
        self.head = 0
        self.count = 0
        self.size = info['shape'][0]
        self.sample_idx = torch.randperm(self.size).to(self.store_device)
        self.sample_head = 0
        for k,info in info_dict.items():
            assert self.size == info['shape'][0]

    def store(self, key_val_dict):
        key, val = list(key_val_dict.items())[0]
        new_len = val.shape[0]
        
        assert key_val_dict.keys() == self.tensor_dict.keys()
        for key,val in key_val_dict.items():
            assert val.shape[0] == new_len
            assert val.dtype == self.tensor_dict[key].dtype

        if new_len > self.size:
            rand_idx = torch.randperm(val.shape[0])
            rand_idx = rand_idx[:self.size]
            for key in key_val_dict:
                key_val_dict[key] = key_val_dict[key][rand_idx]
            new_len = self.size

        store_n = min(new_len, self.size - self.head)
        remind_n = new_len - store_n
        for key,val in key_val_dict.items():
            self.tensor_dict[key][self.head : self.head + store_n] = val[ : store_n].to(self.store_device)
            if remind_n > 0:
                self.tensor_dict[key][: remind_n] = val[store_n: ].to(self.store_device)
        self.head = (self.head + new_len) % self.size
        self.count = self.count + new_len
        
    
    def sample(self, n):
        idx = torch.arange(self.sample_head, self.sample_head + n, dtype = torch.long, device = self.store_device)
        idx = idx % self.size
        idx = self.sample_idx[idx]
        if self.count < self.size:
            idx = idx % self.count

        self.sample_head += n
        if self.sample_head > self.size:
            self.sample_idx = torch.randperm(self.size).to(self.store_device)
            self.sample_head = 0
        return {
            key : val[idx].to(self.device)
            for key, val in self.tensor_dict.items()
        }
    


    def __str__(self) -> str:
        size = sum([v.numel() * v.element_size() / 4 for k,v in self.tensor_dict.items()])
        items = ''
        for k,v in self.tensor_dict.items():
            items += f'\t{k:20s} shape={str(v.shape):30s} dtype={v.dtype}\n'
        return f'Replay Buffer: store_device={self.store_device} data_device={self.device} size={size}@FP32\n{items}'



