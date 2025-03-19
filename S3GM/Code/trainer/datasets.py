import numpy as np
from torch.utils.data import Dataset
from einops import rearrange


def get_dataset(config, data, train=True):
    num_samples = config.num_samples_train if train else config.num_samples_val
    if len(data) >= 10:
        ind_train = int(config.train_portion*len(data))
        data = data[:ind_train] if train else data[ind_train:]
    else:
        ind_train = int(config.train_portion*len(data[0]))
        data = data[:, :ind_train] if train else data[:, ind_train:]
    if 'kse' in config.data.lower():
        return DatasetKSE(data, num_samples, config.num_frames, config.num_interval, is_scalar=config.is_scalar)
    elif 'kolmogorov' in config.data.lower():
        return DatasetKolmogorov(data, num_samples, config.num_frames, config.num_interval, is_scalar=config.is_scalar)
    elif 'era5' in config.data.lower():
        return DatasetERA5(data, num_samples, config.num_frames, config.num_interval, is_scalar=config.is_scalar)
    elif 'cylinder' in config.data.lower():
        return DatasetCylinder(data, num_samples, config.num_frames, config.num_interval, is_scalar=config.is_scalar)
    else:
        raise NotImplementedError('Unexpected type of data!')


class DatasetKolmogorov(Dataset):
    def __init__(self, data, length, num_frames, num_interval, is_scalar=True, dtype='float32'):
        super().__init__()
        """data: ndarray with shape b*t*h*w*c"""
        self.length = length
        # general setting
        self.num_b = len(data)
        self.num_t = len(data[0])
        self.num_w = data.shape[-3]
        self.num_h = data.shape[-2]
        self.num_c = data.shape[-1]
        self.num_frames = num_frames
        self.num_interval = num_interval        # list [num_interval for dataset1, num_interval for dataset2, ...]
        _, self.y = np.meshgrid(np.linspace(0, 2 * np.pi, self.num_w, endpoint=False),
                           np.linspace(0, 2 * np.pi, self.num_h, endpoint=False), indexing='ij')
        # the following is for Kolmogorov flow
        u0 = 1.0
        rey = np.linspace(100, 1050, 20, endpoint=True)
        sigma = np.linspace(2, 8, 7, endpoint=True)
        r, s = np.meshgrid(rey, sigma, indexing='ij')
        self.rey = r.reshape(-1)
        self.num_rey = len(self.rey)

        self.vis = u0 / self.rey
        self.f = [1. * np.sin(k * self.y) for k in s.reshape(-1)]

        self.mean = np.mean(data)
        self.std = np.std(data)
        if is_scalar:
            self.data = (data-self.mean)/self.std
        else:
            self.data = data
        self.data = self.data.astype(dtype)

    def __getitem__(self, item):
        i_b = np.random.choice(self.num_b, 1)[0]
        i_t = np.random.choice(self.num_t-self.num_frames*self.num_interval, 1)[0]
        i_w = np.random.choice(self.num_w-self.num_h+1, 1)[0]
        x = self.data[i_b, i_t:i_t+self.num_frames*self.num_interval:self.num_interval, i_w:i_w+self.num_h]
        x = x.transpose(0, 3, 1, 2)
        shift_x, shift_y = np.random.choice(self.num_h, 1)[0], np.random.choice(self.num_w, 1)[0]   # data aug.
        x = np.roll(x, (shift_x, shift_y), axis=(2, 3))     # data aug.
        r = self.vis[i_b%self.num_rey]*100*np.ones_like(self.y)[np.newaxis, np.newaxis].repeat(self.num_frames, 0)
        f = self.f[i_b%self.num_rey][np.newaxis, np.newaxis].repeat(self.num_frames, 0)
        latent_mask = np.ones([self.num_frames, 1, 1, 1]).astype(bool)
        obs_mask = np.zeros([self.num_frames, 1, 1, 1]).astype(bool)
        frame_indices = np.arange(self.num_frames)
        return np.concatenate([x, f, r], axis=1), frame_indices, obs_mask, latent_mask       # B T C H W

    def __len__(self):
        return self.length
    

class DatasetERA5(Dataset):
    def __init__(self, data, length, num_frames, num_interval, is_scalar=True, dtype='float32'):
        super().__init__()
        """data: ndarray with shape b*t*h*w*c"""
        self.length = length
        # general setting
        self.num_b = len(data)
        self.num_t = len(data[0])
        self.num_w = data.shape[-3]
        self.num_h = data.shape[-2]
        self.num_c = data.shape[-1]
        self.num_frames = num_frames
        self.num_interval = num_interval        # list [num_interval for dataset1, num_interval for dataset2, ...]
        _, self.y = np.meshgrid(np.linspace(0, 2 * np.pi, self.num_w, endpoint=False),
                           np.linspace(0, 2 * np.pi, self.num_h, endpoint=False), indexing='ij')

        self.mean = np.mean(data)
        self.std = np.std(data)
        if is_scalar:
            self.data = (data-self.mean)/self.std
        else:
            self.data = data
        self.data = self.data.astype(dtype)

    def __getitem__(self, item):
        i_b = np.random.choice(self.num_b, 1)[0]
        i_t = np.random.choice(self.num_t-self.num_frames*self.num_interval, 1)[0]
        i_w = np.random.choice(self.num_w-self.num_h+1, 1)[0]
        x = self.data[i_b, i_t:i_t+self.num_frames*self.num_interval:self.num_interval, i_w:i_w+self.num_h]
        x = x.transpose(0, 3, 1, 2)
        latent_mask = np.ones([self.num_frames, 1, 1, 1]).astype(bool)
        obs_mask = np.zeros([self.num_frames, 1, 1, 1]).astype(bool)
        frame_indices = np.arange(self.num_frames)
        return x, frame_indices, obs_mask, latent_mask       # B T C H W

    def __len__(self):
        return self.length


class DatasetKSE(Dataset):
    def __init__(self, data, length, num_frames, num_interval, is_scalar=True, dtype='float32'):
        super().__init__()
        """data: ndarray with shape b*t*h*w*c"""
        self.length = length
        # general setting
        self.num_b = len(data)
        self.num_t = len(data[0])
        self.num_w = data.shape[-3]
        self.num_h = data.shape[-2]
        self.num_c = data.shape[-1]
        self.num_frames = num_frames
        self.num_interval = num_interval        # list [num_interval for dataset1, num_interval for dataset2, ...]
        _, self.y = np.meshgrid(np.linspace(0, 2 * np.pi, self.num_w, endpoint=False),
                           np.linspace(0, 2 * np.pi, self.num_h, endpoint=False), indexing='ij')
        # the following is for Kolmogorov flow

        self.mean = np.mean(data)
        self.std = np.std(data)
        if is_scalar:
            self.data = (data-self.mean)/self.std
        else:
            self.data = data
        self.data = self.data.astype(dtype)

    def __getitem__(self, item):
        i_b = np.random.choice(self.num_b, 1)[0]
        i_t = np.random.choice(self.num_t-self.num_frames*self.num_interval, 1)[0]
        i_w = np.random.choice(self.num_w-self.num_h+1, 1)[0]
        x = self.data[i_b, i_t:i_t+self.num_frames*self.num_interval:self.num_interval, i_w:i_w+self.num_h]
        x = x.transpose(0, 3, 1, 2)
        latent_mask = np.ones([self.num_frames, 1, 1, 1]).astype(bool)
        obs_mask = np.zeros([self.num_frames, 1, 1, 1]).astype(bool)
        frame_indices = np.arange(self.num_frames)
        return x, frame_indices, obs_mask, latent_mask       # B T C H W

    def __len__(self):
        return self.length


class DatasetCylinder(Dataset):
    def __init__(self, data, length, num_frames, num_interval, is_scalar=True, dtype='float32'):
        super().__init__()
        """data: list/tuple [dataset1, dataset2, ...], each of the dataset is of shape b*t*h*h*c"""
        data = data
        self.length = length
        # general setting
        self.num_b = len(data)
        self.num_t = len(data[0])
        self.num_w = data.shape[-3]
        self.num_h = data.shape[-2]
        self.num_c = data.shape[-1]
        self.num_frames = num_frames
        self.num_interval = num_interval        # list [num_interval for dataset1, num_interval for dataset2, ...]
        _, self.y = np.meshgrid(np.linspace(0, 2 * np.pi, self.num_w, endpoint=False),
                           np.linspace(0, 2 * np.pi, self.num_h, endpoint=False), indexing='ij')
        # the following is for cylinder flow
        self.vis = np.arange(0.003, 0., -0.0001).astype(dtype)
        self.u0 = 0.1
        # self.f = [np.zeros_like(y) for _ in range(len(self.vis))]
        self.rey = self.u0 / self.vis
        self.num_rey = len(self.rey)
        # the following is for Kolmogorov flow

        self.mean = np.mean(rearrange(data, 'b t h w c -> (b t h w) c'), axis=0)
        self.std = np.std(rearrange(data, 'b t h w c -> (b t h w) c'), axis=0)
        if is_scalar:
            self.data = (data-self.mean[np.newaxis, np.newaxis, np.newaxis, np.newaxis])/self.std[np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        else:
            self.data = data
        self.data = self.data.astype(dtype)

    def __getitem__(self, item):
        i_b = np.random.choice(self.num_b, 1)[0]
        i_t = np.random.choice(self.num_t-self.num_frames*self.num_interval, 1)[0]
        i_w = np.random.choice(self.num_w-self.num_h+1, 1)[0]
        x = self.data[i_b, i_t:i_t+self.num_frames*self.num_interval:self.num_interval, i_w:i_w+self.num_h]
        x = x.transpose(0, 3, 1, 2)
        r =  self.vis[i_b%self.num_rey]*1000*np.ones_like(self.y)[np.newaxis, np.newaxis].repeat(self.num_frames, 0)
        latent_mask = np.ones([self.num_frames, 1, 1, 1]).astype(bool)
        obs_mask = np.zeros([self.num_frames, 1, 1, 1]).astype(bool)
        frame_indices = np.arange(self.num_frames)
        return np.concatenate([x, r], axis=1), frame_indices, obs_mask, latent_mask       # B T C H W

    def __len__(self):
        return self.length
