from torch.utils.data import DataLoader

import dataset
from custom_file_io import *
from meta_data_holder import MetaDataHolder
from utils import seed_worker

try:
    from functools import cached_property
except ImportError:
    from cached_property import cached_property


class DataProvider:
    def __init__(self, setting):
        self.setting = setting

    @cached_property
    def swer(self):
        return SubWordEmbReaderUtil(self.setting.subWordEmb_path)

    @cached_property
    def meta_holder(self):
        return MetaDataHolder(*make_metadata_from_setting(self.setting, self.swer))

    def get_dataset(self, mode):
        dialogs, data_coordi, data_rank, data_dialog = make_raw_io_data(self.setting, self.meta_holder, mode)

        if self.setting.dataset == 'OneDataset':
            ds = dataset.OneDataset(data_dialog, data_coordi, data_rank, self, mode, dialogs=dialogs)
        else:
            raise NotImplementedError()

        return ds

    def get_dataloader(self, mode):
        ds = self.get_dataset(mode)

        g = torch.Generator()
        g.manual_seed(self.setting.seed)

        args = {
            'num_workers': self.setting.num_workers,
            'batch_size': self.setting.batch_size,
            'collate_fn': ds.collate_fn,
            'worker_init_fn': seed_worker,
            'generator': g,
        }
        if mode == 'train':
            sep_args = {
                'shuffle': True,
            }
        else:
            sep_args = {
                'shuffle': False,
            }

        args.update(sep_args)
        dl = DataLoader(ds, **args)

        return dl
