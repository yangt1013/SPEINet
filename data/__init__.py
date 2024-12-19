from importlib import import_module
from torch.utils.data import DataLoader

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

def videoblur_collate_fn(batch):
    max_input_length = max(item[0].shape[0] for item in batch)
    max_gt_length = max(item[1].shape[0] for item in batch)

    padded_inputs = []
    padded_gts = []
    labels = []
    filenames = []
    input_masks = []

    for inputs, gts, label, filename, input_mask in batch:
        pad_input = np.zeros((max_input_length,) + inputs.shape[1:], dtype=inputs.dtype)
        pad_gt = np.zeros((max_gt_length,) + gts.shape[1:], dtype=gts.dtype)

        pad_input[:inputs.shape[0], ...] = inputs
        pad_gt[:gts.shape[0], ...] = gts

        padded_inputs.append(torch.tensor(pad_input))
        padded_gts.append(torch.tensor(pad_gt))
        labels.append(torch.tensor(label))
        filenames.append(filename)
        input_masks.append(torch.tensor(np.pad(input_mask, (0, max_input_length - len(input_mask)), 'constant')))
        # input_masks.append(input_mask)
    return torch.stack(padded_inputs), torch.stack(padded_gts), torch.stack(labels), filenames, torch.stack(input_masks) #torch.tensor(input_masks)
    
class Data:
    def __init__(self, args):
        self.args = args
        self.data_train = args.data_train
        self.data_test = args.data_test

        # load training dataset
        if not self.args.test_only:
            m_train = import_module('data.' + self.data_train.lower())
            trainset = getattr(m_train, self.data_train.upper())(self.args, name=self.data_train, train=True)
            # import pdb
            # pdb.set_trace()
            self.loader_train = DataLoader(
                trainset,
                batch_size=self.args.batch_size,
                shuffle=True,
                pin_memory=not self.args.cpu,
                num_workers=self.args.n_threads,
                #collate_fn=videoblur_collate_fn
            )
        else:
            self.loader_train = None

        # load testing dataset
        m_test = import_module('data.' + self.data_test.lower())
        testset = getattr(m_test, self.data_test.upper())(self.args, name=self.data_test, train=False)
        self.loader_test = DataLoader(
            testset,
            batch_size=1,
            shuffle=False,
            pin_memory=not self.args.cpu,
            num_workers=self.args.n_threads,
            #collate_fn=videoblur_collate_fn
        )
