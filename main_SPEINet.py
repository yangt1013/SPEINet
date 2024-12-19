import torch
import data
import model
import loss
import option
from trainer.trainer_swint_hsa_nsf import Trainer_SPEINet
from logger import logger

args = option.args
torch.manual_seed(args.seed)
chkp = logger.Logger(args)
torch.set_float32_matmul_precision('medium')
if args.task == 'VideoDeblur':
    print("Selected task: {}".format(args.task))
    model = model.Model(args, chkp)
    loss = loss.Loss(args, chkp) if not args.test_only else None
    loader = data.Data(args)
    t = Trainer_SPEINet(args, loader, model, loss, chkp)
    while not t.terminate():
        t.train()
        t.test()
else:
    raise NotImplementedError('Task [{:s}] is not found'.format(args.task))

chkp.done()
