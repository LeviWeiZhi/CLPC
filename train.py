import os
import os.path as op
import torch
import numpy as np
import random
import time
from ptflops import get_model_complexity_info

from datasets import build_dataloader
from processor.processor import do_train
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model
from utils.metrics import Evaluator
from utils.options import get_args
from utils.comm import get_rank, synchronize


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    args = get_args()
    set_seed(1+get_rank())
    name = args.name

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    
    device = "cuda"
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.output_dir = op.join(args.output_dir, args.dataset_name, f'{cur_time}_{name}')
    logger = setup_logger('CLPC', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(',', '\n'))
    save_train_configs(args.output_dir, args)

    # get image-text pair datasets dataloader
    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes)
    logger.info('Total params: %2.fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)

    # 可训练参数量
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Trainable params: %.2fM' % (total_trainable_params / 1e6))

    device = "cpu"
    model = model.float()
    model.to(device)

    def build_dummy_batch(args):
        batch_size = 1
        img = torch.randn(batch_size, 3, 192, 256)

        txt = torch.randint(0, args.vocab_size, (batch_size, 77))

        batch = {
            'images': img,
            'caption_ids': txt,
            'pids': torch.randint(0, 11003, (batch_size,)),
            'caption_template': txt
        }
        return batch

    class Wrapper(torch.nn.Module):
        def __init__(self, model, args):
            super().__init__()
            self.model = model
            self.args = args

        def forward(self, x):
            # x 是伪输入，用不到
            batch = build_dummy_batch(self.args)

            # flag 必须传，不然 forward 会报错
            flag = "false"   # 或者你需要的其他值

            return self.model(batch, flag)


    wrapped = Wrapper(model, args)

    # 这里必须传入一个 Tensor 输入形状
    flops, params = get_model_complexity_info(
        wrapped,
        (3, 224, 224),
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False,
    )

    print("FLOPs:", flops)
    print("Params:", params)


    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)

    is_master = get_rank() == 0
    checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
    evaluator = Evaluator(val_img_loader, val_txt_loader)

    start_epoch = 1
    if args.resume:
        checkpoint = checkpointer.resume(args.resume_ckpt_file)
        start_epoch = checkpoint['epoch']

    do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer)