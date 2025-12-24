import logging
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
import numpy as np

from sklearn.cluster import DBSCAN
from utils.faiss_rerank import compute_jaccard_distance
from utils.cluster_utils import pseudo_labels_mining
from utils.update_dataLoader import add_pseudo_label, update_train_loader

# Clustering
def cluster_begin_epoch(train_loader, model, args, epoch, logger):
    device = "cuda"

    feature_size = args.embed_dim
    max_size =  ( len(train_loader) )  * args.batch_size

    image_bank = torch.zeros((max_size, feature_size))
    prompt_bank = torch.zeros((max_size, feature_size))

    index = 0

    model.to(device)
    model = model.eval()

    with torch.no_grad():
        for n_iter, batch in enumerate(train_loader):       
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_size = batch['images'].shape[0]   
            i_feats, prompt_feats = model(batch, flag=False)

            image_bank[index: index + batch_size] = i_feats.cpu()
            prompt_bank[index: index + batch_size] = prompt_feats.cpu()
            index = index + batch_size

        image_bank = image_bank[:index]
        prompt_bank = prompt_bank[:index]

        # DBSCAN cluster
        cluster = DBSCAN(eps= 0.6, min_samples=4, metric='precomputed', n_jobs=-1)

        # image clustering
        image_rerank_dist = compute_jaccard_distance(image_bank, k1=30, k2=6, search_option=0)        
        image_pseudo_labels = cluster.fit_predict(image_rerank_dist)    
        del image_rerank_dist
        image_cluster_number1 = len(set(image_pseudo_labels)) - (1 if -1 in image_pseudo_labels else 0)    
        count_uncluster_image1 = np.count_nonzero(image_pseudo_labels == -1)

        # textual prompt clustering
        prompt_rerank_dist  = compute_jaccard_distance(prompt_bank, k1=30, k2=6, search_option=0)
        prompt_pseudo_labels = cluster.fit_predict(prompt_rerank_dist)    
        del prompt_rerank_dist
        prompt_cluster_number = len(set(prompt_pseudo_labels)) - (1 if -1 in prompt_pseudo_labels else 0)    
        count_uncluster_prompt = np.count_nonzero(prompt_pseudo_labels == -1)

        logger.info("==> Statistics for epoch [{}]: image num_clusters:{}".format(epoch, image_cluster_number1))
        logger.info("The number of non-clustering image instances is {}.".format(count_uncluster_image1) )
        logger.info("==> Statistics for epoch [{}]: prompt num_clusters:{}".format(epoch, prompt_cluster_number))
        logger.info("The number of non-clustering prompt instances is {}.".format(count_uncluster_prompt))


    enhanced_image_pseudo_labels = pseudo_labels_mining(image_bank, image_pseudo_labels, prompt_bank, prompt_pseudo_labels)
    count_uncluster_image2 = np.count_nonzero(enhanced_image_pseudo_labels == -1)

    logger.info("After NGPM, the number of non-clustering image instances is {}.".format(count_uncluster_image2) )

    del image_bank, prompt_bank

    return enhanced_image_pseudo_labels



def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,scheduler, checkpointer):
    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("CLPC.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "ndm_loss": AverageMeter(),
        "dmt_loss": AverageMeter(),
        "ccm_loss": AverageMeter(),
        "fcm_loss": AverageMeter()
    }
    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        image_pseudo_labels = cluster_begin_epoch(train_loader, model, args, epoch, logger)

        start_time = time.time()
        for meter in meters.values():
            meter.reset()

        model.train()

        new_train_loader = add_pseudo_label(train_loader, image_pseudo_labels)

        updated_train_loader = update_train_loader(new_train_loader, image_pseudo_labels, args, logger)

        for n_iter, batch in enumerate(updated_train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            image_pseudo_labels = batch['pseudo_label']

            ret = model(batch, True, image_pseudo_labels, epoch) 

            total_loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)

            meters['ndm_loss'].update(ret.get('ndm_loss', 0), batch_size)
            meters['dmt_loss'].update(ret.get('dmt_loss', 0), batch_size)
            meters['ccm_loss'].update(ret.get('ccm_loss', 0), batch_size)
            meters['fcm_loss'].update(ret.get('fcm_loss', 0), batch_size)
   
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(updated_train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)


        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")


def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("CLPC.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())
