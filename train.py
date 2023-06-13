import  time, tqdm, torch

from config import get_config
from model.loss import *
from datasets.perspective_dataset import GIFTPerspectiveDataset
from utils.logger import Logger
from model.descriptor_utils import DescGroupPoolandNorm

from evaluate import EvaluatePlanarScenes 

from model.load_model import load_model

if __name__ == "__main__":
    args = get_config()
    logger = Logger.initialize(args, training=True)

    model = load_model(args)
    pool_and_norm = DescGroupPoolandNorm(args)

    ## Define training dataloader
    dataset = GIFTPerspectiveDataset(args)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    training_breaker = len(dataloader) + args.training_breaker if args.training_breaker <= 0 else args.training_breaker

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.1) 

    start = time.time()

    best_mma = 0
    for epoch in range(args.num_epochs):
        ## training
        model.switch_mode('train')
        model.train()
        iterate = tqdm.tqdm(enumerate(dataloader), total=training_breaker)
        for idx, data in iterate:
            
            desc1 = model(data['image1'].float().cuda(), data['pts1'].float().cuda())
            desc2 = model(data['image2'].float().cuda(), data['pts2'].float().cuda())
            
            ## ground-truth shifting value computation.
            GT_rotation = data['rotation'].cuda()
            GTShift = torch.round( GT_rotation / 360 * args.num_group).to(torch.int8) % args.num_group
            zero_shift = torch.zeros_like(GTShift).cuda()

            desc1 = pool_and_norm.desc_pool_and_norm(desc1, zero_shift)  
            desc2 = pool_and_norm.desc_pool_and_norm(desc2, GTShift)

            ## loss computation.
            ori_loss = orientation_shift_loss(desc1, desc2, GT_rotation, args.num_group)
            desc_loss = info_nce_contrastive_loss(desc1, desc2)

            ori_loss = ori_loss.mean()
            desc_loss = desc_loss.mean()
            
            loss = ori_loss * args.alpha + desc_loss 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            msg = "ep {} Loss: {:.5f}+{:.5f}={:.6f}".format(epoch, ori_loss.detach().cpu().numpy(), \
                        desc_loss.detach().cpu().numpy(), loss.detach().cpu().numpy())
            iterate.set_description(msg)

            if idx == training_breaker:
                break

        model.switch_mode('test')
        model.eval()


        with torch.no_grad():
            ### A. logging hpatches dataset accuracy.
            model.eval()
            args.eval_dataset = 'hpatches_val'
            evaluator = EvaluatePlanarScenes(args)
            result = evaluator(model)
            result.print_results(logger, printing='console')  
            mma_5px = result.get_target_metric()  
            model.train()

        ## B. save model of iteration.
        Logger.save_model(logger, model, epoch)
        logger.info(' {} epoch Model saved.'.format(epoch))
        print("mma : {:.2f}, best_mma: {:.2f}.\n".format(mma_5px, best_mma))

        ## C. best model selection using hpatches real set mma
        if mma_5px > best_mma:
            Logger.save_model(logger, model, "Best")

            best_mma = mma_5px
            best_epoch = epoch


    ## load the best model weights.
    best_model_name =  "{}/{}_model.pt".format(Logger.get_logpath(), best_epoch)

    logger.info("Saving Logpath : {} with best model:\n {} \n".format(Logger.get_logpath(), best_model_name))
    logger.info("===========Best model evaluation in transferred setting =============")

    checkpoint = torch.load(best_model_name)
    model.load_state_dict(checkpoint)
    print("Successfully loaded: {}".format(best_model_name))
    model.cuda()

    ## evaluate test set.
    model.eval()
    args.eval_dataset = 'roto360'
    evaluator = EvaluatePlanarScenes(args)
    a1 = evaluator(model)
    result.print_results(logger, printing='to_file')  

    args.eval_dataset = 'hpatches'
    evaluator = EvaluatePlanarScenes(args)
    result = evaluator(model)
    result.print_results(logger, printing='console')      
    logger.info("=====================================================================")

    end = time.time()
    logger.info("Total training time {:.2f} (sec.)".format(end-start))
