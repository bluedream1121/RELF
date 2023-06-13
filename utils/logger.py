r""" Logging """

import datetime
import logging
import os

import torch

import numpy as np
import pandas as pd
from utils.evaluate_utils import compute_matching_results


class Logger:
    r""" Writes results of training/testing """
    @classmethod
    def initialize(cls, args, training):
        logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
        logpath = args.logpath if training else '_TEST_' + args.load_dir.split('/')[-1].split('.')[0] + logtime
        if logpath == '': logpath = logtime
        
        logpath = logpath + '_g' + str(args.num_group)  

        cls.logpath = os.path.join('logs', logpath + '.log')
        # cls.benchmark = args.benchmark
        os.makedirs(cls.logpath)

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)    # every level logs send to handler
        formatter = logging.Formatter('%(message)s')  ## formatter

        # Console log config (also to file)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # file log config
        file_debug_handler = logging.FileHandler(os.path.join(cls.logpath, 'log.txt'))
        file_debug_handler.setLevel(logging.DEBUG)
        file_debug_handler.setFormatter(formatter)
        logger.addHandler(file_debug_handler)

        # Log arguments
        if training:
            logger.info(':======== Rotation-Equivariant Local Descriptors =========')
            for arg_key in args.__dict__:
                logger.info('| %20s: %-24s' % (arg_key, str(args.__dict__[arg_key])))
            logger.info(':========================================================\n')
        logger.info(" Save at {}\n".format(cls.logpath))

        return logger


    @classmethod
    def save_model(cls, logger, model, epoch ):
        torch.save(model.state_dict(), os.path.join(cls.logpath, str(epoch) + '_model.pt'))


    @classmethod
    def get_logpath(cls):
        return cls.logpath

        
class AverageMeterMatching:
    """ rotation_wise matching results logging."""
    def __init__(self):
        r""" Constructor of AverageMeter """
        self.reset()

    @classmethod
    def reset(self):
        self.results = {}
        self.results_dict_init("all")

    @classmethod
    def results_dict_init(self, key):
        self.results[key] = {'num_matches': [], 'distances': [], 'num_points': []}

    @classmethod
    def update(self, angle, num_matches, distances, total_points):
        self.results['all']['num_matches'].append(num_matches)
        self.results['all']['distances'].append(distances)
        self.results['all']['num_points'].append(total_points)

        if angle not in self.results: 
            self.results_dict_init(angle)
        self.results[angle]['num_matches'].append(num_matches)
        self.results[angle]['distances'].append(distances)
        self.results[angle]['num_points'].append(total_points)
    
    @classmethod
    def get_target_metric(self):
        match_cnts, correct_cnts, precisions  = compute_matching_results(self.results['all'], 10 )
        
        return precisions[4]  ## 5px mma

    @classmethod
    def print_results(self, logger, printing='to_file'):
        results_df = self.compute_results()

        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 9000):
            if printing == 'to_file':
                logger.debug(results_df.sort_index(axis=1))   ## only print to log.txt
            elif printing == 'console':
                logger.info(results_df.sort_index(axis=1).iloc[[0,1,4,6,11], :])
            else:
                raise NotImplementedError

        log_results = {
            'correct/mma(3px)': results_df['all'].iloc[4],  ## pixel thres 3 
            'correct/mma(5px)': results_df['all'].iloc[6],  ## pixel thres 5
            'correct/mma(10px)': results_df['all'].iloc[11],  ## pixel thres 10
            'pred_matches': results_df['all'].iloc[1],
            'total_points': results_df['all'].iloc[0],
            }

        # logger.info(log_results)
        # logger.info('')

        match_msg = " ".join(["{}: {}\n".format(k, v ) for k, v in log_results.items()])
        logger.info(" " + match_msg)

        return log_results

    @classmethod
    def compute_results(self, corr_thres=10):
        index = ['total points', 'pred matches'] + [str(idx+1) +'px (correct/mma)' for idx in range(corr_thres)]
        columns = []
        data = []

        for key, values_dict in sorted(self.results.items()):
            ## init columns
            column = "all" if key == "all" else str(key) + " (degree)"
            columns.append(column)  ## avg number of correct match / avg number of pred match  = mma

            ## compute  num_points, pred_matches
            match_cnts, correct_cnts, precisions  = compute_matching_results(values_dict, corr_thres)
            num_points = "{:.1f}".format(np.mean(values_dict['num_points']))
            pred_matches = "{:.1f}".format(match_cnts, 2)
            series = [num_points, pred_matches]

            ## compute correct_cnt, precision
            for idx, (correct_cnt, precision) in enumerate(zip(correct_cnts, precisions)):
                cell =  "{:.1f}/{:.2f}".format(correct_cnt, precision)
                series.append(cell)
            
            data.append(series)
        
        results_df = pd.DataFrame(data=np.array(data).T, index=index, columns=columns)

        return results_df
