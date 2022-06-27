# coding=utf-8
import argparse
import textwrap
import time
import os, sys
sys.path.append(os.path.dirname(__file__))
from utils.config import process_config, check_config_dict
from utils.logger import ExampleLogger
from trainers.example_model import ExampleModel
from trainers.example_trainer import ExampleTrainer
from data_loader.dataset import get_data_loader
import argparse




class ImageClassificationPytorch:
    def __init__(self, config, now, suffix, wandb_mode, test=None):
        gpu_id = config['gpu_id']
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        # check_config_dict(config)
        self.config = config
        self.time = now
        self.suffix = suffix
        self.wandb_mode = wandb_mode
        self.test = test
        self.init()


    def init(self):
        # create net
        
        self.model = ExampleModel(self.config, self.time, suffix=self.suffix)
        # load
        if self.config['pretrained_path'] is not None:

            self.model.load()

        if self.test is not None:
            self.model.load_test()
        
        # create your data generator
        self.train_loader, self.val_loader = get_data_loader(self.config)
        
        # create logger
        self.logger = ExampleLogger(self.config, self.time, suffix=self.suffix)

        self.artifact_dir = self.logger.model_dir

        # create trainer and path all previous components to it
        self.trainer = ExampleTrainer(self.model, self.train_loader, self.val_loader, 
                                      self.config, self.logger, self.time, suffix=self.suffix,
                                      wandb_mode=self.wandb_mode)
    def test_fn(self, positive_auc=False):
        # here you train your model

        self.trainer.test(positive_auc)


    def run(self):
        # here you train your model
        
        self.trainer.train()
        #self.trainer.test()

    def close(self):
        # close
        self.logger.close()


def main(args, now):
    print('args', args)
    config = process_config(os.path.join(os.path.dirname(__file__), args.config))
    # config['num_epochs']=5
    imageClassificationPytorch = ImageClassificationPytorch(config, now, args.suffix, args.wandb_mode)
    imageClassificationPytorch.run()
    imageClassificationPytorch.close()

    return imageClassificationPytorch.artifact_dir.replace('logs', 'save_model')



def test(args, now):
    config = process_config(os.path.join(os.path.dirname(__file__), args.config))
    config['test'] = args.test
    imageClassificationPytorch = ImageClassificationPytorch(config, now, args.suffix, args.wandb_mode, test=args.test)
    imageClassificationPytorch.test_fn(positive_auc=True)
    imageClassificationPytorch.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--suffix', type=str)
    parser.add_argument('--wandb_mode', type=str)
    parser.add_argument('--test', type=None) # test path

    args = parser.parse_args()

    now = time.strftime('%Y-%m-%d | %H:%M:%S', time.localtime(time.time()))

    if args.test is not None:
        print('Testing \n ----------------------------------------------------------------------')
        print('Time: ' + now)
        print('----------------------------------------------------------------------')
        print('                    Now start ...')
        print('----------------------------------------------------------------------')

        test(args, now)
        
    else:

        print('----------------------------------------------------------------------')
        print('Time: ' + now)
        print('----------------------------------------------------------------------')
        print('                    Now start ...')
        print('----------------------------------------------------------------------')


        main(args, now)


    print('----------------------------------------------------------------------')
    print('                      All Done!')
    print('----------------------------------------------------------------------')
    print('Start time: ' + now)
    print('Now time: ' + time.strftime('%Y-%m-%d | %H:%M:%S', time.localtime(time.time())))
    print('----------------------------------------------------------------------')

