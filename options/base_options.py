import argparse
import os

import models


class BaseOptions():

    def __init__(self, cmd_line=None):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        self.cmd_line = None
        if cmd_line is not None:
            self.cmd_line = cmd_line.split()

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        
        ## basic parameters
        parser.add_argument('--name', type=str, default='new_color', 
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='3', 
                            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        ## model parameters
        parser.add_argument('--model', type=str, default='cut', 
                            help='chooses which model to use.')
        parser.add_argument('--numbel', type=int, default=3, 
                            help='number of layers in the model')

        ## save parameters
        parser.add_argument('--save_dir', type=str, default='./checkpoints', 
                            help='models are saved here')
        
        ## display parameters
        parser.add_argument('--display_freq', type=int, default=400, 
                            help='frequency of showing training results on screen')

        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        if self.cmd_line is None:
            opt, _ = parser.parse_known_args([])
        else:
            opt, _ = parser.parse_known_args(self.cmd_line)

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser)
        if self.cmd_line is None:
            opt, _ = parser.parse_known_args([])  # parse again with new defaults     #这里通过不同模型得到了新的参数，于是要再解析进入opt里面
        else:
            opt, _ = parser.parse_known_args(self.cmd_line)  # parse again with new defaults


        # save and return the parser
        self.parser = parser
        if self.cmd_line is None:
            return parser.parse_args([])
        else:
            return parser.parse_args(self.cmd_line)   #将参数全部返回，但是实际上参数已经保存在opt结构中了。

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        try:
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')
        except PermissionError as error:
            print("permission error {}".format(error))
            pass

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)           #调用打印参数函数


        self.opt = opt
        return self.opt
