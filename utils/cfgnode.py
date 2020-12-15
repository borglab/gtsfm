from yacs.config import CfgNode as YACS
import ast
import os 
import argparse

class CfgNode:
    """ 
    YACS Configuration Node custimized for gtsfm 
    I keep it as simple as possible for now
    Pls let me know if you want more features with a tag ToDo
    """
    def __init__(self, cfg_init: YACS) -> None:
        """ Initialize the configuration node  
        args:
        cfg_init: default config file
        """
        self.param = cfg_init
        self.param.freeze()

    def print_modules(self):
        """ Print out the configuration that has already loaded  """
        print(self.param)
    
    def load_file(self, file_name:str) -> None:
        """ 
        Load config from .yaml file
        Args:
        file_name: path to yaml file
        """
        self.param.defrost()
        self.param.merge_from_file(file_name)
        self.param.set_new_allowed(True)
        self.param.freeze()

    def load_list(self, list_param:list) -> None:
        """ 
        Load from argparser 
        Args:
        list_param: a parameter list loaded from argsparser with format
            [ module.submodule.hyperparameter1 value 
              module.submodule.hyperparameter2 value
              ...
            ]
        """
        self.param.defrost()
        self.param.merge_from_list(list_param)
        self.param.set_new_allowed(True)
        self.param.freeze()

class ArgsCfgNode:
    """ 
    ArgsCfgNode is small applet to facillitate the interation between argsparse with CfgNode, which
    enables some functionalities:
    
    1. add arguments to parser for the load of config file and parameter list    
    2. load arguments into CfgNode for esier initialization
    """
    def __init__(self, description:str, config:CfgNode=CfgNode(YACS())) -> None:
        """
        Initialization of argsparser and argument to initialize CfgNode
        Args:
        description: description of parser
        config(optional): if the config is initialized, we can add argument from config
        """
        self.parser = argparse.ArgumentParser(
            description=description
        )
        self.parser.add_argument(
            '-cf',
            '--config-file',
            nargs='+',
            type=str,
            help="Path to config files",
        )
        self.parser.add_argument(
            '-cp',
            '--config-param',
            nargs='+',
            type=str,
            help="Config parameter(s): Module1.Submodule1.Param1 Value1 Module2.param2 Value2 ... ",
        )
    
    def init_config(
            self, 
            config:CfgNode, 
            config_file:list = None, 
            config_param:list = None
        ) -> CfgNode:
        """
        Initialization CfgNode using loaded arguments
        Args:
        config: config without initialization
        test: is it for test or not
        Returns:
        config: config after initialization
        """
        if config_file:
            cfg_args = ['--config-file']
            for file in config_file:
                cfg_args.append(file)
            args = self.parser.parse_args(
                cfg_args
            )
            print(args)
        if config_param:
            cfg_args = ['--config-param']
            for param in config_param:
                cfg_args.append(param) 
            args = self.parser.parse_args(
                cfg_args
            )
            print(args)
        if (not config_file) and (not config_param):
            args = self.parser.parse_args()

        if args.config_file:
            for file in args.config_file:
                config.load_file(file)
        if args.config_param:
            config.load_list(args.config_param)
        return config
