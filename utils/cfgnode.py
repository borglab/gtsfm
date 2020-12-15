
import argparse
import os
from typing import List, Optional

from yacs.config import CfgNode as YACS


"""
Provides 2 classes for GTSFM users to interface with command-line
input parameters and YAML files. Both classes wrap around YACS.

CfgNode is designed to be used without any command-line input, and
ArgsCfgNode is designed to merge argparse data with YAML data.

Our main requirement is that users will not be allowed to modify
these parameters after they are initialized here, i.e. "frozen".
"""

class CfgNode:
    """Class that reads YAML and freezes parameters (no argparse interface)."""
    def __init__(self, cfg_init: YACS) -> None:
        """ Initialize the configuration node  
        Args:
            cfg_init: default config file
        """
        self.param = cfg_init
        self.param.set_new_allowed(False)
        self.param.freeze()

    def print_modules(self) -> None:
        """ Print out the configuration that has already loaded  """
        print(self.param)
    
    def load_file(self, file_name:str) -> None:
        """ 
        Load config from .yaml file
        Args:
            file_name: path to yaml file
        """
        self.param.defrost()
        self.param.set_new_allowed(True)
        self.param.merge_from_file(file_name)
        self.param.set_new_allowed(False)
        self.param.freeze()

    def load_list(self, list_param: List[str]) -> None:
        """
        Load from argparser. YACS will infer the appropriate types to convert strings
        
        Args:
            list_param: a parameter list loaded from argparser with format
                [ module.submodule.hyperparameter1 value 
                  module.submodule.hyperparameter2 value
                  ...
                ]
        """
        self.param.defrost()
        self.param.merge_from_list(list_param)
        self.param.freeze()

        
class ArgsCfgNode:
    """ Class that merges command-line input parameters with pre-defined parameters
    from a YAML file.
    
    1. add arguments to parser for the load of config file and parameter list    
    2. load arguments into CfgNode for easier initialization
    """
    def __init__(self, description: str, config: CfgNode=CfgNode(YACS())) -> None:
        """
        Initialization of argsparser and argument to initialize CfgNode
        Args:
            description: a string that provides a helpful description of parser
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
            config_file: List[str] = None, 
            config_param: List[str] = None
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
