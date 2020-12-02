import yaml
import logging
import copy
import ast
import enum
import argparse

_YAML_EXTS  = {"yaml", "yml"}
_DATA_TYPES = {tuple, list, str, int, float, bool, type(None)}

class config_input(enum.Enum):
    INPUT_FILE = 1
    INPUT_LIST = 2
    INPUT_DICT = 3
    INPUT_FLATDICT = 4

class CfgNode(dict):
    """ 
    CfgNode is hierachical hyperparameter management data structure designed for small systems, which
    enable the merge of multi-source config hyperparameters into a hierachical CfgNode.

    Compare with its reference [dict], CfgNode enables some favorable properties for config management:
    1. hierarchical structure to better manage hyperparameters from modules/submodules/subsubmodules 
    2. read-only() switch to prevent unexpected modification of global parameters
    3. conversion between CfgNode <-> dict <-> flat dict <-> list to help the easy interaction with argparser 
    """
    # read-only tag
    IMMUTABLE       = '__immutable__' 
    
    def __init__(self, init_dict:dict=None) -> None:
        """ 
        Initialization of CfgNode
        Args:
        init_dict: hierarchical dict for initialization
        """
        init_dict = {} if init_dict is None else init_dict
        super(CfgNode, self).__init__(init_dict)
        
        # read-only() is disabled as default
        self.__dict__[CfgNode.IMMUTABLE] = False

    def __getattr__(self, name: str) -> any:
        """ 
        Redefine the __getattr__ for CfgNode
        Args:
        name: module or submodule or hyperparameter name 
        Returns:
        value: submodule CfgNode or hyperparameter value 
        """
        if name in self:
            return self[name]
        else:
            raise AttributeError(
                "Attempted to get {}, but fail to locate it".format(
                    name
                )
            )
    
    def __setattr__(self, name:str, value:any) -> None:
        """ 
        Redefine the __setattr__ for CfgNode to enable read-only() functionality
        Args:
        name: module or submodule or hyperparameter name
        value: submodule CfgNode or hyperparameter value
        """
        if self.is_read_only():
            raise AttributeError(
                "Attempted to set {} to {}, but Config is read-only".format(
                    value, name
                )
            )
        if not self._valid_type(value):
            raise KeyError(
                "Attempted to set {} to {}, but the type {} is not supported".format(
                    value, name, type(value)
                )
            )
        self[name] = value   
            
    def enable_read_only(self) -> None:
        """ 
        Switch on read-only()
        """
        self.__dict__[self.IMMUTABLE] = True
    
    def disable_read_only(self) -> None:
        """ 
        Switch off read-only()
        """
        self.__dict__[self.IMMUTABLE] = False 
    
    def is_read_only(self) -> bool:
        """ 
        Check the status of read-only()
        Returns:
        is_read_only or not
        """
        return self.__dict__[self.IMMUTABLE]

    def add_new_cfgs(self, inputs:list, option:config_input) -> None:   
        """ 
        Load and merge a list of CfgNodes into the main cfgNode
        Args:
        inputs: a list of supported data
        option: type of input in the list, specified at confit_input
        """
        for input in inputs:     
            cfg = self._cfg_from_input(input, option)
            self._merge_cfg(cfg,self)

    def add_new_cfg(self, input:any, option:config_input) -> None:  
        """ 
        Load and merge a CfgNode into the main cfgNode
        Args:
        inputs: input data
        option: type of the input, specified at confit_input
        """      
        cfg = self._cfg_from_input(input, option)
        self._merge_cfg(cfg,self)

    def _merge_cfg(self, cfg_from:dict, cfg_to:dict) -> None:
        """ 
        Merge a CfgNode into the main cfgNode
        Args:
        cfg_from: a CfgNode to be inserted into main cfgNode
        cfg_to: the main CfgNode
        """  
        for key, value in cfg_from.items():
            if key in self.keys():
                if isinstance(cfg_from[key],CfgNode):
                    self._merge_cfg(value,cfg_to[key])
                else:
                    cfg_to[key] = value
            else:
                cfg_to[key] = value
    
    def _valid_type(self, value:any) -> bool:
        """ 
        Check it the value hold a valid type
        Args:
        value: a value to be checked
        Returns:
        valid or not
        """ 
        return isinstance(value,tuple(_DATA_TYPES)) or isinstance(value,dict)

    def dict_from_cfg(self) -> dict:
        """ 
        Output base CfgNode as a hierarchical dict
        Returns:
        hierachical dict 
        """
        cfg_dict = self._dict_from_cfg(self)
        return cfg_dict

    def flatdict_from_cfg(self) -> dict:
        """ 
        Output base CfgNode as a flat dict
        Returns:
        flat dict 
        """
        cfg_dict = self._dict_from_cfg(self)
        cfg_dict = self._flatdict_from_dict(cfg_dict,"")
        return cfg_dict
    
    @classmethod
    def _cfg_from_input(cls, input:any, option:config_input) -> dict:
        """ 
        Convert differet inputs into a new created CfgNode
        Args:
                    option:                     input: 
            config_input.INPUT_FILE     -> path to yaml file
            config_input.INPUT_LIST     -> list of parameters 
            config_input.INPUT_DICT     -> hierarchical dict
            config_input.INPUT_FLATDICT -> flat dict
               
        Returns:
        a new CfgNode 
        """
        if option == config_input.INPUT_FILE:
            cfg_dict = cls._dict_from_file(input)
        elif option == config_input.INPUT_LIST:
            cfg_dict = cls._dict_from_list(input)
        elif option == config_input.INPUT_FLATDICT:
            cfg_dict = cls._dict_from_flatdict(input)
        elif option == config_input.INPUT_DICT:
            cfg_dict = input
        else:
            raise Exception(
                "Attempted to choose option[{}], but it is not supported. "
                "The valid option is among config_input.[INPUT_FILE, INPUT_LIST, INPUT_DICT, INPUT_FLATDICT].".format(
                    option
                )
            )

        cfg_dict = cls._decode_dict(cfg_dict)
        return cls._cfg_from_dict(cfg_dict)

    @classmethod
    def _cfg_from_dict(cls, cfg_dict:dict) -> dict:
        """ 
        Convert hierachical dict into a new created CfgNode
        Args:
        cfg_dict: hierachical dictionary
        Returns:
        a new CfgNode 
        """
        cfg_dict = copy.deepcopy(cfg_dict)
        for key, value in cfg_dict.items():
            if isinstance(value, dict):
                cfg_dict[key] = cls._cfg_from_dict(value)
        return cls(cfg_dict)

    @classmethod
    def _dict_from_cfg(cls, cfg:dict) -> dict:
        """ 
        Convert CfgNode into a hierachical dict
        Args:
        cfg: CfgNode
        Returns:
        hierarchical dict 
        """
        cfg = copy.deepcopy(cfg)
        for key, value in cfg.items():
            if isinstance(value, CfgNode):
                cfg[key] = cls._dict_from_cfg(value)
        return dict(cfg)

    @classmethod
    def _dict_from_file(cls, filename:str) -> dict:
        """ 
        Convert a (yaml) file into a hierachical dict
        Args:
        filename: the path to the file
        Returns:
        hierarchical dict 
        """
        with open(filename, "r") as f:
            name, ext = f.name.split(".")
            if ext in _YAML_EXTS:
                cfg_dict = yaml.safe_load(f.read())
            else: 
                raise NotImplementedError(
                    "Attempted to load {}, but only yaml is supported for now".format(
                        f.name
                    )
                )
        return cfg_dict
    
    @classmethod
    def _dict_from_list(cls, cfg_list:list) -> dict:
        """ 
        Convert a param list into a hierachical dict
        Args:
        cfg_list: a parameter list with a format
                  [module1.submodule1.hyperparameter1:value1 module2.hyperparameter3:value3 ... ]
        Returns:
        hierarchical dict 
        """
        cfg_dict = {}
        for param in cfg_list:
            full_key, value = param.split(":")
            key_list = full_key.split('.')
            d = cfg_dict
            for key in key_list[:-1]:
                if key not in d.keys():
                    d[key] = {}
                d = d[key]
            d[key_list[-1]] = value
        return cfg_dict

    @classmethod
    def _dict_from_flatdict(cls, flat_dict:list) -> dict:
        """ 
        Convert a flat dict into a hierachical dict
        Args:
        flat_dict: a flat dict with a format
                  {module1.submodule1.hyperparameter1:value1 module2.hyperparameter3:value3 ... }
        Returns:
        hierarchical dict 
        """
        cfg_dict = {}
        for full_key, value in flat_dict.items():            
            key_list = full_key.split('.')
            d = cfg_dict
            for key in key_list[:-1]:
                if key not in d.keys():
                    d[key] = {}
                d = d[key]
            d[key_list[-1]] = value
        return cfg_dict

    @classmethod
    def _flatdict_from_dict(cls, hier_dict:list, hier_key:str) -> dict:
        """ 
        Convert a hierarchical dict into a flat dict
        Args:
        hier_dict: a hierarchical dict
        hier_key: the inhierited key to the dict
        Returns:
        flat dict 
        """
        cfg_dict = {}
        for key, value in hier_dict.items():
            if hier_key:
                full_key = hier_key + '.' + key
            else:
                full_key = key            
            if isinstance(value,dict):
                cfg_dict.update(cls._flatdict_from_dict(value,full_key))
            else:
                cfg_dict[full_key] = value
        return cfg_dict

    @classmethod
    def _decode_dict(cls, cfg_dict:dict) -> dict:
        """ 
        Decode the str values into int, float, bool, and list
        Args:
        dict: a dict
        Returns:
        decoded dict 
        """
        for key, value in cfg_dict.items():
            if isinstance(value, dict):
                cfg_dict[key] = cls._decode_dict(value)
            if isinstance(value, str):
                try:
                    cfg_dict[key] = ast.literal_eval(value)
                except:
                    pass
        return cfg_dict

class ArgsCfgNode:
    """ 
    ArgsCfgNode is small applet to facillitate the interation between argsparse with CfgNode, which
    enables some functionalities:
    
    1. add arguments to parser for the load of config file and parameter list    
    2. load arguments into CfgNode for esier initialization
    """
    def __init__(self, description:str, config:CfgNode=CfgNode()) -> None:
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
            help="Config parameter(s): Module.Submodule:Value",
        )
        cfg_args = config.flatdict_from_cfg()
        for key, value in cfg_args.items():
            self.parser.add_argument(
            '--'+key,
            type=type(value),
            help="",
        )
    
    def init_config(self, config:CfgNode) -> CfgNode:
        """
        Initialization CfgNode using loaded arguments
        Args:
        config: config without initialization
        Returns:
        config: config after initialization
        """
        args = self.parser.parse_args()
        if args.config_file:
            config.add_new_cfgs(args.config_file, config_input.INPUT_FILE)
        if args.config_param:
            config.add_new_cfg(args.config_param, config_input.INPUT_LIST)
        return config

        