from yacs.config import CfgNode as YACS_CfgNode
import ast
import os 
import argparse

_YAML_EXTS = {"", ".yaml", ".yml"}
_PY_EXTS = {".py"}

class YACS(YACS_CfgNode):
    """ There are some small bugs from yacs I corrected here """
    
    @classmethod
    def _decode_cfg(cls, cfg:YACS_CfgNode) -> YACS_CfgNode:
        """ 
        Decode the str values into int, float, bool, and list
        Args:
        dict: a dict
        Returns:
        decoded dict 
        """
        for key, value in cfg.items():
            if isinstance(value, dict):
                cfg[key] = cls._decode_cfg(value)
            if isinstance(value, str):
                try:
                    cfg[key] = ast.literal_eval(value)
                except:
                    pass
        return cfg
    
    @classmethod
    def _load_cfg_from_file(cls, file_obj):
        """Load a config from a YAML file or a Python source file."""
        _, file_extension = os.path.splitext(file_obj.name)
        if file_extension in _YAML_EXTS:
            cfg = cls._load_cfg_from_yaml_str(file_obj.read())
        elif file_extension in _PY_EXTS:
            cfg = cls._load_cfg_py_source(file_obj.name)
        else:
            raise Exception(
                "Attempt to load from an unsupported file type {}; "
                "only {} are supported".format(file_obj, _YAML_EXTS.union(_PY_EXTS))
            )
        return cls._decode_cfg(cfg)

    pass
    

class CfgNode:
    """ 
    YACS Configuration Node custimized for gtsfm 
    I keep it as simple as possible for now
    Pls let me know if you want more features with a tag ToDo
    """
    def __init__(self) -> None:
        """ Initialize the configuration node  """
        self.param = YACS(new_allowed=True)
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
            help="Config parameter(s): Module1.Submodule1.Param1 Value1 Module2.param2 Value2 ... ",
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
            for file in args.config_file:
                config.load_file(file)
        if args.config_param:
            config.load_list(args.config_param)
        return config
