from dataclasses import dataclass

from omegaconf import MISSING

@dataclass
class SubModule1:
    param_str: str = MISSING
    param_int: int = MISSING
    param_float: float = MISSING 
    param_bool: bool = MISSING

@dataclass
class SubModule2:
    param_str: str = MISSING
    param_int: int = MISSING
    param_float: float = MISSING 
    param_bool: bool = MISSING

@dataclass
class SubModule3:
    param_str: str = MISSING
    param_int: int = MISSING
    param_float: float = MISSING 
    param_bool: bool = MISSING

@dataclass
class SubModule4:
    param_str: str = MISSING
    param_int: int = MISSING
    param_float: float = MISSING 
    param_bool: bool = MISSING

@dataclass
class SubModule5:
    param_str: str = MISSING
    param_int: int = MISSING
    param_float: float = MISSING 
    param_bool: bool = MISSING

@dataclass
class SubModule6:
    param_str: str = MISSING
    param_int: int = MISSING
    param_float: float = MISSING 
    param_bool: bool = MISSING

@dataclass
class SceneOptimizer:
    param_str: str = MISSING
    param_int: int = MISSING
    param_float: float = MISSING 
    param_bool: bool = MISSING
  
@dataclass
class FeatureExtractor:
    submodule1: SubModule1 = SubModule1()
    submodule2: SubModule2 = SubModule2()

@dataclass
class TwoViewEstimator:
    submodule1: SubModule3 = SubModule3()
    submodule2: SubModule4 = SubModule4()

@dataclass
class MultiViewOptimizer:
    submodule1: SubModule5 = SubModule5()
    submodule2: SubModule6 = SubModule6()
