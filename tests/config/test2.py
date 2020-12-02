from utils.cfgnode import CfgNode, ArgsCfgNode, config_input

if __name__ == "__main__":
    # Test 2
    config = CfgNode()
    config.add_new_cfg('config1.yaml',config_input.INPUT_FILE)

    argparser = ArgsCfgNode('Test for config loading and saving',config)
    config = argparser.init_config(config)
    print(config)
    

    
