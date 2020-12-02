from utils.cfgnode import CfgNode, ArgsCfgNode

if __name__ == "__main__":
    # Test 1
    config = CfgNode()

    argparser = ArgsCfgNode('Test for config loading and saving')

    config = argparser.init_config(config)
    print(config)


    
