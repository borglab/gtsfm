from utils.cfgnode import CfgNode, ArgsCfgNode

""" Pls don't review it for now, I will make it a unittest soon """
if __name__ == "__main__":
    # Test 2
    config = CfgNode()
    config.load_file('config1.yaml')

    argparser = ArgsCfgNode('Test for config loading and saving',config)
    config = argparser.init_config(config)
    config.print_modules()

    
