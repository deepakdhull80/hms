import pandas as pd

from common.data import Datasetv1
from config.config import ConfigV1
from models.base_model import Classifier

if __name__ == "__main__":
    
    ################################
    # # Test Dataset
    # df = pd.read_csv(ConfigV1.data.data_prefix + "/" + ConfigV1.data.meta_file_name)
    # ds = Datasetv1(df, ConfigV1)
    # row = next(iter(ds))
    # print(row)
    
    ################################################################
    
    ## Test Model
    import torch
    classifier = Classifier(ConfigV1)
    x = torch.randn(1, ConfigV1.model.conv_in_channels, *ConfigV1.data.image_size)
    y = torch.randn(1, len(ConfigV1.class_columns))
    # return loss
    out2 = classifier(x, y)
    
    classifier.eval()
    # return inference output
    out1 = classifier(x)
    print("out1[infer]: ",out1)
    print("out2[train]:", out2)
    import os
    print(os.path.dirname(os.path.realpath(__file__)))
    
    ################################################################
    
    pass