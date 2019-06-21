from expConfig import ExpConfig
from model import RNN
from dataset import MIMIC
from metric import AUC
from utils import ConfigReader
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--taskid', type=int, default=0, help='the experiment task to run')
args = parser.parse_args()

# parse parameters
t = args.taskid
c_reader = ConfigReader() # This class creates/loads experimental configurations

# Example elements of configuration:
#   - Number of RNN layers
#   - Imputation strategy

if t == 0: # New concept: task ID-specific runs
    config = c_reader.read(t) # Load configuration parameters for dataset/model for task t
    d = MIMIC(config) # Dataset might depend on parameters
    m = RNN(config)
    e = [AUC()]
    p = ExpConfig(dataset=d, # Piece together a dataset, model, and evaluation metric
                  model=m,
                  metric=e,
                  config=config)
    p.run() # Train/test the model

if t == 1:
    config = c_reader.read(t)
    d = MIMIC(config)
    m = RNN(config)
    e = [AUC()]
    p = ExpConfig(dataset=d,
                  model=m,
                  metric=e,
                  config=config)
    p.run()
