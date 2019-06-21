from expConfig import *
from model import *
from dataset import *
from metric import *
from setting import *
from utils import config_reader
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--taskid', type=int, default=0, help='the experiment task to run')
args = parser.parse_args()

# parse parameters
t = args.taskid

c_reader = config_reader()

if t == 0:
    # --- Iteration: 1 ---
    config = c_reader.read(t)
    d = MIMIC(config, "ICD9", use_masks=True, use_diffs=False)
    m = RNN(config, d._n_features, d._n_classes)
    e = [HammingLoss(), Jaccard(), Recall(), Precision(), F1(), AUC(), AveragePrecision()]
    p = ExpConfig(dataset=d,
                  model=m,
                  metric=e,
                  config=config,
                  iteration=t%5)
    p.run()
