#%%
import time
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

combined_og = pd.read_pickle('/mnt/data2/david/data/c_02/s_02/e4/filtered_data_with_ibi_presence.pkl')
print(combined_og.columns)


