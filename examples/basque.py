#Import packages
import pandas as pd
import numpy as np
from synth import Synth

#Import data
data = pd.read_csv("examples/datasets/basque_data.csv")
data = data.drop(columns="regionno", axis=1)

#Fit Synthetic Control
synth = Synth(data, "gdpcap", "regionname", "year", 1969, "basque")