#Import packages
import pandas as pd
import numpy as np

from SyntheticControlMethods import Synth, DiffSynth

#Get data
data_dir = "https://raw.githubusercontent.com/danilofreire/homicides-sp-synth/master/data/df.csv"
df = pd.read_csv(data_dir)
df.drop(columns=["abbreviation", "code"], axis=1, inplace=True)

sc = Synth(df, "homicide.rates", "state", "year", 1998, "São Paulo", 10, pen=1)

print(sc.original_data.weight_df)
print(sc.original_data.comparison_df)
print(sc.original_data.pen)

#Visualize
sc.plot(["original", "pointwise", "cumulative"], treated_label="São Paulo", 
            synth_label="Synthetic São Paulo", treatment_label="Policy Change")