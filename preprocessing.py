import pandas as pd
import numpy as np

## Removing rows with values in columns over a certain threshold
df.loc[df['column'] < 0, 'column'] = None #e.g. sets to NaN those values lower than 0
df.loc[df['sleep'] > 3600, 'column'] = None

# Dropping rows with NaNs in all of its columns
df.dropna(how="all", inplace=True)
# Dropping rows with NaNs in ANY of its columns
df.dropna(how="any", inplace=True)

# TODO...