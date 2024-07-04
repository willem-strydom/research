import pandas as pd

def record_access(dict, filename):

    df = pd.DataFrame(dict)
    df.to_csv(filename, mode='a', index=False, header=False)