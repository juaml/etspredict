import os

import pandas as pd
import pkg_resources


def load_struct_connectomes(parcellation="schaefer200x17"):
    """Load group structural connectomes

    Parameters
    ----------
    parcellation : str
        Can be any of the preprocessed parcellations, i.e. schaefer200x17
    """

    path = os.path.join(
        f"hcp_ya/struct_conn/hcpya_struct_conn_{parcellation}.csv"
    )
    stream = pkg_resources.resource_stream(__name__, path)
    df_sc = pd.read_csv(stream, index_col=0)
    df_sc.index = df_sc.index.astype(str)
    df_sc.dropna(inplace=True)

    return df_sc
