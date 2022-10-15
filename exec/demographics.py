import os

import numpy as np
import pkg_resources
from ptpython.ipython import embed

from etspredict.data.hcp_aging_pheno.load import load_hcp_aging_pheno
from etspredict.data.hcp_ya_pheno.get_data import load_behavioural_data


def load_subject_list(dataset, sc=False):

    path = os.path.join("..", "etspredict", "data", f"{dataset}_subjects.txt")
    if sc:
        path = os.path.join(
            "..", "etspredict", "data", f"hcp_subjects_sc_fc.txt"
        )
    stream = pkg_resources.resource_stream(__name__, path)
    return list(np.loadtxt(stream, dtype=str))


def main():

    # hcp ya
    subs = load_subject_list("hcp")
    data = load_behavioural_data(
        subjects=subs, datafields=["Age_in_Yrs", "Gender"]
    )
    embed()
    print("------------------------------------------------------------------")
    print(data.mean())
    print(data.std())
    print(data["Gender"].value_counts())
    print("------------------------------------------------------------------")

    # hcp aging
    subs = load_subject_list("hcp_aging")
    data = load_hcp_aging_pheno("hcp_aging_cogcomp01.txt")[
        ["sex", "interview_age", "src_subject_id"]
    ].drop(labels=0)
    data["interview_age"] = data["interview_age"].astype(int) / 12
    data = data[data["src_subject_id"].isin(subs)]

    print("------------------------------------------------------------------")
    print(data.mean())
    print(data.std())
    print(data["sex"].value_counts())
    print("------------------------------------------------------------------")

    # hcp ya - sc
    subs = load_subject_list("hcp", sc=True)
    data = load_behavioural_data(
        subjects=subs, datafields=["Age_in_Yrs", "Gender"]
    )

    print("------------------------------------------------------------------")
    print(data.mean())
    print(data.std())
    print(data.min(), data.max())
    print(data["Gender"].value_counts())
    print("------------------------------------------------------------------")


if __name__ == "__main__":
    main()
