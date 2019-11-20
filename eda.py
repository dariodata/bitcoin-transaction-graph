import bz2
import os
import pickle

import pandas as pd
import pandas_profiling

# load data
path = os.path.dirname(os.path.realpath(__file__))
df_edges = pd.read_csv(path + "/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv")
df_classes = pd.read_csv(path + "/elliptic_bitcoin_dataset/elliptic_txs_classes.csv")
df_features = pd.read_csv(
    path + "/elliptic_bitcoin_dataset/elliptic_txs_features.csv", header=None
)

# rename the classes to ints that can be handled by pytorch as labels
df_classes["label"] = df_classes["class"].replace({"unknown": -1, "2": 0}).astype(int)
rename_dict = dict(
    zip(
        range(0, 167),
        ["txId", "time_step"]
        + [f"local_{i:02d}" for i in range(1, 94)]
        + [f"aggr_{i:02d}" for i in range(1, 73)],
    )
)
df_features.rename(columns=rename_dict, inplace=True)
df = df_features.merge(
    df_classes.set_index("txId")[["class"]],
    how="left",
    left_on="txId",
    right_index=True,
)
profile = df.profile_report(title="EDA Bitcoin Transaction Data")
profile.to_file(output_file=path + "/features_report.html")
rejected_variables = profile.get_rejected_variables(threshold=0.9)
description = profile.get_description()

with bz2.BZ2File(path+'/features_profile.pkl.bz2', 'w') as handle:
    pickle.dump(profile, handle)
# with bz2.BZ2File(path+'/features_profile.pkl.bz2', 'r') as handle:
#     p2 = pickle.load(handle)
