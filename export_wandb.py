import pandas as pd
import wandb

api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("dario_/btc-tsx-graph")
summary_list = []
config_list = []
name_list = []
history_list = []
history_dict = {}
for run in runs:
    # run.summary are the output key/values like accuracy.
    # We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)

    # run.config is the input metrics.
    # We remove special values that start with _.
    config = {k: v for k, v in run.config.items() if not k.startswith("_")}
    config_list.append(config)

    # run.name is the name of the run.
    name_list.append(run.name)
    df_ = pd.DataFrame(run.history())
    df_["name"] = run.name
    history_list.append(df_)


summary_df = pd.DataFrame.from_records(summary_list)
config_df = pd.DataFrame.from_records(config_list)
name_df = pd.DataFrame({"name": name_list})
all_df = pd.concat([name_df, config_df, summary_df], axis=1)

history_df = pd.concat(history_list, sort=False)


for i, df in enumerate(history_list):
    df["name"] = name_list[i]


all_df.set_index(['name', '_step'], inplace=True)
history_df['_step'] = history_df['_step'].astype(int)
history_df.set_index(['name', '_step'], inplace=True)

all_df.to_csv("experiments_summary.csv")
history_df.to_csv("experiments_metrics.csv")