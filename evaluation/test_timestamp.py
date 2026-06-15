import pandas as pd

df = pd.read_csv("../AgentSimulator/simulated_data/BPIChallenge2019_3WayMatchingEC_processed/orchestrated/simulated_log_0.csv")

start_col = "start_timestamp"
end_col = "end_timestamp"

df[start_col] = pd.to_datetime(df[start_col], utc=True, format="mixed")
df[end_col] = pd.to_datetime(df[end_col], utc=True, format="mixed")

bad = df[df[end_col] < df[start_col]].copy()
bad["duration_seconds"] = (bad[end_col] - bad[start_col]).dt.total_seconds()

print("negative duration rows:", len(bad))
print(bad[[start_col, end_col, "duration_seconds"]].head(20))