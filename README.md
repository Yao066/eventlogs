# Seminar Paper 1 eventlogs repository

## Simod

input event log path: `/simod_workspace/resources/event_logs/*`

output event log path: `/simod_workspace/outputs/*`
* BPI Challenge 2019: `/simod_workspace/outputs/20260615_080302_27CA8B16_D20B_4925_8429_26255D39014E`
* Production: `/simod_workspace/20260608_181447_24D690D3_ECDF_42D3_B1BA_9CA0EB114459`

## AgentSimulator

input event log path: `/AgentSimulator/raw_data/*`

output event log path: `/AgentSimulator/simulated_data/*`
* BPI Challenge 2019: `/AgentSimulator/simulated_data/BPIChallenge2019_3WayMatchingEC_processed/`
* Production: `/AgentSimulator/simulated_data/Production_Data_processed/`


## scripts
Delete non-useful columns from the csv file: `/evaluation/preprocess.py`

Compute trace variants and case length: `/evaluation/compute_log_structure_stats.py`

Compute metric values of AgentSimulator's result: `/evaluation/compute_agent_metrics.py`

Compute the average metric values of Simod's result: `/evaluation/evaluation_simod_metrics.py`
