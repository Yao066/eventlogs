# Seminar Topic 1 eventlogs repository

## Simod

- input event log path: `/simod_workspace/resources/event_logs/*`

- output event log path: `/simod_workspace/outputs/*`
  -  BPI Challenge 2019: `/simod_workspace/outputs/20260615_080302_27CA8B16_D20B_4925_8429_26255D39014E`
  -  Production: `/simod_workspace/outputs/20260616_100931_8A682B3F_A946_4AE9_96C4_7F6F0A363D84`

## AgentSimulator

- input event log path: `/AgentSimulator/raw_data/*`

- output event log path: `/AgentSimulator/simulated_data/*`
  -  BPI Challenge 2019: `/AgentSimulator/simulated_data/BPIChallenge2019_3WayMatchingEC_processed/`
  -  Production: `/AgentSimulator/simulated_data/Production_Data_processed/`


## scripts
Delete non-useful columns from the csv file: `/evaluation/preprocess.py`

Compute trace variants and case length: `/evaluation/compute_log_structure_stats.py`
 
```powershell
python evaluation/compute_log_structure_stats.py \
  --original-log ../simod_workspace/resources/event_logs/bpi2019_test_preprocessed.csv \
  --simod-dir ../simod_workspace/outputs/20260615_080302_27CA8B16_D20B_4925_8429_26255D39014E/best_result/evaluation \
  --agentsim-dir ../AgentSimulator/simulated_data/BPIChallenge2019_3WayMatchingEC_processed/orchestrated \
  --original-case-col "case_id" \
  --original-activity-col "activity_name" \
  --original-start-col start_timestamp \
  --simod-case-col case_id \
  --simod-activity-col activity \
  --simod-start-col start_time \
  --agentsim-case-col case_id \
  --agentsim-activity-col activity_name \
  --agentsim-start-col start_timestamp
```

Compute metric values of AgentSimulator's result: `/evaluation/compute_agent_metrics.py`
```powershell
python evaluation/compute_agent_metrics.py \
    --ref ../AgentSimulator/simulated_data/BPIChallenge2019_3WayMatchingEC_processed/orchestrated/test_preprocessed.csv \
    --sim ../AgentSimulator/simulated_data/BPIChallenge2019_3WayMatchingEC_processed/orchestrated/ \
    --case-col "case_id" \
    --activity-col "activity_name" \
    --start-col start_timestamp \
    --end-col end_timestamp \
    --resource-col "resource" \
    --sim-case-col case_id \
    --sim-activity-col activity_name \
    --sim-start-col start_timestamp \
    --sim-end-col end_timestamp \
    --sim-resource-col resource
```

Compute the average metric values of Simod's result: `/evaluation/evaluation_simod_metrics.py`
```powershell
python evaluation/evaluation_simod_metrics.py \
  --dir ../simod_workspace/outputs/20260615_080302_27CA8B16_D20B_4925_8429_26255D39014E/best_result/evaluation \
  --input-filename evaluation_metrics.csv \
  --output-filename mean_metrics.csv
```
