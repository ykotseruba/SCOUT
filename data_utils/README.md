## Dataset utility functions

Replace dataset with dreyeve/bdda/lbw/maad

Print dataset statistics

```
python3 <dataset>_data_utils.py print_dataset_stats
python3 <dataset>_data_utils.py print_context_stats
python3 <dataset>_data_utile.py print_driver_action_stats

```

`vis_dreyeve_data.py` is a script for visualizing DR(eye)VE data.

```
python3 vis_dreyeve_data.py --video_id 70 --gar_frame_range 2289 2334 --show_etg_video --show_gar_video --show_etg_gaze --show_gar_gaze --show_gt --show_old_gt 
```

Use `--show_pred` option to visualize predicted saliency maps (see comments in the script).


