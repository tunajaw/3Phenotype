data_path=./dataset/p12/
wandb_project=TEEDAM_supervised
wandb_tag=RD70
user_prefix=original
supervised_tag=RD75

python Main.py  -data $data_path -setting raindrop -split 0 -demo -data_label multilabel -wandb -wandb_project $wandb_project -event_enc 1 -state -mod ml -next_mark 1 -mark_detach 1 -sample_label 1 -user_prefix $user_prefix -time_enc concat -wandb_tag $wandb_tag > run.log 2>&1
