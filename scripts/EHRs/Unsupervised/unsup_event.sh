#!/bin/bash


# this is added as a prefix to the wandb run name
USER_PREFIX=event_mc_unsup-

# address of data folder
PRE="./dataset"
DATA_NAME="event"

# hyperparameters. Use `python Main.py -h` for more information
COMMON="-demo -data_label multilabel  -epoch 60 -per 100  -label_class 6 -K 7 -cluster 0 -sample_gap 90 -draw_plt 1 -ES_pat 100 -wandb -wandb_project TEEDAM_unsupervised "

TEE_CONFIG_C1="--te_d_mark 8 --te_d_time 4 --te_d_inner 32 --te_d_k 8 --te_d_v 8 --te_n_head 4 --te_n_layers 4 --te_dropout 0.1"

DAM_CONFIG_C2="--dam_output_activation relu --dam_output_dims  16 --dam_n_phi_layers 3  --dam_phi_width 128  --dam_phi_dropout 0.2  --dam_n_psi_layers 2  --dam_psi_width 64  --dam_psi_latent_width 128 --dam_dot_prod_dim 64  --dam_n_heads 4  --dam_attn_dropout 0.1  --dam_latent_width 64  --dam_n_rho_layers 2  --dam_rho_width 128  --dam_rho_dropout 0.1  --dam_max_timescale 1000  --dam_n_positional_dims 16 
" 

OPT_HPs="-batch_size 32  -lr 0.0001 -weight_decay 0.1 -w_pos_label 0.10206294 0.34780188 0.13642752 1.16843034 2.07644497 2.56515121 0.60368114" # simpler2

HPs="$OPT_HPs $TEE_CONFIG_C1 $DAM_CONFIG_C2"



# coefficients for multi-objective loss function
COEFS="-w_sample_label 100  -w_time 1 -w_event 1"


############################################################ Possible TEE loss

# DAM + TEE with AE loss
TEDA__nextmark="-event_enc 1    -state          -mod none      -next_mark 1     -mark_detach 0      -sample_label 2"

# DAM + TEE with PP(single)(equation 4 in the paper)
TEDA__pp_single_mark="-event_enc 1    -state          -mod single    -next_mark 1     -mark_detach 0      -sample_label 2"

# DAM + TEE with PP(ML)(equation 3 in the paper)
TEDA__pp_ml="-event_enc 1    -state          -mod ml        -next_mark 1     -mark_detach 1      -sample_label 2"




# TEE with AE loss
TE__nextmark="-event_enc 1          -mod none      -next_mark 1     -mark_detach 0      -sample_label 2"

# TEE with PP(single)(equation 4 in the paper)
TE__pp_single_mark="-event_enc 1          -mod single    -next_mark 1     -mark_detach 0      -sample_label 2"

# TEE with PP(ML)(equation 3 in the paper)
TE__pp_ml="-event_enc 1          -mod ml        -next_mark 1     -mark_detach 1      -sample_label 2"


i_diag=1


############################################################ EXPERIMENTS

for i_split in {0..0}
do
    SETTING=" -diag_offset $i_diag -data  $PRE/$DATA_NAME/  -split $i_split " 

        echo "split #$i_split" 

        # each of the following lines corresponds to a row in Table 4

        # TE__pp_single_mark
        echo "TE single"
        python Main.py  $HPs $COEFS $SETTING $COMMON $TE__pp_single_mark -user_prefix "[$USER_PREFIX-TE__pp_single_mark-concat-d$i_diag]" -time_enc concat -wandb_tag RD74-single3 > logs/Unsupervised/TE_single.log 2>&1
        # TEDA__pp_single_mark
        echo "TEDA single"
        python Main.py  $HPs $COEFS $SETTING $COMMON $TEDA__pp_single_mark -user_prefix "[$USER_PREFIX-TEDA__pp_single_mark-concat-d$i_diag]" -time_enc concat -wandb_tag RD74-single3 > logs/Unsupervised/TEDA_single.log 2>&1




        # TE__pp_ml
        echo "TE ML"
        python Main.py  $HPs $COEFS $SETTING $COMMON $TE__pp_ml -user_prefix "[$USER_PREFIX-TE__pp_ml-concat-d$i_diag]" -time_enc concat -wandb_tag RD74-ml3 > logs/Unsupervised/TE_ML.log 2>&1
        # TEDA__pp_ml
        echo "TEDA ML"
        python Main.py  $HPs $COEFS $SETTING $COMMON $TEDA__pp_ml -user_prefix "[$USER_PREFIX-TEDA__pp_ml-concat-d$i_diag]" -time_enc concat -wandb_tag RD74-ml3 > logs/Unsupervised/TEDA_ML.log 2>&1




        # TE__nextmark
        echo "TE AE"
        python Main.py  $HPs $COEFS $SETTING $COMMON $TE__nextmark -user_prefix "[$USER_PREFIX-TE__nextmark-concat-d$i_diag]" -time_enc concat -wandb_tag RD74-nextmark3 > logs/Unsupervised/TE_AE.log 2>&1
        # TEDA__nextmark
        echo "TEDA AE"
        python Main.py  $HPs $COEFS $SETTING $COMMON $TEDA__nextmark -user_prefix "[$USER_PREFIX-TEDA__nextmark-concat-d$i_diag]" -time_enc concat -wandb_tag RD74-nextmark3 > logs/Unsupervised/TEDA_AE.log 2>&1



done

