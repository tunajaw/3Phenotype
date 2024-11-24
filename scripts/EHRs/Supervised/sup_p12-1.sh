#!/bin/bash


# this is added as a prefix to the wandb run name
USER_PREFIX=H70-

# address of data folder
PRE="./dataset"
DATA_NAME="p12"

# hyperparameters. Use `python Main.py -h` for more information
COMMON=" -demo -data_label multilabel  -epoch 140 -label_class 1 -per 100 -ES_pat 100 -wandb -wandb_project TEEDAM_supervised "
HPs="-batch_size 16 -lr 0.001 -weight_decay 0.1 -w_pos_label 0.5 1.5"


# coefficients for multi-objective loss function
COEFS="-w_sample_label 100  -w_time 1 -w_event 1"

# encoder type
ENCODER_ORG="--te_type THP --dam_type SeFT"
ENCODER_M0="--te_type AttNHP --dam_type SeFT"
ENCODER_M1="--te_type AttNHP --dam_type mTAN"


############################################################ Possible TEE loss
# DAM + TEE with AE loss and label loss
TEDA__nextmark="-event_enc 1    -state          -mod none      -next_mark 1     -mark_detach 0      -sample_label 1"

# DAM + TEE with PP(single) and label loss(equation 4 in the paper)
TEDA__pp_single_mark="-event_enc 1    -state          -mod single    -next_mark 1     -mark_detach 0      -sample_label 1"

# DAM + TEE with PP(ML) and label loss(equation 3 in the paper)
TEDA__pp_ml="-event_enc 1    -state          -mod ml        -next_mark 1     -mark_detach 1      -sample_label 1"

# DAM + TEE with label loss only
TEDA__none="-event_enc 1    -state          -mod none        -next_mark 1     -mark_detach 1      -sample_label 1"

# Only DAM with label loss
DA__base="-event_enc 0    -state          -mod none      -next_mark 1     -mark_detach 1      -sample_label 1"

# DAM + TEE with PP(single) and label loss(equation 4 in the paper)
TEDA__pp_ml_plus="-event_enc 1    -state          -mod ml_plus    -next_mark 1     -mark_detach 0      -sample_label 1"

# for different splits (raindrop-same splits as raindro's paper)    
for i_split in {0..0}
do
    SETTING=" -data  $PRE/$DATA_NAME/ -split $i_split --te_d_mark 16 --dam_output_dims 32" 

        echo "TEE+DAM (MLplus)"            
        python Main.py  -seed 88888 $ENCODER_M0 $HPs $COEFS $SETTING $COMMON $TEDA__pp_ml_plus -user_prefix "AttNHP(256)/mTAN(64)" -time_enc concat -wandb_tag RD75 > logs/Supervised/debug_pda1.log 2>&1


        # # DA__base (DAM in Table 5)
        # echo python Main.py  $HPs $COEFS $SETTING $COMMON $DA__base -user_prefix "[$USER_PREFIX-DA__base-concat]" -time_enc concat -wandb_tag RD75    

        # # TEDA__none (TEE+DAM in Table 5)
        # echo python Main.py  $HPs $COEFS $SETTING $COMMON $TEDA__none -user_prefix "[$USER_PREFIX-TEDA__none-concat]" -time_enc concat -wandb_tag RD75 

        # # TEDA__nextmark (TEE+DAM (AE loss) in Table 5)            
        # echo python Main.py  $HPs $COEFS $SETTING $COMMON $TEDA__nextmark -user_prefix "[$USER_PREFIX-TEDA__nextmark-concat]" -time_enc concat -wandb_tag RD75

        # # TEDA__pp_single_mark (TEE+DAM (single) in Table 5)                        
        # echo python Main.py  $HPs $COEFS $SETTING $COMMON $TEDA__pp_single_mark -user_prefix "[$USER_PREFIX-TEDA__pp_single_mark-concat]" -time_enc concat -wandb_tag RD75    

        # # TEDA__pp_ml (TEE+DAM (ML) in Table 5)            
        # echo python Main.py  $HPs $COEFS $SETTING $COMMON $TEDA__pp_ml -user_prefix "[$USER_PREFIX-TEDA__pp_ml-concat]" -time_enc concat -wandb_tag RD75 


done





