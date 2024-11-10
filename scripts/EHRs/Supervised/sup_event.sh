#!/bin/bash


# this is added as a prefix to the wandb run name
USER_PREFIX=event_mc_sup-

# address of data folder
PRE="./dataset"
DATA_NAME="event"

# hyperparameters. Use `python Main.py -h` for more information
COMMON=" -demo -data_label multilabel -epoch 140 -per 100 -label_class 6 -K 7 -cluster 0 -sample_gap 90 -draw_plt 0 -ES_pat 100 -wandb -wandb_project TEEDAM_supervised "
HPs="-batch_size 16 -lr 0.0001 -weight_decay 0.1 -w_pos_label 0.01522564 0.36339237 0.21814051 0.94168525 3.02247956 1.97214451 0.46693216"

# sanc : 0.01522564 0.36339237 0.21814051 0.94168525 3.02247956 1.97214451 0.46693216

# inverse: 0.0141619  0.40760237 0.09645988 1.8442039  1.73344863 2.31126484

# beta=0.99995: 0.10206294 0.34780188 0.13642752 1.16843034 2.07644497 2.56515121 0.60368114

# beta=0.9999: 0.21569968 0.472899   0.23350841 1.7127121  1.61663042 2.1180565 0.63049388


# coefficients for multi-objective loss function
COEFS="-w_sample_label 100  -w_time 1 -w_event 1"

# encoder type
ENCODER_ORG="--te_type THP --dam_type SeFT"
ENCODER_M0="--te_type AttNHP --dam_type SeFT"

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

# TEE with AE loss and label loss
TE__pp_ml="-event_enc 1       -mod ml        -next_mark 1     -mark_detach 1      -sample_label 1"

# TEE and label loss(equation 4 in the paper)
TE__pp_single_mark="-event_enc 1      -mod single    -next_mark 1     -mark_detach 0      -sample_label 1"

# DAM + TEE with PP(single) and label loss(equation 4 in the paper)
TEDA__pp_ml_plus="-event_enc 1    -state          -mod ml_plus    -next_mark 1     -mark_detach 0      -sample_label 1"

EXPER="-use_TE_to_decode 1"


# for different splits (raindrop-same splits as raindro's paper)    
for i_split in {0..0}
do
    SETTING=" -data  $PRE/$DATA_NAME/ -split $i_split " 

        echo "first split #$i_split" 

        # # DA__base (DAM in Table 5)
        # echo "DAM" 
        # python Main.py  $HPs $COEFS $SETTING $COMMON $DA__base -user_prefix "[$USER_PREFIX-DA__base-concat]" -time_enc concat -wandb_tag RD75 > logs/Supervised/DA.log 2>&1

        # # TEDA__none (TEE+DAM in Table 5)
        # echo "TEE+DAM"
        # python Main.py  $HPs $COEFS $SETTING $COMMON $TEDA__none -user_prefix "[$USER_PREFIX-TEDA__none-concat]" -time_enc concat -wandb_tag RD75 > logs/Supervised/TEDA.log 2>&1

        # # TEDA__nextmark (TEE+DAM (AE loss) in Table 5)
        # echo "TEE+DAM (AE)"            
        # python Main.py  $HPs $COEFS $SETTING $COMMON $TEDA__nextmark -user_prefix "[$USER_PREFIX-TEDA__nextmark-concat]" -time_enc concat -wandb_tag RD75 > logs/Supervised/TEDA_AE.log 2>&1

        echo "TEE+DAM (MLplus) - TE only"            
        python Main.py  $ENCODER_ORG $EXPER $HPs $COEFS $SETTING $COMMON $TEDA__pp_ml_plus -user_prefix "[$USER_PREFIX-TEDA_mlplus_teOnly]" -time_enc concat -wandb_tag RD75 > logs/Supervised/TEDA_MLplus_TEOnly.log 2>&1

        # TEDA__pp_single_mark (TEE+DAM (single) in Table 5)     
        echo "TEE+DAM (single) - TE only"                   
        python Main.py  $ENCODER_ORG $EXPER $HPs $COEFS $SETTING $COMMON $TEDA__pp_single_mark -user_prefix "[$USER_PREFIX-TEDA_single_teOnly]" -time_enc concat -wandb_tag RD75 > logs/Supervised/TEDA_single_TEOnly.log 2>&1  

        # TEDA__pp_ml (TEE+DAM (ML) in Table 5)  
        echo "TEE+DAM (ML) - TE only"            
        python Main.py  $ENCODER_ORG $EXPER $HPs $COEFS $SETTING $COMMON $TEDA__pp_ml -user_prefix "[$USER_PREFIX-TEDA_ml_teOnly]" -time_enc concat -wandb_tag RD75 > logs/Supervised/TEDA_ML_TEOnly.log 2>&1


        # TEDA__pp_ml_plus (TEE+DAM (ML+single mark) in Table 5)  
        echo "TEE+DAM (MLplus)"            
        python Main.py  $ENCODER_ORG $HPs $COEFS $SETTING $COMMON $TEDA__pp_ml_plus -user_prefix "[$USER_PREFIX-TEDA_mlplus]" -time_enc concat -wandb_tag RD75 > logs/Supervised/TEDA_MLplus.log 2>&1

        # TEDA__pp_single_mark (TEE+DAM (single) in Table 5)     
        echo "TEE+DAM (single)"                   
        python Main.py  $ENCODER_ORG $HPs $COEFS $SETTING $COMMON $TEDA__pp_single_mark -user_prefix "[$USER_PREFIX-TEDA_single]" -time_enc concat -wandb_tag RD75 > logs/Supervised/TEDA_single.log 2>&1  

        # TEDA__pp_ml (TEE+DAM (ML) in Table 5)  
        echo "TEE+DAM (ML)"            
        python Main.py  $ENCODER_ORG $HPs $COEFS $SETTING $COMMON $TEDA__pp_ml -user_prefix "[$USER_PREFIX-TEDA_ml]" -time_enc concat -wandb_tag RD75 > logs/Supervised/TEDA_ML.log 2>&1
      
        ############### M0 #####################

        # TEDA__pp_ml_plus (TEE+DAM (ML+single mark) in Table 5)  
        echo "TEE+DAM (MLplus)"            
        python Main.py  -d_type_emb $ENCODER_M0 $HPs $COEFS $SETTING $COMMON $TEDA__pp_ml_plus -user_prefix "[$USER_PREFIX-TEDA_mlplus]" -time_enc concat -wandb_tag RD75 > logs/Supervised/TEDA_MLplus.log 2>&1

        # TEDA__pp_single_mark (TEE+DAM (single) in Table 5)     
        echo "TEE+DAM (single)"                   
        python Main.py  $ENCODER_M0 $HPs $COEFS $SETTING $COMMON $TEDA__pp_single_mark -user_prefix "[$USER_PREFIX-TEDA_single]" -time_enc concat -wandb_tag RD75 > logs/Supervised/TEDA_single.log 2>&1  

        # TEDA__pp_ml (TEE+DAM (ML) in Table 5)  
        echo "TEE+DAM (ML)"            
        python Main.py  $ENCODER_M0 $HPs $COEFS $SETTING $COMMON $TEDA__pp_ml -user_prefix "[$USER_PREFIX-TEDA_ml]" -time_enc concat -wandb_tag RD75 > logs/Supervised/TEDA_ML.log 2>&1

        echo "TEE+DAM (MLplus) - TE only"            
        python Main.py  $ENCODER_M0 $ENCODER_ORG $EXPER $HPs $COEFS $SETTING $COMMON $TEDA__pp_ml_plus -user_prefix "[$USER_PREFIX-TEDA_mlplus_teOnly]" -time_enc concat -wandb_tag RD75 > logs/Supervised/TEDA_MLplus_TEOnly.log 2>&1

        # TEDA__pp_single_mark (TEE+DAM (single) in Table 5)     
        echo "TEE+DAM (single) - TE only"                   
        python Main.py  $ENCODER_M0 $EXPER $HPs $COEFS $SETTING $COMMON $TEDA__pp_single_mark -user_prefix "[$USER_PREFIX-TEDA_single_teOnly]" -time_enc concat -wandb_tag RD75 > logs/Supervised/TEDA_single_TEOnly.log 2>&1  

        # TEDA__pp_ml (TEE+DAM (ML) in Table 5)  
        echo "TEE+DAM (ML) - TE only"            
        python Main.py  $ENCODER_M0 $EXPER $HPs $COEFS $SETTING $COMMON $TEDA__pp_ml -user_prefix "[$USER_PREFIX-TEDA_ml_teOnly]" -time_enc concat -wandb_tag RD75 > logs/Supervised/TEDA_ML_TEOnly.log 2>&1

done



# Run this only after running unsupervised scripts, we develop the models using ## TRANSFER LEARNING ##


# for i_split in {0..0}
# do
#     SETTING=" -data  $PRE/$DATA_NAME/ -split $i_split " 

#         echo "second split #$i_split" 

#         # TEDA__nextmark ([TEE with AE] + DAM in Table 5)
#         echo "[TEE with AE] + DAM "
#         TL="-transfer_learning -freeze TE -tl_tag RD74-nextmark3"
#         python Main.py  $TL $HPs $COEFS $SETTING $COMMON $TEDA__nextmark -user_prefix "[$USER_PREFIX-TEDA__nextmark-concat]" -time_enc concat -wandb_tag RD75 > logs/Supervised/TEAE_DA.log 2>&1

#         # TEDA__pp_single_mark ([TEE with PP(single)] + DAM in Table 5)
#         echo "[TEE with PP(single)] + DAM "
#         TL="-transfer_learning -freeze TE -tl_tag RD74-single3"
#         python Main.py  $TL $HPs $COEFS $SETTING $COMMON $TEDA__pp_single_mark -user_prefix "[$USER_PREFIX-TEDA__pp_single_mark-concat]" -time_enc concat -wandb_tag RD75 > logs/Supervised/TEsingle_DA.log 2>&1

#         # TEDA__pp_ml ([TEE with PP(ML)] + DAM in Table 5)
#         TL="-transfer_learning -freeze TE -tl_tag RD74-ml3"
#         echo "[TEE with PP(ML)] + DAM "
#         python Main.py  $TL $HPs $COEFS $SETTING $COMMON $TEDA__pp_ml -user_prefix "[$USER_PREFIX-TEDA__pp_ml-concat]" -time_enc concat -wandb_tag RD75 > logs/Supervised/TEML.log 2>&1


# done





