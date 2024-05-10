Step 1.
* rename datapath
* remove setting args
- TBC


Step 3. (Unsupervised)
* consider `-demo` in `COMMON`
* add unsupervised bash script
* modify `PRE` and `DATA_NAME`
* i_diag=1
* remove setting args in for loop (/wo raindrop)
    SETTING=" -diag_offset $i_diag -data  $PRE/$DATA_NAME/  -split $i_split " 
        

Step 4. (Supervised)
* consider `-demo` in `COMMON`
* add unsupervised bash script
    * modify `PRE` and `DATA_NAME`
    * i_diag=1
    * remove setting args in for loop (/wo raindrop)
        SETTING=" -diag_offset $i_diag -data  $PRE/$DATA_NAME/  -split $i_split " 

* i_split from {0..0}



