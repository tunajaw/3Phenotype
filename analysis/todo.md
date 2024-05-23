
* Labeling
    - [x] label on exam -> label on ccs (remove mapping)
    - [x] add memo
        -- [x] figure out how memo affects the model - MLP squeeze into `dim_s`, concat to embedded data
    - [x] add MCC metric
    - [x] realize nextmark/single mark/ml in table 4
        * LOSS TYPE for CIFs (CIF_config)
            - nextmark: No event_decoder -> No LL loss
            - single: n_cif = 1
            - ml: n_cif = K
        * ~~opt.num_types = 1 if not 'ml' else 282~~ (not useful) -> num_marks is always 282
    - [x] realize next_pred_type
        if next_time_config:
            self.pred_next_time = Predictor(self.d_con, 1)

        if next_type_config:
            self.pred_next_type = Predictor(
                self.d_con, next_type_config['n_marks'])
            self.mark_detach = next_type_config['mark_detach']
    - [x] realize pred_label
        if label_config:
            self.pred_label = Predictor(self.d_con, 1)
            self.sample_detach = label_config['sample_detach']
    - [x] sample_detach? mark_detach? 
        detach 的部分(loss)不計算 backprop，用於支線下游任務中
        mark detach: ML only, single/AE no detach
        sample detach: unsup only, sup no detach
    - [] find out why training epoch in wandb would break in some settings
    - [] multi-class labeling
        - ~~num_types~~
        - CIFs [v]
        - current: only use last predictions as eval. metric. Find out if can directly use all y_labels (yes!)
        - Predcitor output (final layer of Predictor activation) 
    - [-] add HPO metric
        deprecated? since it doesn't consider type error loss, just considering time loss (like DTW)
        predict next event seems to be not important
    - [x] Think about whole training process
    - [] Find out AutoEncoder (AE)
    - [] Improved section
        - How to serpeate and combine all time-series data (Maybe by Meta Learning?)
        - Concating embedded data

    - [x] Confirm data
        - Draw confusion matrix by wandb [v]
        - Data collection period modification [v]
        


* experiment results
    - only predict last event (binary)
        - unsupervised: `TEDA_singlemark` and `TE_nextmark` are trainable (only 1 binary CIF instensity) 
            * valid prec_label MCC: 0.4065 / 0.3368
        - supervised: only `TEDA_none` is trainable (AE loss) (>> two types of `TEDA_nextmark`)
            * valid pred_label MCC: 0.362/ 0.086
    - predict all event (binary)
        - 

* consider more biomarkers

* report
    - typo: 俞鋒學長的 architecture 中 C = {0, ..., L}
    - Q: motivation of changing encoder, ex. 不用補值, ... etc?
    - Q: PRED 的時機與原因
    - Q: 資料篩選 dataflow