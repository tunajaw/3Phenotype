
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
    - [-] add HPO metric
        deprecated? since it doesn't consider type error loss, just considering time loss (like DTW)
    - [] Think about whole training process


* experiment results
    - unsupervised: `TEDA_nextmark` and `TE_nextmark are trainable (only 1 binary CIF instensity) 
        * valid prec_label MCC: 0.4065/0.3368
    - supervised: only `TEDA_none` is trainable (AE loss)
        * valid pred_label MCC: 0.362

* consider more biomarkers
