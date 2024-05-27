* Labeling
    5/20
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
    
    - [-] add HPO metric
        deprecated? since it doesn't consider type error loss, just considering time loss (like DTW)
        predict next event seems to be not important
    - [x] Think about whole training process
    - [x] Confirm data
        - Draw confusion matrix by wandb [v]
        - Data collection period modification [v]