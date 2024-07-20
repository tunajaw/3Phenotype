* Labeling
    - [] match ctd events and ConfMatrix summation
        - [] Cancel weighted sampling?
    - [-] find out why training epoch in wandb would break in some settings
        - Observation: Just in best epoch
    - [x] multi-class labeling
        - ~~num_types~~
        - CIFs [v]
        - current: only use last predictions as eval. metric. Find out if can directly use all y_labels (yes!)
        - Predcitor output (final layer of Predictor activation) 
    - [] Find out AutoEncoder (AE)
    - [x] CCS/LAB/Static data time alignment
        * Observation: enc_out ([B,L,d_in]) 的時間維度有跟 event_time ([B, L]) 對齊
        - time alignment - similar to label mapping
    - [x] Meaning of Obj - Can the weights of Lcif and Ltask can be adjust?
        * Obj is defined as f1-score - Doesn't imply bp
        * weight can be adjust but still not implemetned
        - [] Maybe consider adding a parameter adjusting weights of loss term?
    - [x] debug labeling on ctd

* experiment results
    - only predict last event (binary)
        - unsupervised: `TEDA_singlemark` and `TE_nextmark` are trainable (only 1 binary CIF instensity) 
            * valid prec_label MCC: 0.4065 / 0.3368
        - supervised: only `TEDA_none` is trainable (AE loss) (>> two types of `TEDA_nextmark`)
            * valid pred_label MCC: 0.362/ 0.086
    - [x] predict all event (binary)
        - Compare results with different settings:
            - tracking_days: (365) / 730
            - cx_pred_days: 90 / 180 / 365
            - Remove pred_time/pred_type loss
            - LR scheduler

* combine t-phennotype clustering
    * Distance Matrix Construction
        - [x] preprocessing (data storage)
            - [x] data loader combination
            - [x] main function at main
            - [x] model.cluster
            - [x] data storage
            - [x] data sampler
                - [] Implicit issue: two samples may be too close ex. | 89 | 1 |
        - [x] sync_clustering
        - [] Do we need to interpolate x_corpus?
        - [] Do we need to conduct warm start of Graph KMeans?
        - [x] predict_proba_from_path
            - [x] add a sigmoid layer at the last
    * Constrained K-Means Clustering
        - [x] graph_kmeans

* data imbalance
    - [] Why predictor always only predict 0 or 2?
        - [] add weight at the predictor
    - [] find of how unsupervised learning train the predictor

* consider more biomarkers

- [] Improved section
        - How to serpeate and combine all time-series data (Maybe by Meta Learning?)
            - Search papers for "Meta Learning for selecting parameters"
        - Concating embedded 
        


- [] Questions
    * Why only choose patients diagnosed MORE THAN 2 times of SADs?
    * Why filtered patients daignosed /w more than 2 kinds of SADs?