
* Labeling
    - [] label on exam -> label on ccs
    - [] multi-class labeling


* [x] Running error checking
    -- TEDA_AE: 
    -- TEDA_single:
    -- TEDA_ML:
    File "/home/tunajaw/TEE4EHR/preprocess/Dataset.py", line 153, in collate_fn
        mod= pad_type(mod) # [B,P]
    File "/home/tunajaw/TEE4EHR/preprocess/Dataset.py", line 111, in pad_type
        if isinstance(insts[0][0],np.ndarray):
    IndexError: list index out of range

    -- TE_ML:
    (solved by reducing batch_size to 64) torch.cuda.OutOfMemoryError: CUDA out of memory.


