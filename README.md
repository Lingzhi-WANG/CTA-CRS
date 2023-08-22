# CTA-CRS

This is the readme to run the code of the paper "[Improving Conversational Recommender System via Contextual and Time-Aware Modeling with Less Domain-Specific Knowledge](https://arxiv.org/abs/2209.11386)". The code is based on FAIRSEQ framework.  

The running enviroment.
```
    conda create -n <environment-name> --file req.txt
    pip install --editable ./
```

Download datasets and BART base model.
    Download ReDial from https://redialdata.github.io/website/ 
    Download OpenDialKG from https://github.com/facebookresearch/opendialkg/tree/main/data
    (Then transfer the format of OpenDialKG into the same format as ReDial.)

    Download BART base model from https://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz

Data preprocessing.
    Binarize dataset:
    
  ```
    fairseq-preprocess \
      --source-lang "source" \
      --target-lang "target" \
      --trainpref convrec_process/train.bpe \
      --validpref convrec_process/valid.bpe \
      --destdir convrec-bin/ \
      --workers 60 \
      --srcdict convrec_process/dict.txt \
      --tgtdict convrec_process/dict.txt
  ```


Commmand Line example to train our models. 
```
    CUDA_DEVICE=0
    TOTAL_NUM_UPDATES=20000
    WARMUP_UPDATES=500
    LR=3e-03
    DROP=0.1
    WEIGHT=0.01
    MAX_TOKENS=4096
    CLIPNORM=0.1
    UPDATE_FREQ=8
    PATIENCE=3
    TIME_WEIGHT=1.5
    GR=0.5
    GLR=0
    RR=1.0
    BRR=1.0

    FT_PATH=[PATH1]/checkpoint_best.pt  # We first finetune BART-base model only on response generation with the dataset. [PATH1] is the saved path.
    MODEL_DIR=[PATH2]  # [PATH2] is the path to save the final checkpoint
    DATA_PATH=convrec-bin  # convrec-bin stores the binarized dataset in step (3)
    mkdir -p $MODEL_DIR

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python fairseq_cli/train.py $DATA_PATH \
        --load-finetuned-bart-from $FT_PATH \
        --max-tokens $MAX_TOKENS \
        --save-dir $MODEL_DIR \
        --task crs \
        --source-lang source --target-lang target \
        --layernorm-embedding \
        --share-all-embeddings \
        --share-decoder-input-output-embed \
        --reset-optimizer --reset-dataloader --reset-meters \
        --required-batch-size-multiple 1 \
        --arch convrec_time_redial[OR convrec_time_opendial] \
        --with-recommender \
        --criterion crs_time_loss \
        --conv-lr $GLR \
        --report-recall \
        --best-checkpoint-metric rec --maximize-best-checkpoint-metric \
        --generate-ratio $GR \
        --recommend-ratio $RR \
        --bart-recommend-ratio $BRR \
        --label-smoothing 0.1 \
        --use-text-entities \
        --time-weight $TIME_WEIGHT \
        --dropout $DROP --attention-dropout $DROP \
        --weight-decay $WEIGHT --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
        --clip-norm $CLIPNORM \
        --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
        --update-freq $UPDATE_FREQ \
        --keep-best-checkpoints 1 \
        --no-epoch-checkpoints \
        --patience $PATIENCE \
        --skip-invalid-size-inputs-valid-test \
        --find-unused-parameters | tee $MODEL_DIR/logging_history.txt
```
Commmand Line example to evaluate our models. 
```
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python fairseq_cli/bart_recommendation.py \
         $DATA_PATH \
         --task crs \
         --gen-subset test \
         --ignore-movies-not-in-kg \
         --max-tokens $MAX_TOKENS \
         --skip-invalid-size-inputs-valid-test \
         --path $MODEL_DIR/checkpoint_best.pt | tee $MODEL_DIR/rec_his.txt
```



