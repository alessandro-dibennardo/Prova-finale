We provide our code of dynamic categorization and scripts for data format transformation (BASE-to-BIO, BASE-to-SWIFT) in this repo.

### Dynamic dataset pipeline
- `categorization/dynamic1.py` – `dynamic4.py`: individual stages used to build the dynamic classify sets.
- `categorization/main.py`: reference pipeline (adjust file paths when re-running).
- `categorization/metrics.py` & `evaluate.py`: compute distribution statistics and sanity checks for any JSON file.
- `check_dynamic_classify.py`: verifies that each assistant answer appears in the option list.
- `prune_dynamic_classify.py`: removes invalid samples reported by the checker.
- `sync_extract.py`: copies the SWIFT extract files from `base/` into `dynamic/extract/`.

### Format conversion helpers
- `transformation/stage1_trans.py` and `stage2_trans.py`: convert datasets from BASE format into the SWIFT prompts used for extraction and classification.
- `BIO_trans.py` / `BIO_trans_zh.py`: export BASE data into token-level BIO files for evaluation with sequence-labeling toolkits.
