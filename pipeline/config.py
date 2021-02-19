from pathlib import Path


class Config:
    RANDOM_SEED = 47
    ARTIFACTS_PATH = Path("./artifacts")
    DATASET_PATH = Path("./dataset")
    RAW_DATA = str(DATASET_PATH / "data_raw.txt")
    TEST_SET = str(ARTIFACTS_PATH / "test_set.tsv")
