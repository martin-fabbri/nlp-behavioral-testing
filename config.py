from pathlib import Path

class Config:
    ARTIFACTS_PATH = Path("./artifacts")
    RAW_DATA = str(ARTIFACTS_PATH / "data_raw.csv")