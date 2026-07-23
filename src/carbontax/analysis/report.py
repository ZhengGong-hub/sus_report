"""Run stage 4: parsed CSV → review workbook. Reads the run named in analysis.CONFIG_PATH."""

import pandas as pd

from carbontax.analysis import CONFIG_PATH
from carbontax.analysis.reporter import ExcelReporter
from carbontax.config import load_run_config
from carbontax.paths import parsed_csv, report_xlsx


def build_report(input_path: str, dest_path: str) -> None:
    df = pd.read_csv(input_path, dtype={"chunk_ids": "string"})
    print(f"Read {len(df)} rows from {input_path}")
    ExcelReporter(df).write(dest_path)
    print(f"Wrote workbook → {dest_path}")


def main() -> None:
    run_name = load_run_config(CONFIG_PATH)["run_name"]
    build_report(input_path=parsed_csv(run_name), dest_path=report_xlsx(run_name))


if __name__ == "__main__":
    main()
