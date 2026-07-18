"""Run stage 2: PDFs → chunks → batch JSONL. Configure in config/run.yaml, then just run this."""

from carbontax.config import load_run_config, stage_section
from carbontax.prepare_batch_input.preparer import BatchInputPreparer
from carbontax.utils.logger import setup_logging


def main() -> None:
    setup_logging()
    cfg = load_run_config()
    BatchInputPreparer(cfg["run_name"], stage_section(cfg, "prepare_batch_input")).run()


if __name__ == "__main__":
    main()
