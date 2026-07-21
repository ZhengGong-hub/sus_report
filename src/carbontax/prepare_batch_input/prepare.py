"""Run stage 2: PDFs → chunks → batch JSONL. Pick CONFIG_PATH below, then just run this."""

from carbontax.config import load_run_config, stage_section
from carbontax.prepare_batch_input.preparer import BatchInputPreparer
from carbontax.utils.logger import setup_logging

# Which run config this stage reads — switch by commenting/uncommenting.
CONFIG_PATH = "config/run_test_trucost.yaml"
# CONFIG_PATH = "config/run_test_whole_universe.yaml"
# CONFIG_PATH = "config/run.yaml"


def main() -> None:
    setup_logging()
    cfg = load_run_config(CONFIG_PATH)
    BatchInputPreparer(cfg["run_name"], stage_section(cfg, "prepare_batch_input"),
                       cfg["data"]).run()


if __name__ == "__main__":
    main()
