"""Run stage 5b: company-year panel → spec grid of two-way FE regressions."""

from carbontax.config import load_run_config, stage_section
from carbontax.regression.regressor import Regressor
from carbontax.utils.logger import setup_logging

# Which run config this stage reads — switch by commenting/uncommenting.
CONFIG_PATH = "config/run_test_trucost.yaml"
# CONFIG_PATH = "config/run.yaml"


def main() -> None:
    setup_logging()
    cfg = load_run_config(CONFIG_PATH)
    Regressor(cfg["run_name"], stage_section(cfg, "regression")["specs"]).run()


if __name__ == "__main__":
    main()
