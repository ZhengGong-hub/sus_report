"""Run stage 5a: parsed chunk flags + filing metadata → company-year panel."""

from carbontax.config import load_run_config, stage_section
from carbontax.regression.panel_builder import PanelBuilder
from carbontax.utils.logger import setup_logging

# Which run config this stage reads — switch by commenting/uncommenting.
CONFIG_PATH = "config/run_test_trucost.yaml"
# CONFIG_PATH = "config/run.yaml"


def main() -> None:
    setup_logging()
    cfg = load_run_config(CONFIG_PATH)
    PanelBuilder(cfg["run_name"], stage_section(cfg, "regression")["panel"], cfg["data"]).run()


if __name__ == "__main__":
    main()
