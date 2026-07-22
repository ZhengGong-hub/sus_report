"""Upload + submit the batch to OpenAI. Configure in config/run.yaml, then just run this."""

from carbontax.config import load_run_config
from carbontax.openai_batch.batch_job import OpenAIBatchJob
from carbontax.utils.logger import setup_logging

CONFIG_PATH = "config/run_test_trucost.yaml"


def main() -> None:
    setup_logging()
    OpenAIBatchJob(load_run_config(CONFIG_PATH)["run_name"]).submit()


if __name__ == "__main__":
    main()
