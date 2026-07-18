"""Download the finished batch output. Configure in config/run.yaml, then just run this."""

from carbontax.config import load_run_config
from carbontax.openai_batch.batch_job import OpenAIBatchJob
from carbontax.utils.logger import setup_logging


def main() -> None:
    setup_logging()
    OpenAIBatchJob(load_run_config()["run_name"]).download()


if __name__ == "__main__":
    main()
