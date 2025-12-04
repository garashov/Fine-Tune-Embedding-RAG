from src.core.ft_pipeline import FineTuningPipeline



if __name__ == "__main__":
    pipeline = FineTuningPipeline()
    metrics = pipeline.run()