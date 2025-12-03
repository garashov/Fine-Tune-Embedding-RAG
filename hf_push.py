from src.core.hf_push_pipeline import PushPipeline



if __name__ == "__main__":
    pipeline = PushPipeline()
    result = pipeline.run()