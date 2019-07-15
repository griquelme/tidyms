import ms_feature_validation as mfv

if __name__ == "__main__":
    data = mfv.process.data_container_from_excel("examples/example_sim.xlsx")
    config = mfv.process.read_config("examples/config.yaml")
    pipeline = ms_feature_validation.process.pipeline_from_list(config["Pipeline"])
    pipeline.process(data)