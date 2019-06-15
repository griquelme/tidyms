import ms_feature_validation

if __name__ == "__main__":
    data = ms_feature_validation.process.data_container_from_excel("examples/example_sim.xlsx")
    config = ms_feature_validation.process.read_config("examples/config.yaml")
    pipeline = ms_feature_validation.process.pipeline_from_list(config["Pipeline"])
    pipeline.transform(data)
