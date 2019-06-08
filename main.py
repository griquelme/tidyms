import ms_feature_validation

if __name__ == "__main__":
    data = ms_feature_validation.process.data_container_from_excel("examples/example_sim.xlsx")
    blank_corrector = ms_feature_validation.process.BlankCorrector(["SV"])
    prevalence_filter = ms_feature_validation.process.PrevalenceFilter(["healthy", "disease"], 0.8)
    variation_filter = ms_feature_validation.process.VariationFilter()
    pipeline = ms_feature_validation.process.Pipeline(blank_corrector, prevalence_filter)
    pipeline.transform(data)
