import ms_feature_validation

if __name__ == "__main__":
    data = ms_feature_validation.processor.data_container_from_excel("data/example_sim.xlsx")
    # blank_corrector = ms_feature_validation.processor.BlankCorrector(["SV"])
    # prevalence_filter = ms_feature_validation.processor.PrevalenceFilter(["healthy", "disease"], 0.8)
    # variation_filter = ms_feature_validation.processor.VariationFilter()
    # pipeline = ms_feature_validation.processor.Pipeline(blank_corrector, prevalence_filter)
    # pipeline.transform(data)
    # # variation_filter.transform(data)
    # a = data.DataMatrix
    # a.groupby
