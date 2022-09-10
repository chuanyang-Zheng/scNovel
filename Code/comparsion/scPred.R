library("scPred")
library("Seurat")
library("magrittr")

reference <- scPred::pbmc_1
query <- scPred::pbmc_2
reference <- reference %>%
  NormalizeData() %>%
  FindVariableFeatures() %>%
  ScaleData() %>%
  RunPCA() %>%
  RunUMAP(dims = 1:30)

reference <- getFeatureSpace(reference, "cell_type")
reference <- trainModel(reference)
get_probabilities(reference) %>% head()

query <- NormalizeData(query)