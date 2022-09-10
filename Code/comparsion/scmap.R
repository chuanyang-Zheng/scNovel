library(SingleCellExperiment)
library(scmap)
library(pROC)
library(readr)

#scmap websit: 1) code https://github.com/hemberg-lab/scmap   2) introduction of how to use scmap: http://bioconductor.org/packages/release/bioc/vignettes/scmap/inst/doc/scmap.html#scmap-cell
#scID: code https://github.com/BatadaLab/scID
#scPred: https://github.com/powellgenomicslab/scPred
#scLearn: https://github.com/bm2-lab/scLearn

process <- function(path){
  #train data may not be necessary
  # train_typical_data<-read_csv(file.path(path,"train_data.csv"))
  # train_typical_label<-read_csv(file.path(path,"train_label.csv"))

  #read test data
  test_typical_data<-read_csv(file.path(path,"test_data.csv"))
  test_typical_label<-read_csv(file.path(path,"test_label.csv"))
  test_ood_data<-read_csv(file.path(path,"test_ood_data.csv"))
  test_ood_label<-read_csv(file.path(path,"test_ood_label.csv"))

  #check data loading
  if (nrow(test_typical_data)!=nrow(test_typical_label)) or (nrow(test_ood_data)!=nrow(test_ood_label)):
    print("Error!")


  #build ground truth for auc calculation.
  positive_label<-list(1:nrow(test_typical_data))
  for (val in 1:nrow(test_typical_data))
  {
  positive_label[val] <- 1
  }

  negative_label<-list(1:nrow(test_ood_data))
  for (val in 1:nrow(test_ood_data))
  {
  negative_label[val] <- 0
  }





  #Note: From here, bugs are present
  #Note: From here, bugs are present
  #Note: From here, bugs are present
  #Note: From here, bugs are present
  #Note: From here, bugs are present
  #Note: From here, bugs are present
  yan <-rbind(test_typical_data,test_ood_data)
  yan <- as.data.frame(yan)
  yan <- t(yan)
  yan <- as.data.frame(yan)

  ann<-rbind(test_typical_label,test_ood_label)
  auc_label<-c(positive_label,negative_label)

  sce <- SingleCellExperiment(assays = list(normcounts = as.matrix(yan)), colData = ann)
  logcounts(sce) <- log2(normcounts(sce) + 1)
  # use gene names as feature symbols
  rowData(sce)$feature_symbol <- rownames(sce)
  # remove features with duplicated names
  sce <- sce[!duplicated(rownames(sce)), ]
  table(rowData(sce)$scmap_features)


  #scmap-cluster
  sce <- indexCluster(sce)
  head(metadata(sce)$scmap_cluster_index)
  scmapCluster_results <- scmapCluster(
    projection = sce,
    index_list = list(
      yan = metadata(sce)$scmap_cluster_index
    )
  )
  auc(auc_label,scmapCluster_results$scmap_cluster_siml)
  print(auc(auc_label,scmapCluster_results$scmap_cluster_siml))




  scmapCell_results <- scmapCell(
    sce,
    list(
      yan = metadata(sce)$scmap_cell_index
    )
  )
  auc(auc_label,scmapCell_results$yan$scmap_cluster_siml)
}

#test a small dataset
process("/research/dept8/gds/cyzheng21/data/DeepRCD/scBalance_dataset_process/deng")

# process("/research/dept8/gds/cyzheng21/data/DeepRCD/scBalance_dataset_process/Baron Human/Baron Human")
# process("/research/dept8/gds/cyzheng21/data/DeepRCD/scBalance_dataset_process/Baron Mouse/Baron Mouse")
# process("/research/dept8/gds/cyzheng21/data/DeepRCD/scBalance_dataset_process/campbell")
# process("/research/dept8/gds/cyzheng21/data/DeepRCD/scBalance_dataset_process/campLiver")
# process("/research/dept8/gds/cyzheng21/data/DeepRCD/scBalance_dataset_process/darmanis")
# process("/research/dept8/gds/cyzheng21/data/DeepRCD/scBalance_dataset_process/Goolam")
# process("/research/dept8/gds/cyzheng21/data/DeepRCD/scBalance_dataset_process/lake")
# process("/research/dept8/gds/cyzheng21/data/DeepRCD/scBalance_dataset_process/patel")
# process("/research/dept8/gds/cyzheng21/data/DeepRCD/scBalance_dataset_process/usoskin")
# process("/research/dept8/gds/cyzheng21/data/DeepRCD/scBalance_dataset_process/zillionis")
#
# process("/research/dept8/gds/cyzheng21/data/DeepRCD/scBalance_dataset_process/scRNAseq_Benchmark_datasets/Inter-dataset/PbmcBench/10Xv2/10Xv2")
# process("/research/dept8/gds/cyzheng21/data/DeepRCD/scBalance_dataset_process/scRNAseq_Benchmark_datasets/Inter-dataset/PbmcBench/10Xv3/10Xv3")
# process("/research/dept8/gds/cyzheng21/data/DeepRCD/scBalance_dataset_process/scRNAseq_Benchmark_datasets/Inter-dataset/PbmcBench/CEL-Seq/CEL-Seq")
# process("/research/dept8/gds/cyzheng21/data/DeepRCD/scBalance_dataset_process/scRNAseq_Benchmark_datasets/Inter-dataset/PbmcBench/Drop-Seq/Drop-Seq")
# process("/research/dept8/gds/cyzheng21/data/DeepRCD/scBalance_dataset_process/scRNAseq_Benchmark_datasets/Inter-dataset/PbmcBench/inDrop/inDrop")
# process("/research/dept8/gds/cyzheng21/data/DeepRCD/scBalance_dataset_process/scRNAseq_Benchmark_datasets/Inter-dataset/PbmcBench/Seq-Well/Seq-Well")



