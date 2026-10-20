nextflow.enable.dsl=2

include { geneformer_leiden } from params.rootDir + "/target/nextflow/workflows/integration/geneformer_leiden/main.nf"
include { geneformer_finetune } from params.rootDir + "/target/nextflow/workflows/embedding_model/geneformer_finetune/main.nf"

params.resources_test = params.rootDir + "/resources_test"

workflow test_wf {

  resources_test = file(params.resources_test)
  model = resources_test.resolve("geneformer/Geneformer-V1-10M")

  // the fine-tuning starts from the tokens written by geneformer_leiden, so
  // run that first. The labels are two small harmony clusters of the input,
  // expression-space labels like the ones fine-tuning is meant for; the
  // dataset holds one donor, so the cells are split instead.
  output_ch = Channel.fromList([
      [
        id: "simple_execution_test",
        input: resources_test.resolve("pbmc_1k_protein_v3/pbmc_1k_protein_v3_mms.h5mu"),
        model: model,
        model_version: "V1",
        leiden_resolution: [1.0],
      ],
    ])
    | map{ state -> [state.id, state] }
    | geneformer_leiden
    | map { id, state ->
      [id, [
        input: state.output,
        model: model,
        model_version: "V1",
        obs_cluster: "harmony_integration_leiden_1.0",
        disease_clusters: ["7"],
        healthy_clusters: ["8"],
        split_level: "cell",
        obs_report_by: ["geneformer_integration_leiden_1.0"],
        disease_marker_genes: ["CD14", "LYZ", "S100A8"],
        healthy_marker_genes: ["CD3E", "CD3D", "IL7R"],
        var_gene_names: "gene_symbol",
        marker_layer: "log_normalized",
        freeze_layers: [1, 2],
        n_trials: 1,
        epochs: [1],
        batch_size: 4,
        forward_batch_size: 4,
      ]]
    }
    | geneformer_finetune
    | view { output ->
      assert output.size() == 2 : "Outputs should contain two elements; [id, state]"

      // check id
      def id = output[0]
      assert id == "simple_execution_test" : "the searches should be gathered back into one event, got $id"

      // check output
      def state = output[1]
      assert state instanceof Map : "State should be a map. Found: ${state}"
      assert state.output.isFile() : "'output' should be a file."
      assert state.output.toString().endsWith(".h5mu")

      assert state.output_model.isDirectory() : "'output_model' should be a directory"
      def model_files = state.output_model.listFiles().collect { it.name }.sort()
      assert model_files == ["config.json", "id_class_dict.pkl", "model.safetensors", "training_args.bin", "training_params.json"]
      assert state.output_model.resolve("config.json").text.contains("BertForSequenceClassification")

      // one search per number of frozen layers
      assert state.output_trials instanceof List : "'output_trials' should be a list"
      assert state.output_trials.size() == 2 : "expected 2 searches, got ${state.output_trials.size()}"
      def best = new groovy.json.JsonSlurper().parseText(state.output_model.resolve("training_params.json").text)
      def objectives = state.output_trials.collect { f ->
        def lines = f.readLines()
        assert lines[0].startsWith("trial_id,freeze_layers,learning_rate,weight_decay,num_train_epochs")
        assert lines.size() == 2 : "--n_trials 1 should give 1 trial, got ${lines.size() - 1}"
        lines[1].split(",")[-1].toDouble()
      }
      assert Math.abs(best.objective - objectives.max()) < 1e-9 : "the final model should use the best trial over the searches"

      def qc = state.output_label_qc.readLines()
      assert qc[0] == "set,label,n_cells,mean_disease_marker_score,mean_healthy_marker_score"
      assert qc.size() == 1 + 5 * 2 : "5 sets x 2 labels expected"

      def metrics = state.output_metrics.readLines()
      assert metrics[0].startsWith("group_by,group,n_cells,accuracy,macro_f1")
      assert metrics[1].startsWith("all,all,")
      assert metrics.size() > 2 : "expected metrics per geneformer cluster"
      assert state.output_confusion.readLines().size() == 3
      assert state.output_predictions.readLines().size() > 1

      "Output: $output"
    }
    | toSortedList({a, b -> a[0] <=> b[0]})
    | map { output_list ->
      assert output_list.size() == 1 : "output channel should contain 1 event"
    }
}
