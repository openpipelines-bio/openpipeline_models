workflow run_wf {
  take:
  input_ch

  main:
  datasets_ch = input_ch
    | map {id, state ->
      [id, state + ["workflow_output": state.output]]
    }
    // 1. disease / healthy / unused labels; this file is the h5mu output
    | label_cells.run(
      fromState: [
        "input": "input",
        "modality": "modality",
        "obs_cluster": "obs_cluster",
        "obs_reference": "obs_reference",
        "disease_clusters": "disease_clusters",
        "disease_reference_clusters": "disease_reference_clusters",
        "healthy_clusters": "healthy_clusters",
        "healthy_reference_clusters": "healthy_reference_clusters",
        "obs_output": "obs_label_output",
        "output": "workflow_output",
      ],
      args: ["uns_output": "geneformer_finetune_label_cells"],
      toState: ["input": "output", "output": "output"]
    )
    // 2. train / eval / test split, downsampled search set, marker QC
    | from_h5mu_to_geneformer_dataset.run(
      fromState: [
        "input": "input",
        "modality": "modality",
        "obsm_input": "obsm_tokens",
        "model_version": "model_version",
        "obs_label": "obs_label_output",
        "split_level": "split_level",
        "obs_donor": "obs_donor",
        "test_size": "test_size",
        "eval_size": "eval_size",
        "max_cells_per_label": "max_cells_per_label",
        "seed": "seed",
        "obs_report_by": "obs_report_by",
        "disease_marker_genes": "disease_marker_genes",
        "healthy_marker_genes": "healthy_marker_genes",
        "var_gene_names": "var_gene_names",
        "marker_layer": "marker_layer",
        "output_label_qc": "output_label_qc",
      ],
      toState: [
        "train_dataset": "output_train",
        "train_search_dataset": "output_train_search",
        "test_dataset": "output_test",
        "id_class_dict": "output_id_class_dict",
        "output_label_qc": "output_label_qc",
      ]
    )

  // 3. one hyperparameter search per number of frozen layers
  searches_ch = datasets_ch
    | flatMap {id, state ->
      state.freeze_layers.collect { freeze ->
        [
          "${id}_freeze${freeze}".toString(),
          state + ["run_id": id, "freeze": freeze]
        ]
      }
    }
    | geneformer_classifier_hyperopt.run(
      fromState: [
        "input": "train_search_dataset",
        "id_class_dict": "id_class_dict",
        "model": "model",
        "model_version": "model_version",
        "freeze_layers": "freeze",
        "n_trials": "n_trials",
        "lr_min": "lr_min",
        "lr_max": "lr_max",
        "weight_decay_min": "weight_decay_min",
        "weight_decay_max": "weight_decay_max",
        "epochs": "epochs",
        "batch_size": "batch_size",
        "gradient_accumulation_steps": "gradient_accumulation_steps",
        "mixed_precision": "mixed_precision",
        "seed": "seed",
      ],
      toState: ["trials_csv": "output_trials", "best_json": "output_best"]
    )

  // 4. best trial over the branches, highest objective, fewest frozen layers
  // on a tie; train the final model with it
  output_ch = searches_ch
    | map {id, state -> [state.run_id, state]}
    | groupTuple(sort: { a, b -> a.freeze <=> b.freeze })
    | map {id, states ->
      def best = states
        .collect { s -> [state: s, objective: new groovy.json.JsonSlurper().parseText(s.best_json.text).objective] }
        .sort { a, b -> b.objective <=> a.objective ?: a.state.freeze <=> b.state.freeze }
        .first()
      [id, best.state + [
        "best_params": best.state.best_json,
        "output_trials": states.collect { it.trials_csv },
      ]]
    }
    | geneformer_classifier_train.run(
      fromState: [
        "input": "train_dataset",
        "id_class_dict": "id_class_dict",
        "model": "model",
        "model_version": "model_version",
        "params": "best_params",
        "mixed_precision": "mixed_precision",
        "output_model": "output_model",
      ],
      toState: ["output_model": "output_model"]
    )
    // 5. metrics on the test set
    | geneformer_classifier_evaluate.run(
      fromState: [
        "input": "test_dataset",
        "id_class_dict": "id_class_dict",
        "model": "output_model",
        "model_version": "model_version",
        "obs_report_by": "obs_report_by",
        "forward_batch_size": "forward_batch_size",
        "output_metrics": "output_metrics",
        "output_confusion": "output_confusion",
        "output_predictions": "output_predictions",
      ],
      toState: [
        "output_metrics": "output_metrics",
        "output_confusion": "output_confusion",
        "output_predictions": "output_predictions",
      ]
    )
    | setState([
      "output",
      "output_model",
      "output_trials",
      "output_label_qc",
      "output_metrics",
      "output_confusion",
      "output_predictions",
    ])

  emit:
  output_ch
}
