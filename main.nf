// Acegen ML Nextflow Pipeline (DSL2)

nextflow.enable.dsl=2

// Train RandomForest Classifier
process TRAIN_CLASSIFIER {
    tag "Training Random Forest classifier on dataset"

    publishDir "${projectDir}/data", mode: 'copy'

    input:
    path dataset_csv

    output:
    path "model.pkl", emit: model

    script:
    """
    train_classifier.py --input ${dataset_csv} --output model.pkl
    """
}

// Reinforcement Learning (REINVENT) Optimization
process OPTIMIZE_REINVENT {
    tag "Running Reinvent Optimization"

    input:
    path model_pkl
    path configs_dir
    path scripts_dir

    output:
    path "reinvent_done.txt", emit: marker

    script:
    """
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    python ${scripts_dir}/reinvent.py \
        --config-path ../configs \
        --config-name RFC_config_denovo.yaml

    touch reinvent_done.txt
    """
}

// Analyze and Visualize Optimization Results
process ANALYZE_RESULTS {
    tag "Analyzing optimization results"

    publishDir "${projectDir}/results", mode: 'copy'

    input:
    val marker

    output:
    path "rfc_average_score_steps.png"
    path "rfc_top_molecules.png"

    script:
    """
    analyze_results.py --results-dir ${projectDir}/results --output-dir .
    """
}

// Workflow Orchestration
workflow {
    model_ch = TRAIN_CLASSIFIER(params.dataset)
    opt_marker_ch = OPTIMIZE_REINVENT(model_ch, params.configs_dir, params.scripts_dir)
    ANALYZE_RESULTS(opt_marker_ch)
}
