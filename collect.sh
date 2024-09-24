for i in {0..7}; do
    python bebs_pipeline/bebs_pipeline.py scene_id=$i \
        max_trajectory_number_per_task=5 \
        success_trajectory_number_per_task=5 \
        output_path=your_path_to_output
done
