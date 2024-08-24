# GCondNet with KNN graphs
python src/main.py \
	--model 'dnn' \
	--winit_initialisation 'gcondnet' \
	--winit_first_layer_interpolation_scheduler 'linear' \
	--winit_first_layer_interpolation_end_iteration 200 \
	--winit_first_layer_interpolation 1 \
	--winit_graph_connectivity_type 'knn' \
	\
	--logger 'csv' \
	--experiment_name 'GCondNet_KNN' \
	\
	--lr 0.0001 \
	\
	--dataset 'toxicity' \
	--disable_wandb \
	# --run_repeats_and_cv \  # if you want to runs 25 runs (5-fold cross-validation with 5 repeats) 


# GCondNet with SRD graphs
# python src/main.py \
# 	--model 'dnn' \
# 	--winit_initialisation 'gcondnet' \
# 	--winit_first_layer_interpolation_scheduler 'linear' \
# 	--winit_first_layer_interpolation_end_iteration 200 \
# 	--winit_first_layer_interpolation 1 \
# 	--winit_graph_connectivity_type 'sparse-relative-distance' \
# 	\
# 	--logger 'csv' \
# 	--experiment_name 'GCondNet_SRD' \
# 	\
# 	--lr 0.0001 \
# 	\
# 	--dataset 'toxicity' \
# 	--disable_wandb \
# 	# --run_repeats_and_cv \  # if you want to runs 25 runs (5-fold cross-validation with 5 repeats) 