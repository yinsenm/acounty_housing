# Author: Yinsen Miao
# Step 1: preprocess and clean data
# Step 1.a: clean assessment data
#python src/get_data.py

# Step 1.b: Append alternative data
#python src/get_alternative_data.py

# Step 1.c: Download and aggregate geo coding data
#python src/geo_download.py
#python src/geo_aggregate.py

# Step 2: Bayesian hyperparameter tuning of LGBM model
#python src/tune_ml_house_index.py

# Step 3: MVO portfolio Optimization
python src/plt_port_data.py
python src/run_port_opt.py


