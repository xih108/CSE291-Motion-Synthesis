# moSyn
Prepare to run:
1. create a folder: model_dir
2. put the three model files in model_dir

==============================Train============================
1. train autoencoder: train.py
2. train feedforward: new_train_regressor.py
3. train footstepper: train_locomotion.py

==============================Demo=============================
1. demo_denoise_new.py (autoencoder only, input: noisy motion, output: recover motion)
2. demo_regression_new.py (feedforward + autoencoder + footstepper, input: noisy motion, output: recover motion)
3. demo_punching_new.py->not finished yet
