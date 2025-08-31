# pip install -r requirements.txt
source bosch/bin/activate
echo "setting up environment and other files..."
python setup.py

# # if argument 1 : to run dashboard
echo "Running dashboard for dataset analysis..."
streamlit run data_analysis/dashboard.py

# # # if argument 2 : to run training to show dataloaders: model_training/dataloader
echo "Training a yolov9e model..."
python model_training/yolo_train.py

# # if argument 3 : to run validation
echo "Evaluating on validation dataset..."
python model_training/val.py
python model_training/conversion.py

# # if argument 4: to run visualization
echo "Running dashboard for model evaluation and analysis..."
streamlit run evaluation_visualization/dashboard.py
