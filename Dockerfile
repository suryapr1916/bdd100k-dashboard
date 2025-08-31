FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    libgstreamer1.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python setup.py

ENV CUDA_VISIBLE_DEVICES=""
ENV ARG_NUM=1

EXPOSE 8501

CMD ["sh", "-c", "if [ \"$ARG_NUM\" = \"1\" ]; then echo \"Running dashboard for dataset analysis...\" && streamlit run data_analysis/dashboard.py --server.port=8501 --server.address=0.0.0.0; elif [ \"$ARG_NUM\" = \"2\" ]; then echo \"Training a yolov9e model...\" && python model_training/yolo_train.py; elif [ \"$ARG_NUM\" = \"3\" ]; then echo \"Evaluating on validation dataset...\" && python model_training/val.py && python model_training/conversion.py; elif [ \"$ARG_NUM\" = \"4\" ]; then echo \"Running dashboard for model evaluation and analysis...\" && streamlit run evaluation_visualization/dashboard.py --server.port=8501 --server.address=0.0.0.0; else echo \"Invalid argument. Use ARG_NUM=1,2,3,4\"; fi"]