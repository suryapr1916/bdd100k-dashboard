from ultralytics import YOLO
import os

def main():

    """
    Trains the Ultralytics YOLOv9 model on the BDD100k Dataset.
    
    Returns:
        ultralytics.engine.results.Results: Training results object
    """
    
    model = YOLO("data/models/yolov9e.pt")

    results = model.train(
        data="bdd100k.yaml",
        epochs=10,
        batch=-1,
        freeze=28,
        save=True,
        save_period=1,
        patience=3,
        project="data",
        name="logging",
    )

if __name__ == "__main__":
    main()
