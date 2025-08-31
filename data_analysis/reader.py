import json
import time
import random
from functools import wraps


def timed(func):
    """Decorator that measures and prints the execution time of a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result

    return wrapper


class BDDDatasetReader:
    """A class for reading and filtering BDD (Berkeley DeepDrive) dataset entries."""

    def __init__(self, file_path):
        """
        Initialize the BDD dataset reader.

        Args:
            file_path (str): Path to the dataset file in JSONL format.
        """
        self.file_path = file_path
        self._data = []
        self._load_dataset()

    def _load_dataset(self, max_limit=40000):
        """Load the dataset from file, sampling up to 5000 entries if file is larger."""
        start_time = time.time()

        with open(self.file_path, "r") as file:
            total_lines = sum(1 for _ in file)

        if total_lines <= max_limit:
            with open(self.file_path, "r") as file:
                for line in file:
                    self._data.append(json.loads(line.strip()))
        else:
            random.seed(42)
            selected_indices = set(random.sample(range(total_lines), max_limit))

            with open(self.file_path, "r") as file:
                for i, line in enumerate(file):
                    if i in selected_indices:
                        self._data.append(json.loads(line.strip()))

        load_time = time.time() - start_time
        print(f"Loaded {len(self._data)} entries in {load_time:.4f} seconds")

    def _passes_image_level_filters(self, item, filters):
        """
        Check if an item passes image-level filters.

        Args:
            item (dict): Dataset item to check.
            filters (dict): Dictionary of filters to apply.

        Returns:
            bool: True if item passes all image-level filters.
        """
        image_attrs = item.get("attributes", {})

        for field in ["weather", "scene", "timeofday"]:
            if field in filters and filters[field]:
                attr_value = image_attrs.get(field)
                if attr_value is not None and attr_value not in filters[field]:
                    return False
        return True

    def _passes_label_level_filters(self, item, filters):
        """
        Check if an item has at least one label that satisfies all label-level filters.

        Args:
            item (dict): Dataset item to check.
            filters (dict): Dictionary of filters to apply.

        Returns:
            bool: True if item has at least one label passing all filters.
        """
        labels = item.get("labels", [])

        label_filters = {
            k: v
            for k, v in filters.items()
            if k
            in [
                "category",
                "occluded",
                "truncated",
                "trafficLightColor",
                "areaType",
                "laneDirection",
                "laneStyle",
                "laneType",
            ]
        }

        if not label_filters:
            return True

        for i, label in enumerate(labels):
            label_satisfies_all = True

            if "category" in label_filters and label_filters["category"]:
                category_value = label.get("category")
                if (
                    category_value is not None
                    and category_value not in label_filters["category"]
                ):
                    label_satisfies_all = False
                    continue

            label_attrs = label.get("attributes", {})
            for field in [
                "occluded",
                "truncated",
                "trafficLightColor",
                "areaType",
                "laneDirection",
                "laneStyle",
                "laneType",
            ]:
                if field in label_filters and label_filters[field]:
                    attr_value = label_attrs.get(field)
                    if (
                        attr_value is not None
                        and attr_value not in label_filters[field]
                    ):
                        label_satisfies_all = False
                        break

            if label_satisfies_all:
                return True

        return False

    @timed
    def filter_dataset(self, **filters):
        """
        Filter the dataset based on provided criteria.

        Args:
            **filters: Keyword arguments specifying filter criteria.

        Returns:
            list: Filtered dataset entries.
        """
        filtered_data = []

        for item in self._data:
            if self._passes_image_level_filters(
                item, filters
            ) and self._passes_label_level_filters(item, filters):
                filtered_data.append(item)

        print(f"Filtered to {len(filtered_data)} entries from {len(self._data)} total")
        return filtered_data

    def get_dataset_info(self):
        """
        Get basic statistics about the loaded dataset.

        Returns:
            dict: Dictionary containing dataset statistics.
        """
        total_labels = sum(len(item.get("labels", [])) for item in self._data)
        return {
            "total_images": len(self._data),
            "total_labels": total_labels,
            "avg_labels_per_image": total_labels / len(self._data) if self._data else 0,
        }
