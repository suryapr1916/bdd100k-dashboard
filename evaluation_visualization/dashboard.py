import streamlit as st
import os
import json
import random
import pandas as pd
import plotly.express as px
from PIL import Image, ImageDraw, ImageFont
from reader import BDDDatasetReader
import numpy as np

st.set_page_config(page_title="BDD Dataset Dashboard", layout="wide")

COLORS = {
    "car": (255, 0, 0),
    "person": (0, 255, 0),
    "bike": (0, 0, 255),
    "bus": (255, 255, 0),
    "truck": (255, 0, 255),
    "motor": (0, 255, 255),
    "rider": (128, 0, 128),
    "traffic light": (255, 165, 0),
    "traffic sign": (128, 128, 0),
    "train": (0, 128, 128),
    "drivable area": (128, 0, 0),
    "lane": (0, 128, 0),
}


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "filters_applied" not in st.session_state:
        st.session_state.filters_applied = False
    if "refresh_key" not in st.session_state:
        st.session_state.refresh_key = 0


@st.cache_data
def load_metadata():
    """
    Load image level attributes and label specific attributes from JSON files.

    Returns:
        tuple: A tuple containing image attributes and label attributes dictionaries
    """
    with open("data/metadata/img_attr_dict.json", "r") as f:
        img_attrs = json.load(f)
    with open("data/metadata/label_attr_dict.json", "r") as f:
        label_attrs = json.load(f)
    label_attrs["occluded"] = [False, True]
    label_attrs["truncated"] = [False, True]
    return img_attrs, label_attrs


def draw_bounding_boxes(image, labels, filters):
    """
    Draw bounding boxes on image for labels that satisfy all filters.

    Args:
        image (PIL.Image): The input image
        labels (list): List of label dictionaries
        filters (dict): Dictionary of filter criteria

    Returns:
        PIL.Image: Image with bounding boxes drawn
    """
    draw = ImageDraw.Draw(image)

    for i, label in enumerate(labels):
        label_category = label.get("category")
        should_draw = True

        if "category" in filters and filters["category"]:
            if label_category not in filters["category"]:
                should_draw = False

        if should_draw:
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
                if field in filters and filters[field]:
                    attr_value = label_attrs.get(field)
                    if attr_value is not None and attr_value not in filters[field]:
                        should_draw = False
                        break

        if should_draw:
            box2d = label.get("box2d", {})
            if box2d:
                x1 = box2d.get("x1", 0)
                y1 = box2d.get("y1", 0)
                x2 = box2d.get("x2", 0)
                y2 = box2d.get("y2", 0)

                if x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1:
                    color = COLORS.get(label_category, (255, 255, 255))
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    font = ImageFont.load_default()
                    text = label_category
                    bbox = draw.textbbox((x1, y1 - 20), text, font=font)
                    draw.rectangle(bbox, fill=color)
                    draw.text((x1, y1 - 20), text, fill=(0, 0, 0), font=font)

    return image


def create_image_level_plots(image_df):
    """
    Create image-level statistical plots.

    Args:
        image_df (pd.DataFrame): DataFrame containing image-level attributes
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        if "weather" in image_df.columns:
            weather_counts = image_df["weather"].value_counts()
            fig = px.bar(
                x=weather_counts.index,
                y=weather_counts.values,
                title="Weather Distribution",
                labels={"x": "Weather", "y": "Count"},
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "scene" in image_df.columns:
            scene_counts = image_df["scene"].value_counts()
            fig = px.bar(
                x=scene_counts.index,
                y=scene_counts.values,
                title="Scene Distribution",
                labels={"x": "Scene", "y": "Count"},
            )
            st.plotly_chart(fig, use_container_width=True)

    with col3:
        if "timeofday" in image_df.columns:
            time_counts = image_df["timeofday"].value_counts()
            fig = px.bar(
                x=time_counts.index,
                y=time_counts.values,
                title="Time of Day Distribution",
                labels={"x": "Time of Day", "y": "Count"},
            )
            st.plotly_chart(fig, use_container_width=True)


def create_label_level_plots(label_df):
    """
    Create label-level statistical plots.

    Args:
        label_df (pd.DataFrame): DataFrame containing label-level attributes
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        if "category" in label_df.columns:
            category_counts = label_df["category"].value_counts()
            fig = px.bar(
                x=category_counts.index,
                y=category_counts.values,
                title="Object Category Distribution",
                labels={"x": "Category", "y": "Count"},
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "occluded" in label_df.columns:
            occluded_counts = label_df["occluded"].value_counts()
            fig = px.bar(
                x=occluded_counts.index,
                y=occluded_counts.values,
                title="Occlusion Distribution",
                labels={"x": "Occluded", "y": "Count"},
            )
            st.plotly_chart(fig, use_container_width=True)

    with col3:
        if "truncated" in label_df.columns:
            truncated_counts = label_df["truncated"].value_counts()
            fig = px.bar(
                x=truncated_counts.index,
                y=truncated_counts.values,
                title="Truncation Distribution",
                labels={"x": "Truncated", "y": "Count"},
            )
            st.plotly_chart(fig, use_container_width=True)

    col4, col5, col6 = st.columns(3)

    with col4:
        if "trafficLightColor" in label_df.columns:
            traffic_counts = label_df["trafficLightColor"].value_counts()
            fig = px.bar(
                x=traffic_counts.index,
                y=traffic_counts.values,
                title="Traffic Light Color Distribution",
                labels={"x": "Color", "y": "Count"},
            )
            st.plotly_chart(fig, use_container_width=True)

    with col5:
        if "areaType" in label_df.columns:
            area_counts = label_df["areaType"].value_counts()
            fig = px.bar(
                x=area_counts.index,
                y=area_counts.values,
                title="Area Type Distribution",
                labels={"x": "Area Type", "y": "Count"},
            )
            st.plotly_chart(fig, use_container_width=True)

    with col6:
        if "laneDirection" in label_df.columns:
            lane_dir_counts = label_df["laneDirection"].value_counts()
            fig = px.bar(
                x=lane_dir_counts.index,
                y=lane_dir_counts.values,
                title="Lane Direction Distribution",
                labels={"x": "Direction", "y": "Count"},
            )
            st.plotly_chart(fig, use_container_width=True)

    col7, col8, col9 = st.columns(3)

    with col7:
        if "laneStyle" in label_df.columns:
            lane_style_counts = label_df["laneStyle"].value_counts()
            fig = px.bar(
                x=lane_style_counts.index,
                y=lane_style_counts.values,
                title="Lane Style Distribution",
                labels={"x": "Style", "y": "Count"},
            )
            st.plotly_chart(fig, use_container_width=True)

    with col8:
        if "laneType" in label_df.columns:
            lane_type_counts = label_df["laneType"].value_counts()
            fig = px.bar(
                x=lane_type_counts.index,
                y=lane_type_counts.values,
                title="Lane Type Distribution",
                labels={"x": "Type", "y": "Count"},
            )
            st.plotly_chart(fig, use_container_width=True)

    with col9:
        st.write("**Label Summary:**")
        st.write(f"Total labels: {len(label_df)}")
        st.write(f"Unique categories: {label_df['category'].nunique()}")


def extract_rgb_and_dimensions(filtered_data, split):
    """
    Extract RGB values and dimensions from sampled images.

    Args:
        filtered_data (list): List of filtered image data
        split (str): Dataset split (train/val)

    Returns:
        tuple: Lists of RGB values and dimensions
    """

    sample_size = min(100, len(filtered_data))
    random.seed(42)
    sampled_data = random.sample(filtered_data, sample_size)

    rgb_values = []
    dimensions = []

    for item in sampled_data:
        img_name = item.get("name", "")
        img_path = os.path.join(
            f"data/assignment_data_bdd/bdd100k_images_100k/bdd100k/images/100k/val",
            img_name,
        )

        if os.path.exists(img_path):
            try:
                image = Image.open(img_path)
                if image.size[0] > 0 and image.size[1] > 0:
                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    for label in item.get("labels", []):
                        box2d = label.get("box2d", {})
                        if box2d:
                            x1, y1, x2, y2 = (
                                box2d.get("x1", 0),
                                box2d.get("y1", 0),
                                box2d.get("x2", 0),
                                box2d.get("y2", 0),
                            )

                            if x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1:
                                width = x2 - x1
                                height = y2 - y1
                                area = width * height

                                try:
                                    bbox_image = image.crop(
                                        (int(x1), int(y1), int(x2), int(y2))
                                    )
                                    bbox_array = np.array(bbox_image)

                                    if bbox_array.size > 0:
                                        h, w, _ = bbox_array.shape
                                        if h > 0 and w > 0:
                                            num_samples = min(10, h * w)
                                            indices = np.random.choice(
                                                h * w, num_samples, replace=False
                                            )
                                            for idx in indices:
                                                row, col = idx // w, idx % w
                                                r, g, b = bbox_array[row, col]
                                                rgb_values.append(
                                                    {"R": r, "G": g, "B": b}
                                                )

                                    dimensions.append(
                                        {
                                            "width": width,
                                            "height": height,
                                            "area": area,
                                            "category": label.get(
                                                "category", "unknown"
                                            ),
                                        }
                                    )

                                except Exception as e:
                                    continue

            except Exception as e:
                continue

    return rgb_values, dimensions


def create_additional_analysis_plots(rgb_values, dimensions):
    """
    Create RGB distribution and height vs width scatter plots.

    Args:
        rgb_values (list): List of RGB value dictionaries
        dimensions (list): List of dimension dictionaries
    """
    if rgb_values and dimensions:
        col1, col2 = st.columns(2)

        with col1:
            st.write("**RGB Distribution (KDE)**")
            rgb_df = pd.DataFrame(rgb_values)

            fig = px.histogram(
                rgb_df,
                x=["R", "G", "B"],
                title="RGB Channel Distribution",
                labels={"value": "Pixel Value", "variable": "Channel"},
                opacity=0.7,
                barmode="overlay",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.write("**Height vs Width Scatter Plot**")
            dim_df = pd.DataFrame(dimensions)

            fig = px.scatter(
                dim_df,
                x="width",
                y="height",
                color="category",
                title="Object Dimensions: Height vs Width",
                labels={"width": "Width (pixels)", "height": "Height (pixels)"},
                hover_data=["area"],
            )
            st.plotly_chart(fig, use_container_width=True)

        st.write("**Dimension Summary:**")
        dim_df = pd.DataFrame(dimensions)
        st.write(f"Total valid labels analyzed: {len(dimensions)}")
        st.write(f"Average width: {dim_df['width'].mean():.1f} pixels")
        st.write(f"Average height: {dim_df['height'].mean():.1f} pixels")
        st.write(f"Average area: {dim_df['area'].mean():.1f} pixels²")
    else:
        st.warning("No valid labels found for RGB and dimension analysis")


def plot_statistics(filtered_data, split):
    """
    Plot comprehensive statistics for filtered data including image-level,
    label-level, and additional analysis plots.

    Args:
        filtered_data (list): List of filtered image data
        split (str): Dataset split (train/val)
    """
    st.subheader("Image-Level Statistics")

    image_attrs = []
    for item in filtered_data:
        attrs = item.get("attributes", {})
        image_attrs.append(
            {
                "weather": attrs.get("weather"),
                "scene": attrs.get("scene"),
                "timeofday": attrs.get("timeofday"),
            }
        )

    image_df = pd.DataFrame(image_attrs)
    create_image_level_plots(image_df)

    st.subheader("Label-Level Statistics")

    label_attrs = []
    for item in filtered_data:
        for label in item.get("labels", []):
            attrs = label.get("attributes", {})
            label_attrs.append(
                {
                    "category": label.get("category"),
                    "occluded": attrs.get("occluded"),
                    "truncated": attrs.get("truncated"),
                    "trafficLightColor": attrs.get("trafficLightColor"),
                    # "areaType": attrs.get("areaType"),
                    # "laneDirection": attrs.get("laneDirection"),
                    # "laneStyle": attrs.get("laneStyle"),
                    # "laneType": attrs.get("laneType"),
                }
            )

    if label_attrs:
        label_df = pd.DataFrame(label_attrs)
        create_label_level_plots(label_df)

        st.write(f"Average labels per image: {len(label_df) / len(filtered_data):.2f}")

    st.subheader("Additional Analysis")

    rgb_values, dimensions = extract_rgb_and_dimensions(filtered_data, split)
    create_additional_analysis_plots(rgb_values, dimensions)


def render_filter_ui(img_attrs, label_attrs):
    """
    Render the filter user interface and return selected filter values.

    Args:
        img_attrs (dict): Image attributes dictionary
        label_attrs (dict): Label attributes dictionary

    Returns:
        dict: Dictionary containing all selected filter values
    """
    split = st.selectbox("Split", ["correct", "missed"])

    st.subheader("Image-Level Filters")
    col1, col2, col3 = st.columns(3)
    with col1:
        weather = st.multiselect(
            "Weather", img_attrs["weather"], default=img_attrs["weather"]
        )
    with col2:
        scene = st.multiselect("Scene", img_attrs["scene"], default=img_attrs["scene"])
    with col3:
        timeofday = st.multiselect(
            "Time of Day", img_attrs["timeofday"], default=img_attrs["timeofday"]
        )

    st.subheader("Object-Level Filters")
    category = st.multiselect(
        "Object Category",
        [
            "car",
            "person",
            "bike",
            "bus",
            "truck",
            "motor",
            "rider",
            "traffic light",
            "traffic sign",
            "train",
            "drivable area",
            "lane",
        ],
        default=[
            "car",
            "person",
            "bike",
            "bus",
            "truck",
            "motor",
            "rider",
            "traffic light",
            "traffic sign",
            "train",
            "drivable area",
            "lane",
        ],
    )

    col4, col5, col6 = st.columns(3)
    with col4:
        occluded = st.multiselect(
            "Occluded", label_attrs["occluded"], default=label_attrs["occluded"]
        )
    with col5:
        truncated = st.multiselect(
            "Truncated", label_attrs["truncated"], default=label_attrs["truncated"]
        )
    with col6:
        traffic_light_color = st.multiselect(
            "Traffic Light Color",
            label_attrs["trafficLightColor"],
            default=label_attrs["trafficLightColor"],
        )

    col7, col8, col9 = st.columns(3)
    # with col7:
    #     area_type = st.multiselect(
    #         "Area Type", label_attrs["areaType"], default=label_attrs["areaType"]
    #     )
    # with col8:
    #     lane_direction = st.multiselect(
    #         "Lane Direction",
    #         label_attrs["laneDirection"],
    #         default=label_attrs["laneDirection"],
    #     )
    # with col9:
    #     lane_style = st.multiselect(
    #         "Lane Style", label_attrs["laneStyle"], default=label_attrs["laneStyle"]
    #     )

    col10, col11, col12 = st.columns(3)
    # with col10:
    #     lane_type = st.multiselect(
    #         "Lane Type", label_attrs["laneType"], default=label_attrs["laneType"]
    #     )

    return {
        "split": split,
        "weather": weather,
        "scene": scene,
        "timeofday": timeofday,
        "category": category,
        "occluded": occluded,
        "truncated": truncated,
        "traffic_light_color": traffic_light_color,
        # "area_type": area_type,
        # "lane_direction": lane_direction,
        # "lane_style": lane_style,
        # "lane_type": lane_type,
    }


def render_action_buttons():
    """Render action buttons and handle their functionality."""
    st.subheader("Actions")
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.button("Set Default Filters", type="secondary"):
            st.session_state.filters_applied = False
            st.rerun()

    with col_btn2:
        if st.button("Refresh Images", type="secondary"):
            st.session_state.refresh_key += 1
            st.rerun()

    return st.button("Apply Filters", type="primary")


def validate_filters(filters):
    """
    Validate that all filter fields have values.

    Args:
        filters (dict): Dictionary of filter values

    Returns:
        list: List of empty field names
    """
    empty_fields = []
    field_mapping = {
        "weather": "Weather",
        "scene": "Scene",
        "timeofday": "Time of Day",
        "category": "Object Category",
        "traffic_light_color": "Traffic Light Color",
        "occluded": "Occluded",
        "truncated": "Truncated",
        # "area_type": "Area Type",
        # "lane_direction": "Lane Direction",
        # "lane_style": "Lane Style",
        # "lane_type": "Lane Type",
    }

    for key, display_name in field_mapping.items():
        if not filters[key]:
            empty_fields.append(display_name)

    return empty_fields


def display_sample_data(filtered_data):
    """
    Display sample data from filtered results.

    Args:
        filtered_data (list): List of filtered image data
    """
    if filtered_data:
        st.subheader("Sample Data")
        sample = filtered_data[:5]
        for i, item in enumerate(sample):
            with st.expander(f"Image {i+1}: {item.get('name', 'Unknown')}"):
                st.json(item)


def display_sample_images(filtered_data, filters, split):
    """
    Display sample images with bounding boxes.

    Args:
        filtered_data (list): List of filtered image data
        filters (dict): Dictionary of filter values
        split (str): Dataset split (train/val)
    """
    st.header("Sample Images with Bounding Boxes")
    image_dir = f"data/assignment_data_bdd/bdd100k_images_100k/bdd100k/images/100k/val"
    if os.path.exists(image_dir) and filtered_data:
        all_filtered_image_names = [item.get("name", "") for item in filtered_data]
        random.seed(st.session_state.refresh_key)
        filtered_image_names = random.sample(
            all_filtered_image_names, min(12, len(all_filtered_image_names))
        )

        cols = st.columns(4)
        for idx, img_name in enumerate(filtered_image_names):
            if img_name:
                col_idx = idx % 4
                with cols[col_idx]:
                    img_path = os.path.join(image_dir, img_name)
                    if os.path.exists(img_path):
                        try:
                            image = Image.open(img_path)

                            if image.size[0] <= 0 or image.size[1] <= 0:
                                st.error(f"Invalid image dimensions: {img_name}")
                                continue

                            image_data = None
                            for item in filtered_data:
                                if item.get("name") == img_name:
                                    image_data = item
                                    break

                            if image_data and image_data.get("labels"):
                                filter_dict = {
                                    "category": filters["category"],
                                    "occluded": filters["occluded"],
                                    "truncated": filters["truncated"],
                                    "trafficLightColor": filters["traffic_light_color"],
                                    # "areaType": filters["area_type"],
                                    # "laneDirection": filters["lane_direction"],
                                    # "laneStyle": filters["lane_style"],
                                    # "laneType": filters["lane_type"],
                                }
                                image_with_boxes = draw_bounding_boxes(
                                    image.copy(), image_data["labels"], filter_dict
                                )
                                st.image(
                                    image_with_boxes,
                                    caption=f"{img_name} (with boxes)",
                                    use_container_width=True,
                                )
                            else:
                                st.image(
                                    image, caption=img_name, use_container_width=True
                                )

                        except Exception as e:
                            st.error(f"Error loading {img_name}: {e}")
                    else:
                        st.error(f"Image not found: {img_name}")
    else:
        st.warning("No images found for the filtered results")


def process_filtered_data(filters):
    """
    Process data with applied filters and display results.

    Args:
        filters (dict): Dictionary of filter values
    """
    try:
        file_path = f"data/metadata/{filters['split']}.jsonl"

        if not os.path.exists(file_path):
            st.error(f"Data file not found: {file_path}")
            st.stop()

        with st.spinner("Loading dataset..."):
            reader = BDDDatasetReader(file_path)

        with st.spinner("Applying filters..."):
            filtered_data = reader.filter_dataset(
                weather=filters["weather"],
                scene=filters["scene"],
                timeofday=filters["timeofday"],
                category=filters["category"],
                trafficLightColor=filters["traffic_light_color"],
                occluded=filters["occluded"],
                truncated=filters["truncated"],
                # areaType=filters["area_type"],
                # laneDirection=filters["lane_direction"],
                # laneStyle=filters["lane_style"],
                # laneType=filters["lane_type"],
            )

        st.success(f"Found {len(filtered_data)} matching images")

        display_sample_data(filtered_data)
        display_sample_images(filtered_data, filters, filters["split"])

        if filtered_data:
            plot_statistics(filtered_data, filters["split"])

    except Exception as e:
        st.error(f"Error processing data: {e}")


def main():
    """Main application function."""
    initialize_session_state()

    st.title("BDD Dataset Dashboard")
    st.header("Filters")

    img_attrs, label_attrs = load_metadata()
    filters = render_filter_ui(img_attrs, label_attrs)
    apply_filters = render_action_buttons()

    if apply_filters:
        st.session_state.filters_applied = True

    if st.session_state.filters_applied:
        empty_fields = validate_filters(filters)

        if empty_fields:
            st.warning(f"Empty field values give no results: {', '.join(empty_fields)}")
            st.stop()

        process_filtered_data(filters)

    if not st.session_state.filters_applied:
        st.header("Statistics")


if __name__ == "__main__":
    main()
