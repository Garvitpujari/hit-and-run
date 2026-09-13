import os
import re
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO
import easyocr
import gdown


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Hit & Run Detection",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 Hit & Run Detection System")
st.caption(
    "Vehicle Tracking → Collision Detection → Hit & Run Analysis → "
    "License Plate Recognition"
)


# ============================================================
# CONFIGURATION
# ============================================================

VEHICLE_MODEL = "yolo11n.pt"

# Your custom trained license plate model
PLATE_MODEL_PATH = Path("license_plate_best.pt")

# ------------------------------------------------------------
# PUT YOUR GOOGLE DRIVE FILE ID HERE
# ------------------------------------------------------------
DRIVE_FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID"


VEHICLE_CLASSES = [2, 3, 5, 7]

# Same thresholds used in your notebook
CONFIDENCE = 0.40
COLLISION_DISTANCE = 100
MOVEMENT_THRESHOLD = 5
STOPPED_THRESHOLD = 3
POST_COLLISION_FRAMES = 30


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_vehicle_model():
    """
    Loads pretrained YOLO11n.
    Ultralytics downloads yolo11n.pt automatically if needed.
    """
    return YOLO(VEHICLE_MODEL)


@st.cache_resource
def load_plate_model():
    """
    Downloads and loads the custom license plate model.
    """

    if not PLATE_MODEL_PATH.exists():

        if DRIVE_FILE_ID == "YOUR_GOOGLE_DRIVE_FILE_ID":
            raise RuntimeError(
                "Google Drive file ID for license_plate_best.pt "
                "has not been configured."
            )

        with st.spinner("Downloading license plate model..."):

            gdown.download(
                id=DRIVE_FILE_ID,
                output=str(PLATE_MODEL_PATH),
                quiet=False
            )

    return YOLO(str(PLATE_MODEL_PATH))


@st.cache_resource
def load_ocr():

    return easyocr.Reader(
        ["en"],
        gpu=False
    )


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def calculate_timestamp(frame_number, fps):

    if fps <= 0:
        fps = 25

    total_seconds = frame_number / fps

    hours = int(total_seconds // 3600)

    minutes = int(
        (total_seconds % 3600) // 60
    )

    seconds = int(
        total_seconds % 60
    )

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def center_distance(a, b):

    ax = (a["x1"] + a["x2"]) / 2
    ay = (a["y1"] + a["y2"]) / 2

    bx = (b["x1"] + b["x2"]) / 2
    by = (b["y1"] + b["y2"]) / 2

    return np.sqrt(
        (ax - bx) ** 2 +
        (ay - by) ** 2
    )


def clean_plate_text(text):

    text = text.upper()

    # Keep only letters and numbers
    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


# ============================================================
# STEP 1 — VEHICLE TRACKING
# ============================================================

def track_vehicles(
    video_path,
    model,
    progress_bar,
    status
):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open uploaded video."
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25

    data = []

    frame_no = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model.track(
            frame,
            tracker="bytetrack.yaml",
            persist=True,
            conf=CONFIDENCE,
            verbose=False
        )

        boxes = results[0].boxes

        if (
            boxes.id is not None
            and len(boxes) > 0
        ):

            ids = boxes.id.cpu().numpy()

            classes = boxes.cls.cpu().numpy()

            xyxy = boxes.xyxy.cpu().numpy()

            for obj_id, cls, box in zip(
                ids,
                classes,
                xyxy
            ):

                x1, y1, x2, y2 = box

                data.append(
                    [
                        frame_no,
                        int(obj_id),
                        int(cls),
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2)
                    ]
                )

        frame_no += 1

        if total_frames > 0:

            progress_bar.progress(
                min(
                    frame_no / total_frames,
                    1.0
                )
            )

        status.text(
            f"Tracking vehicles: "
            f"{frame_no}/{total_frames}"
        )

    cap.release()

    df = pd.DataFrame(
        data,
        columns=[
            "frame",
            "id",
            "class",
            "x1",
            "y1",
            "x2",
            "y2"
        ]
    )

    return df, fps, total_frames


# ============================================================
# STEP 2 — CALCULATE MOVEMENT
# ============================================================

def calculate_speeds(df):

    if df.empty:
        return df

    df = df.copy()

    df["cx"] = (
        df["x1"] +
        df["x2"]
    ) / 2

    df["cy"] = (
        df["y1"] +
        df["y2"]
    ) / 2

    df["prev_cx"] = (
        df.groupby("id")["cx"]
        .shift(1)
    )

    df["prev_cy"] = (
        df.groupby("id")["cy"]
        .shift(1)
    )

    df["speed"] = np.sqrt(
        (
            df["cx"] -
            df["prev_cx"]
        ) ** 2
        +
        (
            df["cy"] -
            df["prev_cy"]
        ) ** 2
    )

    df["speed"] = df["speed"].fillna(0)

    return df


# ============================================================
# STEP 3 — COLLISION DETECTION
# ============================================================

def detect_collision_events(df):

    collision_candidates = []

    if df.empty:
        return pd.DataFrame()

    for frame, group in df.groupby("frame"):

        vehicles = group[
            group["class"].isin(
                VEHICLE_CLASSES
            )
        ].to_dict("records")

        for i in range(len(vehicles)):

            for j in range(i + 1, len(vehicles)):

                a = vehicles[i]

                b = vehicles[j]

                distance = center_distance(
                    a,
                    b
                )

                speed_a = float(
                    a.get("speed", 0)
                )

                speed_b = float(
                    b.get("speed", 0)
                )

                # Same logic as notebook
                if (
                    distance < COLLISION_DISTANCE
                    and
                    (
                        speed_a >
                        MOVEMENT_THRESHOLD
                        or
                        speed_b >
                        MOVEMENT_THRESHOLD
                    )
                ):

                    collision_candidates.append(
                        [
                            frame,
                            a["id"],
                            b["id"],
                            distance,
                            speed_a,
                            speed_b
                        ]
                    )

    if not collision_candidates:
        return pd.DataFrame()

    collision_df = pd.DataFrame(
        collision_candidates,
        columns=[
            "frame",
            "vehicle_1",
            "vehicle_2",
            "distance",
            "speed_1",
            "speed_2"
        ]
    )

    collision_df = collision_df.sort_values(
        "frame"
    )

    return collision_df


# ============================================================
# STEP 4 — GROUP COLLISION FRAMES INTO EVENTS
# ============================================================

def group_collision_events(
    collision_df
):

    if collision_df.empty:
        return pd.DataFrame()

    events = []

    for (
        v1,
        v2
    ), group in collision_df.groupby(
        ["vehicle_1", "vehicle_2"]
    ):

        frames = sorted(
            group["frame"].astype(int).values
        )

        start = frames[0]

        previous = frames[0]

        for frame in frames[1:]:

            if frame - previous > 10:

                events.append(
                    [
                        v1,
                        v2,
                        start,
                        previous
                    ]
                )

                start = frame

            previous = frame

        events.append(
            [
                v1,
                v2,
                start,
                previous
            ]
        )

    return pd.DataFrame(
        events,
        columns=[
            "vehicle_1",
            "vehicle_2",
            "start_frame",
            "end_frame"
        ]
    )


# ============================================================
# STEP 5 — HIT & RUN DETECTION
# ============================================================

def detect_hit_and_run(
    df,
    collision_events
):

    candidates = []

    if (
        df.empty
        or collision_events.empty
    ):
        return pd.DataFrame()

    for _, event in collision_events.iterrows():

        v1 = int(
            event["vehicle_1"]
        )

        v2 = int(
            event["vehicle_2"]
        )

        end_frame = int(
            event["end_frame"]
        )

        after = df[
            (df["frame"] > end_frame)
            &
            (
                df["frame"]
                <=
                end_frame +
                POST_COLLISION_FRAMES
            )
            &
            (
                df["id"].isin(
                    [v1, v2]
                )
            )
        ]

        if after.empty:
            continue

        avg_speed = (
            after.groupby("id")["speed"]
            .mean()
        )

        speed_1 = float(
            avg_speed.get(
                v1,
                0
            )
        )

        speed_2 = float(
            avg_speed.get(
                v2,
                0
            )
        )

        # Same logic as notebook
        if (
            speed_1 > MOVEMENT_THRESHOLD
            and
            speed_2 < STOPPED_THRESHOLD
        ):

            offender = v1

        elif (
            speed_2 > MOVEMENT_THRESHOLD
            and
            speed_1 < STOPPED_THRESHOLD
        ):

            offender = v2

        else:

            continue

        candidates.append(
            [
                v1,
                v2,
                offender,
                end_frame,
                speed_1,
                speed_2
            ]
        )

    return pd.DataFrame(
        candidates,
        columns=[
            "vehicle_1",
            "vehicle_2",
            "offender",
            "collision_frame",
            "speed_1",
            "speed_2"
        ]
    )


# ============================================================
# STEP 6 — GET VIDEO FRAME
# ============================================================

def get_frame(
    video_path,
    frame_number
):

    cap = cv2.VideoCapture(
        video_path
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        int(frame_number)
    )

    ret, frame = cap.read()

    cap.release()

    if not ret:
        return None

    return frame


# ============================================================
# STEP 7 — LICENSE PLATE DETECTION + OCR
# ============================================================

def recognize_plate(
    video_path,
    df,
    offender_id,
    collision_frame,
    plate_model,
    reader
):

    offender_data = df[
        (df["id"] == offender_id)
        &
        (
            df["frame"]
            >=
            collision_frame - 30
        )
        &
        (
            df["frame"]
            <=
            collision_frame + 30
        )
    ]

    if offender_data.empty:

        return None, 0.0, None

    ocr_results = []

    best_plate_image = None

    # Same sampling approach as notebook:
    # every 5th detection
    sampled = offender_data.iloc[::5]

    cap = cv2.VideoCapture(
        video_path
    )

    for _, row in sampled.iterrows():

        frame_no = int(
            row["frame"]
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_no
        )

        ret, frame = cap.read()

        if not ret:
            continue

        height, width = frame.shape[:2]

        x1, y1, x2, y2 = map(
            int,
            [
                row["x1"],
                row["y1"],
                row["x2"],
                row["y2"]
            ]
        )

        # Clamp bounding box
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))

        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))

        if (
            x2 <= x1
            or
            y2 <= y1
        ):
            continue

        vehicle = frame[
            y1:y2,
            x1:x2
        ]

        if vehicle.size == 0:
            continue

        # Detect plate inside offender vehicle
        result = plate_model(
            vehicle,
            imgsz=640,
            conf=0.10,
            verbose=False
        )[0]

        if (
            result.boxes is None
            or
            len(result.boxes) == 0
        ):
            continue

        for box in result.boxes.xyxy:

            px1, py1, px2, py2 = (
                box
                .cpu()
                .numpy()
                .astype(int)
            )

            vh, vw = vehicle.shape[:2]

            px1 = max(
                0,
                min(px1, vw - 1)
            )

            px2 = max(
                0,
                min(px2, vw)
            )

            py1 = max(
                0,
                min(py1, vh - 1)
            )

            py2 = max(
                0,
                min(py2, vh)
            )

            if (
                px2 <= px1
                or
                py2 <= py1
            ):
                continue

            plate = vehicle[
                py1:py2,
                px1:px2
            ]

            if plate.size == 0:
                continue

            # Same enhancement used in notebook
            plate = cv2.resize(
                plate,
                None,
                fx=5,
                fy=5,
                interpolation=cv2.INTER_CUBIC
            )

            text_results = reader.readtext(
                plate,
                allowlist=(
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "0123456789"
                )
            )

            for item in text_results:

                if len(item) < 3:
                    continue

                text = clean_plate_text(
                    item[1]
                )

                confidence = float(
                    item[2]
                )

                if len(text) >= 5:

                    ocr_results.append(
                        (
                            text,
                            confidence
                        )
                    )

                    if (
                        best_plate_image
                        is None
                        or
                        confidence
                        >=
                        max(
                            x[1]
                            for x
                            in ocr_results
                        )
                    ):

                        best_plate_image = plate.copy()

    cap.release()

    if not ocr_results:

        return None, 0.0, None

    # Highest-confidence OCR result
    best_text, best_conf = max(
        ocr_results,
        key=lambda x: x[1]
    )

    return (
        best_text,
        best_conf,
        best_plate_image
    )


# ============================================================
# STEP 8 — CREATE ANNOTATED RESULT VIDEO
# ============================================================

def create_result_video(
    video_path,
    output_path,
    vehicle_model,
    offender_id=None,
    collision_frame=None
):

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open video."
        )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    # mp4v for broad compatibility
    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    progress = st.progress(0)

    frame_no = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = vehicle_model.track(
            frame,
            tracker="bytetrack.yaml",
            persist=True,
            conf=CONFIDENCE,
            verbose=False
        )

        annotated = results[0].plot()

        # Highlight suspected offender
        if (
            offender_id is not None
            and
            collision_frame is not None
        ):

            boxes = results[0].boxes

            if (
                boxes.id is not None
                and
                len(boxes) > 0
            ):

                ids = (
                    boxes.id
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                xyxy = (
                    boxes.xyxy
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                for track_id, box in zip(
                    ids,
                    xyxy
                ):

                    if track_id == offender_id:

                        x1, y1, x2, y2 = box

                        cv2.rectangle(
                            annotated,
                            (x1, y1),
                            (x2, y2),
                            (0, 0, 255),
                            4
                        )

                        cv2.putText(
                            annotated,
                            "SUSPECTED OFFENDER",
                            (x1, max(30, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2
                        )

        # Collision marker
        if (
            collision_frame is not None
            and
            abs(
                frame_no -
                collision_frame
            ) <= 15
        ):

            cv2.rectangle(
                annotated,
                (10, 10),
                (width - 10, height - 10),
                (0, 0, 255),
                5
            )

            cv2.putText(
                annotated,
                "COLLISION / HIT-AND-RUN EVENT",
                (30, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )

        out.write(annotated)

        frame_no += 1

        if total_frames > 0:

            progress.progress(
                min(
                    frame_no /
                    total_frames,
                    1.0
                )
            )

    cap.release()
    out.release()

    progress.empty()


# ============================================================
# LOAD MODELS
# ============================================================

try:

    with st.spinner(
        "Loading AI models..."
    ):

        vehicle_model = (
            load_vehicle_model()
        )

        plate_model = (
            load_plate_model()
        )

        reader = load_ocr()

    st.success(
        "AI models loaded successfully."
    )

except Exception as e:

    st.error(
        f"Could not load models: {e}"
    )

    st.info(
        "Check that your Google Drive file ID "
        "points to the trained license-plate best.pt."
    )

    st.stop()


# ============================================================
# VIDEO UPLOAD
# ============================================================

uploaded_video = st.file_uploader(
    "Upload accident / road video",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv"
    ]
)


if uploaded_video is not None:

    st.subheader(
        "Uploaded Video"
    )

    st.video(
        uploaded_video
    )

    if st.button(
        "🚨 Analyze for Hit & Run",
        type="primary",
        use_container_width=True
    ):

        input_path = None
        output_path = None

        try:

            # ------------------------------------------------
            # Save uploaded video
            # ------------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as tmp:

                tmp.write(
                    uploaded_video.read()
                )

                input_path = tmp.name

            # ------------------------------------------------
            # STEP 1
            # ------------------------------------------------

            st.subheader(
                "1️⃣ Vehicle Tracking"
            )

            progress = st.progress(0)

            status = st.empty()

            df, fps, total_frames = (
                track_vehicles(
                    input_path,
                    vehicle_model,
                    progress,
                    status
                )
            )

            progress.empty()
            status.empty()

            if df.empty:

                st.warning(
                    "No vehicles were detected."
                )

                st.stop()

            st.success(
                f"Tracked {df['id'].nunique()} "
                f"vehicle IDs."
            )

            # ------------------------------------------------
            # STEP 2
            # ------------------------------------------------

            st.subheader(
                "2️⃣ Movement Analysis"
            )

            df = calculate_speeds(
                df
            )

            # ------------------------------------------------
            # STEP 3
            # ------------------------------------------------

            st.subheader(
                "3️⃣ Collision Detection"
            )

            collision_df = (
                detect_collision_events(
                    df
                )
            )

            collision_events = (
                group_collision_events(
                    collision_df
                )
            )

            if collision_events.empty:

                st.warning(
                    "No collision event detected "
                    "with the configured thresholds."
                )

                st.stop()

            st.success(
                f"{len(collision_events)} "
                f"collision event(s) found."
            )

            # ------------------------------------------------
            # STEP 4
            # ------------------------------------------------

            st.subheader(
                "4️⃣ Hit & Run Analysis"
            )

            hit_run_df = (
                detect_hit_and_run(
                    df,
                    collision_events
                )
            )

            if hit_run_df.empty:

                st.info(
                    "Collision detected, but the "
                    "post-collision movement pattern "
                    "did not satisfy the hit-and-run criteria."
                )

                st.dataframe(
                    collision_events,
                    use_container_width=True
                )

                st.stop()

            # ------------------------------------------------
            # Take first detected event
            # Same as notebook
            # ------------------------------------------------

            event = hit_run_df.iloc[0]

            offender_id = int(
                event["offender"]
            )

            collision_frame = int(
                event["collision_frame"]
            )

            timestamp = calculate_timestamp(
                collision_frame,
                fps
            )

            st.error(
                "🚨 HIT-AND-RUN DETECTED"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Offender Track ID",
                offender_id
            )

            col2.metric(
                "Collision Frame",
                collision_frame
            )

            col3.metric(
                "Video Timestamp",
                timestamp
            )

            # ------------------------------------------------
            # STEP 5
            # ------------------------------------------------

            st.subheader(
                "5️⃣ License Plate Recognition"
            )

            with st.spinner(
                "Detecting license plate and running OCR..."
            ):

                plate_text, plate_confidence, plate_img = (
                    recognize_plate(
                        input_path,
                        df,
                        offender_id,
                        collision_frame,
                        plate_model,
                        reader
                    )
                )

            if plate_text:

                st.success(
                    f"Registration Number: "
                    f"**{plate_text}**"
                )

                st.metric(
                    "OCR Confidence",
                    f"{plate_confidence * 100:.1f}%"
                )

                if plate_img is not None:

                    st.image(
                        cv2.cvtColor(
                            plate_img,
                            cv2.COLOR_BGR2RGB
                        ),
                        caption="Detected License Plate",
                        width=400
                    )

            else:

                st.warning(
                    "Plate was not successfully read by OCR."
                )

            # ------------------------------------------------
            # FINAL REPORT
            # ------------------------------------------------

            st.subheader(
                "🚨 Incident Report"
            )

            st.markdown(
                f"""
                **Event:** Hit-and-Run Detected  
                
                **Suspected Offender Track ID:** `{offender_id}`  
                
                **Collision Frame:** `{collision_frame}`  
                
                **Video Timestamp:** `{timestamp}`  
                
                **Registration:** `{plate_text if plate_text else "Not readable"}`
                
                **OCR Confidence:** `{plate_confidence * 100:.1f}%`
                """
            )

            # ------------------------------------------------
            # RESULT VIDEO
            # ------------------------------------------------

            st.subheader(
                "🎥 Annotated Video"
            )

            output_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            output_path = output_file.name

            output_file.close()

            with st.spinner(
                "Creating annotated result video..."
            ):

                create_result_video(
                    input_path,
                    output_path,
                    vehicle_model,
                    offender_id,
                    collision_frame
                )

            st.success(
                "Annotated video generated."
            )

            st.video(
                output_path
            )

            with open(
                output_path,
                "rb"
            ) as f:

                video_bytes = f.read()

            st.download_button(
                label="⬇️ Download Annotated Video",
                data=video_bytes,
                file_name="hit_and_run_detected.mp4",
                mime="video/mp4",
                use_container_width=True
            )

        except Exception as e:

            st.error(
                f"Error during analysis: {e}"
            )

        finally:

            if (
                input_path
                and
                os.path.exists(input_path)
            ):

                os.remove(
                    input_path
                )

            # Don't delete output immediately.
            # Streamlit needs it for playback/download.
