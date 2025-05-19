import os
import math
import glob
import natsort
import pathlib
import argparse

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def calculate_distance_3d(x1, y1, z1, x2, y2, z2):
    """
    Calculates the Euclidean distance between two 3D points.

    Args:
        x1 (float): X-coordinate of the first point.
        y1 (float): Y-coordinate of the first point.
        z1 (float): Z-coordinate of the first point.
        x2 (float): X-coordinate of the second point.
        y2 (float): Y-coordinate of the second point.
        z2 (float): Z-coordinate of the second point.

    Returns:
        float: The Euclidean distance between the two points.
    """
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def detect_finger_movement_peaks(distance_array, output_peak_file_path, start_frame_idx, end_frame_idx):
    """
    Detects and records changes in finger state (open/closed) based on distance.

    The function analyzes a time series of distances (e.g., between thumb and index finger).
    It identifies peaks (maximum distance before closing) and bottoms (minimum distance
    before opening) in the movement.

    - It calculates the median of the distances in the specified frame range.
    - Thresholds (peak_threshold, bottom_threshold) are set based on this median.
    - It iterates through the distance array:
        - If the current state is 'closed' (ud_marker=0) and distance exceeds peak_threshold,
          it checks if this point is a local maximum (inflection point). If so, it records
          this as a 'closed' peak (transition to opening).
        - If the current state is 'open' (ud_marker=1) and distance falls below bottom_threshold,
          it checks if this point is a local minimum. If so, it records this as an
          'opened' peak (transition to closing).
        - Points not meeting these criteria or not being clear inflection points are marked
          as 'phase shifting'.

    Args:
        distance_array (list or np.array): Array of distances between fingers for each frame.
        output_peak_file_path (str): Path to the file where detected peaks will be written.
        start_frame_idx (int): The starting frame index in the distance_array to consider.
        end_frame_idx (int): The ending frame index (exclusive) in the distance_array.
    """
    with open(output_peak_file_path, "w") as f:
        relevant_distances = distance_array[start_frame_idx:end_frame_idx]
        if not relevant_distances:
            print(f"Warning: No distances to process for {output_peak_file_path}")
            return

        median_dist = np.median(relevant_distances)
        peak_threshold = median_dist * 1.1
        bottom_threshold = median_dist / 1.1

        # Initialize finger state: 1 for 'open', 0 for 'closed'
        # Based on whether the first distance is greater than the median.
        if relevant_distances[0] > median_dist:
            ud_marker = 1  # Initially considered 'opened'
        else:
            ud_marker = 0  # Initially considered 'closed'

        # Iterate through distances to find peaks (local max/min)
        # Requires at least 3 points (i, i+1, i+2) to check for inflection.
        for i in range(len(relevant_distances) - 2):
            current_frame_global_idx = start_frame_idx + i + 1
            dist_prev = relevant_distances[i]
            dist_curr = relevant_distances[i+1]
            dist_next = relevant_distances[i+2]

            # Current state is 'closed' (ud_marker = 0), looking for an opening peak
            if dist_curr > peak_threshold and ud_marker == 0:
                # Check if dist_curr is a local maximum (peak of an "open" gesture)
                if (dist_curr - dist_prev > 0) and (dist_curr - dist_next >= 0):
                    f.write(f"{current_frame_global_idx} {dist_curr} 1 closed\n") # Peak before closing
                    ud_marker = 1 # State changes to 'open'
                else:
                    # Not a clear peak, consider it phase shifting
                    f.write(f"{current_frame_global_idx} 2 phase shifting\n")
            # Current state is 'open' (ud_marker = 1), looking for a closing peak
            elif dist_curr < bottom_threshold and ud_marker == 1:
                # Check if dist_curr is a local minimum (bottom of a "closed" gesture)
                if (dist_curr - dist_prev < 0) and (dist_curr - dist_next <= 0):
                    f.write(f"{current_frame_global_idx} {dist_curr} 0 opened\n") # Peak before opening
                    ud_marker = 0 # State changes to 'closed'
                else:
                    # Not a clear minimum, consider it phase shifting
                    f.write(f"{current_frame_global_idx} 2 phase shifting\n")
            else:
                # No state change detected according to thresholds or current state
                f.write(f"{current_frame_global_idx} 2 phase shifting\n")

# -----------------------------------------------------------------------------
# Main Processing Function
# -----------------------------------------------------------------------------

def process_patient_session(patient_id, session_state, hand_label, fps,
                            hand_joint_base_dir, output_base_dir):
    """
    Processes hand joint data for a single patient session and hand.

    This involves:
    1. Constructing paths to input NPY files (hand landmarks) and output files.
    2. Creating output directories.
    3. Loading hand landmark data for each frame.
       - The landmarks are expected to be normalized; a transformation `(value + 1) * 100` is applied.
    4. Calculating the 3D distance between thumb tip (landmark 4) and index finger tip (landmark 8).
    5. Saving these per-frame distances to a text file and a NumPy NPY file.
    6. Calling `detect_finger_movement_peaks` to identify and save finger opening/closing peaks.
    7. Reading the peak detector output to compile lists of peak frames and amplitudes.
       - Peak times are also calculated using the provided FPS.

    Args:
        patient_id (str): Identifier for the patient (e.g., "1").
        session_state (str): Identifier for the session/state (e.g., "preopoff").
        hand_label (str): Identifier for the hand (e.g., "L" or "R").
        fps (float): Frames per second for the video corresponding to this session.
        hand_joint_base_dir (str): Base directory where hand joint NPY files are stored.
                                   Expected structure: <hand_joint_base_dir>/<patient_id>/<session_state>/<hand_label>/*.npy
        output_base_dir (str): Base directory where results will be saved.
                               Output structure: <output_base_dir>/<patient_id>_<hand_label>/
    """
    print(f"Processing: Patient {patient_id}, State {session_state}, Hand {hand_label}, FPS {fps}")

    patient_id,session_state,hand_label = str(patient_id),str(session_state),str(hand_label)
    
    session_data_dir = os.path.join(hand_joint_base_dir, patient_id, session_state, hand_label)

    output_patient_hand_dir = os.path.join(output_base_dir, f"{patient_id}_{hand_label}")
    
    pathlib.Path(output_patient_hand_dir).mkdir(parents=True, exist_ok=True)

    npy_files = natsort.natsorted(glob.glob(os.path.join(session_data_dir, '*.npy')))

    if not npy_files:
        print(f"Warning: No NPY files found in {os.path.join(session_data_dir, '*.npy')}. Skipping.")
        return

    distances_output_txt_path = os.path.join(output_patient_hand_dir, f"{patient_id}_{session_state}_{hand_label}.txt")
    frame_distances = []

    with open(distances_output_txt_path, 'w') as patient_result_file:
        for frame_idx, npy_file_path in enumerate(npy_files):
            keypoints = (np.load(npy_file_path)[0] + 1) * 100

            # Thumb tip (landmark 4) and Index finger tip (landmark 8)
            x1, y1, z1 = keypoints[4][0], keypoints[4][1], keypoints[4][2]
            x2, y2, z2 = keypoints[8][0], keypoints[8][1], keypoints[8][2]

            hand_distance = calculate_distance_3d(x1, y1, z1, x2, y2, z2)
            patient_result_file.write(f"{frame_idx} {hand_distance}\n")
            frame_distances.append(hand_distance)

    distances_output_npy_path = os.path.join(output_patient_hand_dir, f"{patient_id}_{session_state}_{hand_label}_dis.npy")
    np.save(distances_output_npy_path, np.array(frame_distances))

    peak_detector_output_txt_path = os.path.join(output_patient_hand_dir, f"{patient_id}_{session_state}_{hand_label}_peakdetector.txt")
    detect_finger_movement_peaks(frame_distances, peak_detector_output_txt_path, 0, len(frame_distances))

    # Process the peak detector output file
    peak_frames = []
    peak_amplitudes = []
    try:
        with open(peak_detector_output_txt_path, 'r') as peak_file:
            for line in peak_file:
                parts = line.strip().split(" ")
                # A valid peak line has format: <frame_idx> <amplitude> <status_code> <status_text>
                # We are interested in lines where status_code is "1" (closed peak) or "0" (opened peak)
                if len(parts) >= 3 and parts[2] in ["0", "1"]: 
                    peak_frames.append(float(parts[0]))
                    peak_amplitudes.append(float(parts[1]))
    except FileNotFoundError:
        pass
        print(f"Error: Peak detector output file not found: {peak_detector_output_txt_path}")
        return
        
    peak_times_sec = [frame / fps for frame in peak_frames]


def main(args):
    """
    Main function to orchestrate the preprocessing.
    Reads a CSV file containing video metadata (name, FPS), then processes
    each entry.
    """
    # Read the CSV file with video names and FPS
    video_metadata_df = pd.read_csv(args.csv_file_path)
    video_names = video_metadata_df['video_name'].to_list() # Expected format: e.g., "preop_on_1_right"
    video_fps_list = video_metadata_df['fps'].to_list()

    if not video_names:
        print("No video entries found in the CSV file.")
        return

    for i, video_name_str in enumerate(video_names):
        try:
            parts = video_name_str.split('_')
            if len(parts) < 4:
                print(f"Warning: Video name '{video_name_str}' has unexpected format. Skipping.")
                continue

            session_state = f"{parts[0]}_{parts[1]}"
            patient_id = parts[2]                   
            hand_label = parts[-1]                  
            current_fps = float(video_fps_list[i])

            process_patient_session(
                patient_id=patient_id,
                session_state=session_state,
                hand_label=hand_label,
                fps=current_fps,
                hand_joint_base_dir=args.hand_joint_dir,
                output_base_dir=args.output_dir
            )
        except Exception as e:
            print(f"Error processing entry '{video_name_str}': {e}")
            continue
    print("Preprocessing complete.")

# -----------------------------------------------------------------------------
# Script Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess hand landmark data for Parkinson's motion analysis.")
    parser.add_argument(
        "--csv_file_path",
        type=str,
        required=True,
        help="Path to the CSV file containing video metadata. "
             "csv must have 'video_name' and 'fps' columns."
    )
    parser.add_argument(
        "--hand_joint_dir",
        type=str,
        required=True,
        help="Base directory where hand joint NPY files are stored. "
             "Expected structure: <hand_joint_dir>/<PatientID>/<State_Condition>/<HandLabel>/*.npy"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base directory where the processed results will be saved."
    )

    cli_args = parser.parse_args()
    main(cli_args)
