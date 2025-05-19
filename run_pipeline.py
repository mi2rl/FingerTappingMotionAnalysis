import subprocess
import os
import sys
import argpase

def run_command(command_list):
    """Helper function to run a command and check for errors."""
    try:
        process = subprocess.run(command_list, check=True, text=True, capture_output=True)
        print(process.stdout)
        if process.stderr:
            print("Stderr:", process.stderr, file=sys.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing: {' '.join(command_list)}", file=sys.stderr)
        print("Return code:", e.returncode, file=sys.stderr)
        print("Stdout:", e.stdout, file=sys.stderr)
        print("Stderr:", e.stderr, file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"Error: The script '{command_list[1]}' was not found. Make sure it's in your PATH or the correct directory.", file=sys.stderr)
        return False


def main():
    # --- Configuration ---
    parser = argparse.ArgumentParser(description="Run Parkinson's FT Analysis Pipeline.")
    parser.add_argument("--master_metadata_csv", type=str, required=True,
                        help="Path to the master metadata CSV file (e.g., total_use_fps_video.csv).")
    parser.add_argument("--hand_joint_data_dir", type=str, required=True,
                        help="Base directory for hand joint data (preprocessing.py input).")
    parser.add_argument("--gt_excel", type=str, required=True,
                        help="Path to the Excel file containing ground truth data (totalfeatures.xlsx).")
    parser.add_argument("--base_work_dir", type=str, required=True,
                        help="Base directory for all output files (default: /workspace/__Done/ParkinsonFT/Output).")

    args = parser.parse_args()

    master_metadata_csv = args.master_metadata_csv
    hand_joint_data_dir = args.hand_joint_data_dir
    gt_excel = args.gt_excel
    base_work_dir = args.base_work_dir
     
    preprocessing_output_dir = os.path.join(base_work_dir, "preprocessed_output")
    extracted_features_csv = os.path.join(base_work_dir, "features_gt.csv")
    ml_results_csv = os.path.join(base_work_dir, "machine_learning_results.csv")

    python_interpreter = sys.executable 

    preprocessing_script = "preprocessing.py"
    feature_extract_script = "feature_extract.py"
    machinelearning_script = "machinelearning.py"

    # --- Pipeline Execution ---
    print("--- Starting Parkinson's FT Analysis Pipeline (Python Orchestrator) ---")

    # 디렉터리 생성
    os.makedirs(preprocessing_output_dir, exist_ok=True)
    os.makedirs(base_work_dir, exist_ok=True) 

    # Step 1: Run Preprocessing
    print("\n[1/3] Running preprocessing.py...")
    cmd_preprocess = [
        python_interpreter,
        preprocessing_script,
        "--csv_file_path", master_metadata_csv,
        "--hand_joint_dir", hand_joint_data_dir,
        "--output_dir", preprocessing_output_dir
    ]
    if not run_command(cmd_preprocess):
        print("Preprocess failed!", file=sys.stderr)
        sys.exit(1)

    # Step 2: Run Feature Extraction
    print("\n[2/3] Running feature_extract.py...")
    cmd_feature_extract = [
        python_interpreter,
        feature_extract_script,
        "--metadata_input_csv_path", master_metadata_csv,
        "--peak_files_base_dir", preprocessing_output_dir,
        "--distance_files_base_dir", preprocessing_output_dir,
        "--gt_excel_path", gt_excel,
        "--output_csv_file", extracted_features_csv
    ]
    if not run_command(cmd_feature_extract):
        print("Feature extraction failed!", file=sys.stderr)
        sys.exit(1)

    # Step 3: Run Machine Learning Tests
    print("\n[3/3] Running machinelearning.py...")
    cmd_ml = [
        python_interpreter,
        machinelearning_script,
        "--input_features_csv_file", extracted_features_csv,
        "--output_csv_file", ml_results_csv
    ]
    if not run_command(cmd_ml):
        print("Machine learning tests failed!", file=sys.stderr)
        sys.exit(1)

    print("\n--- Pipeline finished successfully! ---")
    print(f"Preprocessing outputs in: {preprocessing_output_dir}")
    print(f"Extracted features with GT in: {extracted_features_csv}")
    print(f"Machine learning results in: {ml_results_csv}")

if __name__ == "__main__":
    main()
