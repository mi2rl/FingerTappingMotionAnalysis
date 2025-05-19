import numpy as np
import pandas as pd
import scipy.ndimage
from numpy.fft import fft
import os
import argparse
import statistics 

# Code for arrhythmia analysis (including entropy, aperiodicity) and statistical evaluation was adapted from the ROC-HCI/finger-tapping-severity repository:
# Parameters have been modified to fit our current dataset.
# 'Fatigue' and 'Freeze' variables are newly implemented features.

def entropy(p):
    '''
    p: np.array of probabilities (assumed positive)
    '''
    return -(p * np.log(p)).sum()

def amplitude_entropy(values, min_val=0, max_val=19, n_buckets=18):

    '''Calculate entropy for amplitude values within specified range and buckets.'''

    dA = (max_val - min_val) / (n_buckets - 1)
    buckets = np.arange(min_val, max_val + 1, dA)
    n = np.histogram(values, buckets)[0]
    p = n / n.sum()
    p[p == 0] = 1
    lp = np.log(p)
    ppe = -np.multiply(p, lp).sum() / np.log(n_buckets)
    return ppe


def get_stats(series):
    '''Calculate basic statistics for a series.'''
    return {
        'median': np.median(series),
        'quartile_range': np.subtract(*np.percentile(series, [75, 25])),
        'min': np.min(series),
        'max': np.max(series)
    }

def custom_peaks(amp_list, time_list, fps):
    '''
    Considered peak only when the period is more than 0.15 seconds
    '''
    frame_list=[i*fps for i in time_list]
    min_period = 0.15 
    min_frame = int(min_period  * fps) 

    valid_indices = []
    previous_frame = 0  
    for i, current_frame in enumerate(frame_list):
        if i == 0 or current_frame - previous_frame >= min_frame:  
            valid_indices.append(i)  
            previous_frame = current_frame  

    result_times = [frame_list[i]/fps for i in valid_indices]
    result_amps = [amp_list[i] for i in valid_indices]

    return result_amps, result_times
    
class DistanceAnalysis:
    '''Analyze distance features.'''
    def __init__(self, l, fps):
        self.l = l
        self.fps = fps

    def speed_and_acc(self):
        '''Calculate speed and acceleration from distance.'''
        v = np.abs(np.diff(self.l)) * self.fps
        a = np.diff(v) * self.fps

        speed_s = get_stats(v)
        acc_s = get_stats(a)

        return {
            'speed_median': speed_s['median'],
            'speed_quartile_range': speed_s['quartile_range'],
            'speed_min': speed_s['min'],
            'speed_max': speed_s['max'],
            'acc_median': acc_s['median'],
            'acc_quartile_range': acc_s['quartile_range'],
            'acc_min': acc_s['min'],
            'acc_max': acc_s['max'],
        }

    def aperiodicity(self):
        '''Calculate aperiodicity using FFT.'''
        X = fft(self.l)
        power_spectrum = np.abs(X) ** 2
        power_spectrum /= power_spectrum.sum()
        return entropy(power_spectrum)

    def feature_distance(self):
        '''Aggregate distance features.'''
        return {'aperiodicity': self.aperiodicity(), **self.speed_and_acc()}

class PeakAnalysis:
    '''Analyze peak features.'''
    def __init__(self, amp_l, time_l, fps):
        self.amp_l = np.array(amp_l) 
        self.time_l = np.array(time_l)
        self.fps = fps

    def amplitude_stats(self):
        A = self.amp_l
        stats = get_stats(A)
        return {
            'amplitude_median': stats['median'],
            'amplitude_quartile_range': stats['quartile_range'],
            'amplitude_min': stats['min'],
            'amplitude_max': stats['max'],
            'amplitude_entropy': amplitude_entropy(A) 
        }

    def period(self, min_val=0, max_val=5, n_buckets=50):
        values = np.diff(self.time_l)

        ps = get_stats(values) 
        p_dict = {}
        for k in ps.keys():
            p_dict['period_' + k] = ps[k]

        dA = (max_val - min_val) / (n_buckets - 1)
        buckets = np.arange(min_val, max_val + 1, dA)
        n = np.histogram(values, buckets)[0]
        p = n / n.sum()
        p[p == 0] = 1
        lp = np.log(p)
        ppe = -np.multiply(p, lp).sum() / np.log(n_buckets)
        p_dict['period_entropy'] = ppe
        return p_dict

    def fatigue(self):
        '''
        Fatigue: Calculated when the slope is negative for 6 consecutive times.
        '''
        y_g = scipy.ndimage.gaussian_filter1d(self.amp_l, sigma=1.5) 
        y = self.amp_l
        x = self.time_l
        fps = self.fps

        fatigue_dict = {'fatigue_norm': 0, 'fatigue_frame': 0, 'fatigue_amp': 0}
        n = 5              # Minimum consecutive segment count required
        dy = np.diff(y_g)
        dx = np.diff(x)
        slopes = dy / dx  # Slope per frame of the peak values after applying the Gaussian filter

        temp_y = [] 
        temp_x = []

        for i, slope in enumerate(slopes):

            if slope < 0:
                temp_y.append(y[i])
                temp_x.append(x[i])
                
                # When fatigue is detected end of the signal
                if i == (len(slopes) - 1) and len(temp_y) > n: 
                    slope_ = (y[-1] - max(temp_y)) / (x[-1] - temp_x[np.argmax(temp_y)])
                    if slope_ < 0: 
                        slope_norm = slope_ / np.median(y) * 10
                        fatigue_dict['fatigue_norm'] = abs(slope_norm)
            
            else:
                # When fatigue is detected middle of the signal
                if len(temp_y) > n:
                    slope_ = (y[i + 1] - max(temp_y)) / (x[i + 1] - temp_x[np.argmax(temp_y)])
                    if slope_ < 0:
                        slope_norm = slope_ / np.median(y) * 10
                        fatigue_dict['fatigue_norm'] = abs(slope_norm)
                        
                        return fatigue_dict # Terminate the function once a single instance of fatigue is detected.

                temp_y = []
                temp_x = []

        return fatigue_dict

    def feature_peak(self):
        features = {}
        as_dict = self.amplitude_stats()
        p_dict = self.period()
        fatigue_dict = self.fatigue()
        features.update(as_dict)
        features.update(p_dict)
        features.update(fatigue_dict)
        return features

def freeze(time, distance, fps):
    '''
    Movement interruption during finger tapping
    '''
    # 3 contions 
    freeze_period_threshold = 1.4 # Condition1 The period during which freeze occurs must be 1.4 times longer than the median of period.
    freeze_time_threshold = 0.2   # Condition2 The period during which freeze occurs must be at least 0.2 seconds in duration.
    freeze_slope_threshold = 0.90 # Condition3 During the period of a freeze, the speed must drop to 0.9 times lower.

    frames = [int(t * fps) for t in time]
    frame_diffs = np.diff(frames)
    long_frame_diff = np.median(frame_diffs) * freeze_period_threshold 

    slopes = [abs(dis) for dis in np.diff(distance) * fps]
    small_slope_threshold = np.mean(slopes) * freeze_slope_threshold

    freeze_count = 0
    freeze_starts = []
    freeze_durations = []

    for i, frame_diff in enumerate(frame_diffs):
        if frame_diff > long_frame_diff: # condition1
            start = frames[i]
            end = frames[i + 1] if i + 1 < len(frames) else frames[i]

            if (end - start) / fps > freeze_time_threshold and start > fps: # condition2 
                segment_slopes = slopes[start:end]
                mean_slope = np.mean(segment_slopes) if segment_slopes else 0

                if mean_slope < small_slope_threshold: # condition 3 
                    freeze_count += 1
                    freeze_starts.append(start)
                    freeze_durations.append(round((end - start) / fps, 2))

    return {"freeze_durations": sum(freeze_durations)}

def run_feature_extraction(fps_info_csv_path, peak_files_base_dir, distance_files_base_dir, gt_excel_path,output_csv_file):
    """
    Main function to extract features.
    Reads metadata, processes peak and distance files, and saves features to Excel.
    """
    df_md = pd.read_csv(fps_info_csv_path)

    total_list = df_md['video_name'].tolist()
    total_fps_list = df_md['fps'].tolist()
    
    try: 
        total_features_list = []
        for ij, video_name in enumerate(total_list):
            feature_dict = {}
            patient, state1, state2, hand = video_name.split('_')  

            feature_dict['video_name'] = f'{patient}_{state1}_{state2}_{hand}'
            fps = round(total_fps_list[ij])

            peak_txt_path = os.path.join(peak_files_base_dir,f'{patient}_{hand}/{patient}_{state1}_{state2}_{hand}_peakdetector.txt')
            distance_path =os.path.join(distance_files_base_dir,f'{patient}_{hand}/{patient}_{state1}_{state2}_{hand}.txt')

            peaktime_list, peakamp_list = [], []
            with open(peak_txt_path, 'r') as peak_f:
                for peak_data in peak_f:
                    time, amp, flag1, flag2 = peak_data.split()
                    if flag1 == "1":
                        peaktime_list.append(float(time) / fps)
                        peakamp_list.append(float(amp))
                
            c_peakamp_list, c_peaktime_list = custom_peaks(peakamp_list, peaktime_list, fps)
            c_peakamp_list, c_peaktime_list = c_peakamp_list[:10], c_peaktime_list[:10]

            peakanalyzer = PeakAnalysis(c_peakamp_list, c_peaktime_list, fps)
            peak_dict = peakanalyzer.feature_peak()
            feature_dict.update(peak_dict)

            distance_list = np.loadtxt(distance_path, usecols=1)
            videoframe = int(float(c_peaktime_list[-1]) * fps) if c_peaktime_list else 0
            distance_list = distance_list[:videoframe] if len(distance_list) > videoframe else distance_list

            distanceanalyzer = DistanceAnalysis(distance_list, fps)
            distance_dict = distanceanalyzer.feature_distance()
            feature_dict.update(distance_dict)
            
            freeze_dict = freeze(c_peaktime_list, distance_list, fps)
            feature_dict.update(freeze_dict)

            total_features_list.append(feature_dict)
    except Exception as e:
        print(f"Error: Processing video_name : {video_name}  - {e}")
        pass

    totalfeature_df = pd.DataFrame(total_features_list)

    gt_df = pd.read_excel(gt_excel_path)
    gt_df = gt_df[['video_name', 'GT']]

    totalfeature_gt_df = pd.merge(totalfeature_df, gt_df, on='video_name', how='left')

    desired_columns = [
        'video_name',  'GT',
        'aperiodicity',
        'speed_median', 'speed_quartile_range', 'speed_min', 'speed_max',
        'acc_median', 'acc_quartile_range', 'acc_min', 'acc_max',
        'freeze_durations',
        'amplitude_median', 'amplitude_quartile_range', 'amplitude_min', 'amplitude_max', 'amplitude_entropy',
        'period_median', 'period_quartile_range', 'period_min', 'period_max', 'period_entropy',
        'fatigue_norm'
    ]

    final_df = totalfeature_gt_df[desired_columns]

    try:
        final_df.to_csv(output_csv_file, index=False)
        print(f"Successfully saved features to {output_csv_file}")
    except Exception as e:
        print(f"Error saving features to csv {output_csv_file}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from preprocessed Parkinson's finger tapping data.")
    parser.add_argument("--metadata_input_csv_path", required=True, help="Path to the input CSV file containing video names and FPS.") # Removed GT and video_group, as they seem to come from gt_excel_path
    parser.add_argument("--peak_files_base_dir", required=True, help="Base directory where _peakdetector.txt files are stored.")
    parser.add_argument("--distance_files_base_dir", required=True, help="Base directory where distance .txt files are stored.")
    parser.add_argument("--gt_excel_path", required=True, help="Path to the Excel file containing GT information.") # Clarified purpose
    parser.add_argument("--output_csv_file", required=True, help="Path to save the extracted features in CSV format.")

    args = parser.parse_args()

    run_feature_extraction(
        args.metadata_input_csv_path,
        args.peak_files_base_dir,
        args.distance_files_base_dir,
        args.gt_excel_path,
        args.output_csv_file
    )
