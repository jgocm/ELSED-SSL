import pyelsed
from elsed_analyzer import SegmentsAnalyzer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

def merge_images_results(dataset_label):
    results_path = f'evaluation/{dataset_label}/results'
    filename = f'results_{dataset_label}.csv'
    
    # load images and thresholds
    csv_files = [f for f in os.listdir(results_path) if (f.endswith('csv') and f != filename)]

    results = []
    for csv_file in csv_files:
        csv_path = os.path.join(results_path, csv_file)
        df = pd.read_csv(csv_path)

        TP_count = (df['is_field_boundary'] & df['is_field_boundary_gt']).sum() + \
                   (df['is_field_marking'] & df['is_field_marking_gt']).sum()
        FP_count = (df['is_field_boundary'] & ~df['is_field_boundary_gt']).sum() + \
                   (df['is_field_marking'] & ~df['is_field_marking_gt']).sum()
        FN_count = (~df['is_field_boundary'] & df['is_field_boundary_gt']).sum() + \
                   (~df['is_field_marking'] & df['is_field_marking_gt']).sum()
        if len(df)>0:
            results.append([df['img_file'][0], TP_count, FP_count, FN_count])

    # save result
    columns = ['img_file', 'TP_count', 'FP_count', 'FN_count']

    df = pd.DataFrame(results, columns=columns)
    
    # Add a final row with the sum of each column
    total_row = ['Total', df['TP_count'].sum(), df['FP_count'].sum(), df['FN_count'].sum()]
    df.loc[len(df)] = total_row
    
    df.to_csv(f'{results_path}/{filename}', mode='w', header=True, index=False)

    TP_count, FP_count, FN_count = total_row[1], total_row[2], total_row[3]
    return total_row

def compute_all_results(dataset_labels):
    
    results = []
    for dataset_label in dataset_labels:
        result = merge_images_results(dataset_label)
        TP_count, FP_count, FN_count = result[1], result[2], result[3]
        precision = TP_count/(TP_count + FP_count)        
        recall = TP_count/(TP_count + FN_count)
        nr_of_lines = TP_count + FP_count + FN_count
        results.append([dataset_label, precision, recall, nr_of_lines])
        print(dataset_label, f'{precision:.3f}', recall, nr_of_lines)
    
    columns = ['dataset', 'precision', 'recall', 'nr of lines']
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(f'elsed_results.csv', mode='w', header=True, index=False)

def count_training_images(dataset_label):
    annotations_path = f'trainings/{dataset_label}/segments_annotations.csv'
    df = pd.read_csv(annotations_path)
    img_files = np.array(df['img_path'])
    nr_of_images = len(np.unique(img_files))
   
    return nr_of_images

def evaluate(marking_thresholds, boundary_thresholds, test_data):

    analyzer = SegmentsAnalyzer(pyelsed, 
                                marking_thresholds=marking_thresholds, 
                                boundary_thresholds=boundary_thresholds)

    precision, recall, TP_count, FP_count, FN_count = 0, 0, 0, 0, 0

    for index, row in test_data.iterrows():
        gx = np.array([row['grad_Bx'], row['grad_Gx'], row['grad_Rx']], dtype=np.float32)
        gy = np.array([row['grad_By'], row['grad_Gy'], row['grad_Ry']], dtype=np.float32)
        segment_length = row['segment_length']
        is_field_marking_gt = row['is_field_marking']
        is_field_boundary_gt = row['is_field_boundary']
        
        label = analyzer.classify(gx, -gy, segment_length)
        
        is_field_boundary = (label==1)
        is_field_marking = (label==2)
        
        is_true_positive = (is_field_marking and is_field_marking_gt) or (is_field_boundary and is_field_boundary_gt)
        is_false_positive = (is_field_marking and not is_field_marking_gt) or (is_field_boundary and not is_field_boundary_gt)
        is_false_negative = (not is_field_marking and is_field_marking_gt) or (not is_field_boundary and is_field_boundary_gt)

        
        if is_true_positive:
            TP_count += 1
            
        if is_false_positive: 
            FP_count += 1

        if is_false_negative: 
            FN_count += 1

        if TP_count+FP_count > 0:
            precision = TP_count/(TP_count+FP_count)

        if TP_count+FN_count > 0:
            recall = TP_count/(TP_count+FN_count)
        
    return precision, recall, TP_count, FP_count, FN_count, len(test_data)

def run_cross_evaluation(dataset_labels):
    for thresholds_label in dataset_labels:
        for dataset_label in dataset_labels:
            # load csv
            annotations_path = f"trainings/{dataset_label}/segments_annotations.csv"
            df = pd.read_csv(annotations_path)
            _, test_df = train_test_split(df, train_size=0.5, random_state=42)

            # load thresholds
            marking_thresholds_path = f"trainings/{thresholds_label}/marking_thresholds_30.npy"
            boundary_thresholds_path = f"trainings/{thresholds_label}/boundary_thresholds_30.npy"
            marking_thresholds = np.load(marking_thresholds_path)
            boundary_thresholds = np.load(boundary_thresholds_path)

            # run evaluation
            result = evaluate(marking_thresholds, boundary_thresholds, test_df)
            precision = result[0]
            print(thresholds_label, dataset_label, precision)

if __name__ == "__main__":
    dataset_labels = ['lines_15pm_lights_off_window_open',
                      'lines_15pm_lights_on_window_open',
                      'lines_15pm_lights_on_window_closed']
    
    run_cross_evaluation(dataset_labels)


    
