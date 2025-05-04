import pyelsed
import utils
import numpy as np
import pandas as pd
from elsed_analyzer import SegmentsAnalyzer
from pyswarm import pso
from sklearn.model_selection import train_test_split

def evaluate(thresholds, test_data):

    analyzer = SegmentsAnalyzer(pyelsed, marking_thresholds=thresholds)

    precision, recall, TP_count, FP_count, FN_count = 0, 0, 0, 0, 0

    for index, row in test_data.iterrows():
        gx = np.array([row['grad_Bx'], row['grad_Gx'], row['grad_Rx']], dtype=np.float32)
        gy = np.array([row['grad_By'], row['grad_Gy'], row['grad_Ry']], dtype=np.float32)
        segment_length = row['segment_length']
        is_field_marking_gt = row['is_field_marking']
        
        label = analyzer.classify(gx, -gy, segment_length)
        
        is_field_marking = (label==2)
        
        is_true_positive = (is_field_marking and is_field_marking_gt)
        is_false_positive = (is_field_marking and not is_field_marking_gt)
        is_false_negative = (not is_field_marking and is_field_marking_gt)

        
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
        
# Define the function to calculate mAP given a set of thresholds
def calculate_score(thresholds, train_data):
    
    analyzer = SegmentsAnalyzer(pyelsed, marking_thresholds=thresholds)
    
    TP_count, FP_count = 0, 0

    for index, row in train_data.iterrows():
        gx = np.array([row['grad_Bx'], row['grad_Gx'], row['grad_Rx']], dtype=np.float32)
        gy = np.array([row['grad_By'], row['grad_Gy'], row['grad_Ry']], dtype=np.float32)
        segment_length = row['segment_length']
        is_field_marking_gt = row['is_field_marking']
        
        label = analyzer.classify(gx, -gy, segment_length)
        
        is_field_marking = (label==2)
        
        is_true_positive = (is_field_marking and is_field_marking_gt)
        is_false_positive = (is_field_marking and not is_field_marking_gt)
        
        if is_true_positive:
            TP_count += 1
            
        if is_false_positive: 
            FP_count += 1
            
    #print(f'thresholds: {thresholds} | TP: {TP_count} | FP: {FP_count} | Score: {TP_count - FP_count}')

    return -(TP_count - FP_count)

def train_and_evaluate():
    # Define the bounds for the thresholds
    lb = [10, 20, 50]  # Lower bounds
    ub = [255, 70, 300]  # Upper bounds

    # Load dataset
    configs = utils.load_config_file("configs.json")
    annotations_path = configs["paths"]["segments_annotations"]
    dataset_label = configs["dataset_label"]
    df = pd.read_csv(annotations_path)

    # Define the min, max percentages and the step for the train set
    min_percentage = 0.1
    max_percentage = 0.7
    step = 0.1

    # PSO parameters
    swarmsize = 100     # Number of particles in the swarm
    maxiter = 10        # Max number of iterations
    omega = 0.1         # Particles' velocity

    # Use numpy to generate train sizes from min to max with the given step
    train_sizes = np.arange(min_percentage, max_percentage + step, step)

    thresholds_list = []
    results_list = []
    test_df = None

    # Train thresholds for different train set sizes
    for idx, train_size in enumerate(train_sizes):
        thresholds_path = f'trainings/{dataset_label}/marking_thresholds_{int(100*train_size)}.npy'

        train_df, test_df = train_test_split(df, train_size=train_size, random_state=42)
        
        # Run PSO to optimize the thresholds
        optimal_thresholds, optimal_score = pso(calculate_score, lb, ub, args=(train_df,), swarmsize=swarmsize, maxiter=maxiter, omega=omega)

        thresholds_list.append(optimal_thresholds)
        results_list.append(optimal_score)

        # Print the optimal thresholds and the corresponding score in the training set
        print(f'Train Size: {int(100*train_size)}% | Thresholds: {optimal_thresholds}  | Score: {-optimal_score}')

        #np.save(thresholds_path, optimal_thresholds)
        
    # Evaluate thresholds on the test set
    for idx, (train_size, thresholds) in enumerate(zip(train_sizes, thresholds_list)):
        
        precision, recall, TP_count, FP_count, FN_count, total_lines = evaluate(thresholds, test_df)

        print(f'Train Size {int(100*train_size)}%: Precision: {precision:.3f} | Recall: {recall:.3f} | TP: {TP_count} | FP: {FP_count} | FN: {FN_count} | total lines: {total_lines}')

if __name__ == "__main__":
    train_and_evaluate()
    


