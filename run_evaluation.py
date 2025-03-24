import pandas as pd
import utils
import os

if __name__ == "__main__":
    # load json with paths
    config = utils.load_config_file('configs.json')
    dataset_label = config['dataset_label']
    images_path = config['paths']['eval_images']
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
        results.append([df['img_file'][0], TP_count, FP_count, FN_count])

    # save result
    columns = ['img_file', 'TP_count', 'FP_count', 'FN_count']

    df = pd.DataFrame(results, columns=columns)
    
    # Add a final row with the sum of each column
    total_row = ['Total', df['TP_count'].sum(), df['FP_count'].sum(), df['FN_count'].sum()]
    df.loc[len(df)] = total_row
    
    df.to_csv(f'{results_path}/{filename}', mode='w', header=True, index=False)
