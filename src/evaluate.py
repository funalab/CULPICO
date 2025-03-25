# evaluate Dice, Hausdorff distance, average surface distance
# seg-metrics==1.2.8 scikit-image==0.24.0 required
import os
import json
import statistics
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool
from glob import glob
from tqdm import tqdm
from skimage import io
import seg_metrics.seg_metrics as sg


def process(inputs):
    img_path, metrics_list = inputs
    img_predict = io.imread(img_path)
    img_ground_truth = io.imread(img_path.replace('predict.tif', 'ground_truth.tif'))

    img_predict = np.where(img_predict == 0, 0, 1)
    img_ground_truth = np.where(img_ground_truth == 0, 0, 1)

    # metrics
    metrics = sg.write_metrics(labels=[1],
                               gdth_img=img_ground_truth,
                               pred_img=img_predict,
                               csv_file=None,
                               metrics=metrics_list)
    return metrics, img_path


def main(args):
    process_num = args.process_num
    root_path = args.data_root
    print(f'[Data root path] {root_path}')
    print('*' * 100)
    dir_path_list = glob(f"{root_path}/*")
    save_root = f'{root_path}/all_metrics'
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    metrics_list = ['dice', 'hd95', 'msd', 'jaccard']
    data_list = []
    for dir_path in tqdm(dir_path_list):

        img_path_list = glob(f'{dir_path}/test/imgs/*/predict.tif')

        if len(img_path_list) > 0:
            save_dir = f'{dir_path}/metrics'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            header = {'file_path': None}
            result = {key: [] for key in metrics_list}
            result = dict(**result, **header)

            inputs = [[img_path, metrics_list] for img_path in img_path_list]
            with Pool(processes=process_num) as p:
                for metrics, file_path in p.imap_unordered(process, inputs):
                    result['file_path'] = file_path
                    for key in metrics_list:
                        result[key].append(float(metrics[0][key][0]))

            with open(f'{save_dir}/metrics.json', 'w') as f:
                json.dump(result, f)

            # stats
            result_statics = {key: {'ave': 0, 'std': 0} for key in metrics_list}
            for key in metrics_list:
                result_statics[key]['ave'] = statistics.mean(result[key])
                result_statics[key]['std'] = statistics.stdev(result[key])

            with open(f'{save_dir}/metrics_stats.json', 'w') as f:
                json.dump(result_statics, f)

            dir_name = os.path.basename(dir_path)
            mode = dir_name[:dir_name.find('_')]
            source = dir_name[dir_name.find('source_') + len('source_'):dir_name.find('_target')]
            target = dir_name[dir_name.find('target_') + len('target_'):]

            attr = {'mode': mode, 'source': source, 'target': target}

            stats = {}
            for key in metrics_list:
                stats[f'{key}_ave'] = result_statics[key]['ave']
                stats[f'{key}_std'] = result_statics[key]['std']

            data = dict(**attr, **stats)

            data_list.append(data)

    df = pd.DataFrame(data_list)
    df.to_csv(f'{save_root}/all_metrics.csv', index=False)
    print('*' * 100)
    print('Result')
    print('*' * 100)
    print(df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, help='Root path of the image data to be evaluated')
    parser.add_argument('--process_num', type=int, default=10, help='multiprocess number')
    args = parser.parse_args()

    main(args)