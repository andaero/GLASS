import pandas as pd
import numpy as np
import os
import glob

def combine_datasets(threshold, new_path):
    path_sc = './database/sc_cifs_ordered'
    path_cif = './database/cif'
    path_mp_csv = './predictions/01_21-59/test_predictions_mp_onehot_MP_8_12_100_.csv'
    path_sc_csv = './predictions/01_21-59/test_predictions_mp_onehot_SC_8_12_100_.csv'

    df_mp = pd.read_csv(path_mp_csv)
    df_sc = pd.read_csv(path_sc_csv)
    df_threshold = df_mp[df_mp["prediction"] > threshold]
    print(df_threshold)
    #removing all values in df_threshold that are in df_sc
    same_names = df_sc[df_sc['name'].str.contains('|'.join(df_threshold['name']))]['name'].tolist()
    #split all values of list by '-' and only include the last value
    same_names = [x.split('-')[-1] for x in same_names if x.split('-')[-1] != 'synth_doped']

    #only include the rows in df_threshold where the names of the rows in df_threshold do not equal the names in same_names
    df_threshold = df_threshold[~df_threshold['name'].str.contains('|'.join(same_names))]
    print(df_threshold)
    print(df_sc)
    #create new folder that contains cif files of all the names in df_threshold
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    print("FOR NEW CIFS IN THRESHOLD")
    for name in df_threshold['name']:
        #find the path of the cif file
        path = glob.glob(f'{path_cif}/**/{name}.cif', recursive=True)
        #check if the file was already copied before
        if os.path.exists(f'{new_path}/{name}.cif'):
            print(f'{name} already exists')
            continue
        #copy the cif file to the new folder
        os.system(f'cp {path[0]} {new_path}')
    # print("FOR CIFS IN SC")
    # for name in df_sc['name']:
    #     #find the path of the cif file
    #     path = glob.glob(f'{path_sc}/**/{name}.cif', recursive=True)
    #     #copy the cif file to the new folder
    #     os.system(f'cp {path[0]} {new_path}')
    #concat df_threshold and df_sc
    df = pd.concat([df_threshold, df_sc])
    #export dataframe to csv
    df.to_csv('combined_cifs.csv', index=False)
    print(df)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Crystal Graph Coordinator.')
    parser.add_argument("--threshold", type=float, default=15)
    parser.add_argument("--new_path", type=str, default='./combined_cifs')
    options = parser.parse_args()
    combine_datasets(options.threshold, options.new_path)





