import pandas as pd

def get_pretty_formula(threshold):
    path_mp_csv = './predictions/01_21-59/test_predictions_mp_onehot_MP_8_12_100_.csv'
    path_sc_csv = './predictions/01_21-59/test_predictions_mp_onehot_SC_8_12_100_.csv'
    path_combined_cifs_csv = './combined_cifs.csv'

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
    df_threshold = df_threshold.drop(columns=['prediction'])
    print(df_threshold)
    material_ids = df_threshold['name'].tolist()
    matching_formulas = []
    from mp_api.client import MPRester
    with MPRester("42J8zZHSxx98mcKn0EucWVjKZPETd9ia") as mpr:
        docs = mpr.summary.search(material_ids=material_ids, fields=['material_id', 'formula_pretty'])
    for doc in docs:
        mat_id = doc.material_id
        formula = doc.formula_pretty
        matching_formulas.append((mat_id, formula))
    df_formulas = pd.DataFrame(matching_formulas, columns=['name', 'pretty_formula'])
    #match df_threshold and formulas_df by name column
    df_threshold = df_threshold.merge(df_formulas, on='name', how='left')
    df_combined = pd.read_csv(path_combined_cifs_csv)
    df_sc_ordered = pd.read_csv("C:/Users/axema/icsg3d/data/supercon_ordered/supercon_ordered.csv", usecols=['id', 'pretty_formula'])
    df_sc_ordered = df_sc_ordered.rename(columns={'id': 'name'})
    #combine formulas of df_threshold and df_sc_ordered
    df_sc_threshold = pd.concat([df_threshold, df_sc_ordered])
    df_combined = df_combined.merge(df_sc_threshold, on='name', how='left')
    print(df_combined)
    df_combined.to_csv("combined.csv", index=False)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Crystal Graph Coordinator.')
    parser.add_argument("--threshold", type=float, default=15)
    options = parser.parse_args()
    get_pretty_formula(options.threshold)