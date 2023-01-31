from pymatgen.core.structure import Structure
import pandas as pd
import tqdm
import os

ids_df = pd.read_csv('database/targets_MEGNet_2019.csv')
print(ids_df)
ids = ids_df['id'].to_list()
print(len(ids))

folder = "cifs"
filenames = os.listdir(folder)
filenames_without_extension = [os.path.splitext(f)[0] for f in filenames]

folder_set = set(filenames_without_extension)
ids_set = set(ids)
result = list(ids_set - folder_set)

print(result)
print(len(result))

# from mp_api.client import MPRester
# with MPRester("42J8zZHSxx98mcKn0EucWVjKZPETd9ia") as mpr:
#     # docs = mpr.summary.search(material_ids=["mp-31369"], fields=['material_id', 'pretty_formula', 'structure'])
#     docs = mpr.summary.search(material_ids=["mp-779235"], fields=['material_id', 'structure'])

from pymatgen.ext.matproj import MPRester  ## to access MP database
with MPRester("gCZ2GADLIPQQRNXUejPT") as mpr:
    docs = mpr.query(criteria={"material_id": {"$in": result}}, properties=["material_id", "structure"])

for doc in docs:
    structure = doc['structure']
    structure.to(filename='cifs/' + doc['material_id'] + '.cif', fmt='cif')
# for doc in docs:
#     mat_id = doc.material_id
#     print(mat_id)
#     structure = doc.structure
#     structure.to(filename=f'cifs/{mat_id}.cif', fmt='cif')
#     print(f"structure for {mat_id} saved")
