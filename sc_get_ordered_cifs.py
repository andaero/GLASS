import os
import shutil
from pymatgen.core.structure import Structure

src_folder = "./database/sc_cifs"
dst_folder = "./database/sc_cifs_ordered"

# Create the destination folder if it doesn't exist
if not os.path.exists(dst_folder):
    os.mkdir(dst_folder)

# Loop through the files in the source folder
for filename in os.listdir(src_folder):
    # Load the crystal structure from the file
    structure = Structure.from_file(os.path.join(src_folder, filename))
    # Check if the structure is ordered
    if structure.is_ordered:
        # Copy the file to the destination folder
        print(f"Copying {filename} to {dst_folder}")
        shutil.copy2(os.path.join(src_folder, filename), dst_folder)