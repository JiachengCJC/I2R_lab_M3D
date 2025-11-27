import os
import numpy as np
from PIL import Image
import concurrent.futures
from tqdm import tqdm
from collections import Counter
import unicodedata
import monai.transforms as mtf
from multiprocessing import Pool
from unidecode import unidecode

# input_dir = 'PATH/M3D_Cap/ct_quizze/'
# output_dir = 'PATH/M3D_Cap_npy/ct_quizze/'

input_dir = 'PATH/M3D_Cap/ct_case/'
output_dir = 'PATH/M3D_Cap_npy/ct_case/'

# Get all subfolders [00001, 00002....]
# make sure the script will process each subfolder individually, (how many subfolders are there)
subfolders = [folder for folder in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, folder))]

# the monai transform to crop the foreground and resize to 32x256x256
transform = mtf.Compose([
    mtf.CropForeground(),
    mtf.Resize(spatial_size=[32, 256, 256], mode="bilinear")
])

"""Example of the text file content:
Title:Quiz 175316


presentation:This patient came for a CT scan due to left hemifacial pain with no other associated symptoms. 


patient:Age:60 years
Gender:Female


discussion:
Paranasal sinus osteomas are benign, slowly growing bone tumors that are usually asymptomatic and most often occur in the frontal and ethmoid sinuses 1-4. This case demonstrates the typical radiological features of frontal sinus osteoma.



study_findings:
Computed tomography (CT) scans demonstrated the following:
there is a well-defined bone density mass lesion within the left frontal sinus that approaches the sinus edges. Most of the mass is equivalent to the bone cortex, while the superomedial area shows a ground-glass density. The dimensions of the lesion measure 1.8 x 1.8 x 1.6 cm.
the inflammatory material within the left frontal sinus and spheroidal sinuses indicates rhinosinusitis.
the increased thickness of the mucosa and evidence of polyposis in both maxillary sinuses suggest sinusitis.

Impression:
The radiologic findings are consistent with left frontal osteoma and generalised rhinosinusitis.
"""
def process_subfolder(subfolder):
    output_id_folder = os.path.join(output_dir, subfolder)
    input_id_folder = os.path.join(input_dir, subfolder)

    os.makedirs(output_id_folder, exist_ok=True)

    for subsubfolder in os.listdir(input_id_folder):
        if subsubfolder.endswith('.txt'): # check if the file is a text file, if yes, process the text file, if no, process the image folder
            text_path = os.path.join(input_dir, subfolder, subsubfolder)
            with open(text_path, 'r') as file:
                text_content = file.read() # read the whole text content

            search_text = "study_findings:"
            index = text_content.find(search_text)

            if index != -1: # if the specified string is found, "study_findings"
                filtered_text = text_content[index + len(search_text):].replace("\n", " ").strip() # extract the content after the specified string
            else:
                print("Specified string not found")
                filtered_text = text_content.replace("\n", " ").strip()

            # if the filtered text is still too short, try to find "discussion:"
            if len(filtered_text.replace("\n", "").replace(" ", "")) < 5:
                search_text = "discussion:"
                index = text_content.find(search_text)
                if index != -1:
                    filtered_text = text_content[index + len(search_text):].replace("\n", " ").strip()
                else:
                    print("Specified string not found")
                    filtered_text = text_content.replace("\n", " ").strip()

            # If still too short, fallback → use the whole text.
            if len(filtered_text.replace("\n", "").replace(" ", "")) < 5:
                filtered_text = text_content.replace("\n", " ").strip()

            # write the filtered text to a new text file in the output directory
            new_text_path = os.path.join(output_dir, subfolder, subsubfolder)
            with open(new_text_path, 'w') as new_file:
                new_file.write(filtered_text)

        subsubfolder_path = os.path.join(input_dir, subfolder, subsubfolder)

        # when the subsubfolder is actually a folder (not a text file), process the images inside
        if os.path.isdir(subsubfolder_path):
            subsubfolder = unidecode(subsubfolder) # "Pöschl" -> Poschl, clean the folder name
            output_path = os.path.join(output_dir, subfolder, f'{subsubfolder}.npy')

            # gather all image files in the folder
            image_files = [file for file in os.listdir(subsubfolder_path) if
                           file.endswith('.jpeg') or file.endswith('.png')]

            if len(image_files) == 0:
                continue

            image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

            images_3d = []
            for image_file in image_files:
                image_path = os.path.join(subsubfolder_path, image_file)
                try:
                    img = Image.open(image_path)
                    img = img.convert("L") # convert image to grayscale
                    img_array = np.array(img) # convert image to numpy array
                    # normalization
                    img_array = img_array.astype(np.float32) / 255.0
                    images_3d.append(img_array[None])
                except:
                    print("This image is error: ", image_path)

            # filter out images that do not match the most common shape
            images_3d_pure = []
            try:
                img_shapes = [img.shape for img in images_3d]
                item_counts = Counter(img_shapes)
                most_common_shape = item_counts.most_common(1)[0][0]
                for img in images_3d:
                    if img.shape == most_common_shape:
                        images_3d_pure.append(img)
                final_3d_image = np.vstack(images_3d_pure)

                image = final_3d_image[np.newaxis, ...]

                image = image - image.min()
                image = image / np.clip(image.max(), a_min=1e-8, a_max=None)

                img_trans = transform(image)

                np.save(output_path, img_trans)
            except:
                print([img.shape for img in images_3d])
                print("This folder is vstack error: ", output_path)



with Pool(processes=32) as pool:
    with tqdm(total=len(subfolders), desc="Processing") as pbar:
        for _ in pool.imap_unordered(process_subfolder, subfolders):
            pbar.update(1)

''' Input example:
ct_case/
 ├── 00001/
 │    ├── Pöschl/              <--- folder with many JPEG slices
 │    │     ├── 0001.jpeg
 │    │     ├── 0002.jpeg
 │    │     ├── 0003.jpeg
 │    │     └── ...
 │    ├── meta.txt             <--- patient description
 │
 ├── 00002/
 │    ├── Axial/
 │    │     ├── 0001.jpeg
 │    │     ├── ...
 │    ├── info.txt
 │
 └── 00003/
      ├── Coronal/
      ├── notes.txt
'''

''' Output example:
ct_case_npy/
 ├── 00001/
 │    ├── meta.txt               <--- cleaned text extracted from input text
 │    ├── Poschl.npy             <--- processed 3D CT volume
 │
 ├── 00002/
 │    ├── info.txt
 │    ├── Axial.npy
 │
 └── 00003/
      ├── notes.txt
      ├── Coronal.npy
'''