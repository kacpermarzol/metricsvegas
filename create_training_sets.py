import os
import nibabel as nib
import numpy as np
from PIL import Image

def save_slices(nii_path, output_dir):
    img = nib.load(nii_path)
    data = img.get_fdata()

    #jezeli ma więcej niz 3 kanaly:
    #data = np.sum(data, axis=-1)

    #jezeli w .nii sa maski:
    data = (data>0).astype(np.uint8)

    os.makedirs(output_dir, exist_ok=True)
    num_slices = data.shape[0]
    print(data.shape)
    for i in range(num_slices):
        slice_img = data[i, :, :] #* 255
        img_pil = Image.fromarray(slice_img.astype(np.uint8))
        img_pil.save(os.path.join(output_dir, f"{i:04d}.png"))

    print(f"Saved {num_slices} slices to: {output_dir}")

def batch_convert_us_labels(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for fname in sorted(os.listdir(input_dir)):
        if fname.endswith(".nii.gz"):
            name = os.path.splitext(os.path.splitext(fname)[0])[0]
            nii_path = os.path.join(input_dir, fname)
            out_dir = os.path.join(output_dir, name)
            save_slices(nii_path, out_dir)

if __name__ == "__main__":
    batch_convert_us_labels("prostate/prostate_dset/val/us_images",
                            "prostate/prostate_dset_img")
