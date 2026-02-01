
from cpr import extract_coordinates, curve_planar_reformat, extract_multi_branch_centerlines





#extract_coordinates("./data/vessel_segmentation.nii.gz", "./points.npy", skip=10)

paths = extract_multi_branch_centerlines(
    "./data/vessel_segmentation.nii.gz",
    save_path="./centerlines.npy",
    skip=5,
    max_branches=3
)

#curve_planar_reformat("./data/ct_scan.nii.gz", "./points.npy", "./cpr_output.nii.gz", 60)
