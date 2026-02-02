
from cpr import *




paths = extract_multi_branch_centerlines(
    "./data/vessel_segmentation2.nii.gz",
    save_path="./centerline.npy",
    skip=5,
    max_branches=3
)

curve_planar_reformat(
    image_path="./data/ct_scan2.nii.gz",
    points_path="./centerline_path_1.npy",
    save_path="cpr_result.nii.gz",
    fov_mm=65,
    rotation_angle=[0, 45, 90, 135]
)

extract_axial_slice(
        cpr_nifti_path="cpr_result_angle0.nii.gz",  # CPR重建结果
        save_path="./cpr_result"
    )
