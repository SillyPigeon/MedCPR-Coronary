
from cpr import *





#extract_coordinates("./data/vessel_segmentation.nii.gz", "./points.npy", skip=10)
#curve_planar_reformat("./data/ct_scan.nii.gz", "./points.npy", "./cpr_output.nii.gz", 60)

paths = extract_multi_branch_centerlines(
    "./data/vessel_segmentation2.nii.gz",
    save_path="./centerline.npy",
    skip=5,
    max_branches=3
)
#
# for i, path_points in enumerate(paths):
#     curve_planar_reformat(
#         image_path="./data/ct_scan2.nii.gz",
#         points_path=f"./centerline_path_{i+1}.npy",
#         save_path=f"./cpr_result_{i+1}.nii.gz"
#     )

curve_planar_reformat(
    image_path="./data/ct_scan2.nii.gz",
    points_path="./centerline_path_3.npy",
    save_path="cpr_result.nii.gz",
    fov_mm=65,
    rotation_angle=[0, 30, 90, 150, 360]  # 生成0°, 45°, 90°, 135°
)
