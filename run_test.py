
from cpr import extract_coordinates, curve_planar_reformat





extract_coordinates("./data/vessel_segmentation.nii.gz", "./points.npy", skip=10)

curve_planar_reformat("./data/ct_scan.nii.gz", "./points.npy", "./cpr_output.nii.gz", 60)
