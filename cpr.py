import numpy as np
from scipy.ndimage import map_coordinates
from scipy.interpolate import CubicSpline
import medpy.io as mio
from skimage.morphology import skeletonize
from itertools import product
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def curve_planar_reformat(image_path, points_path, save_path, fov_mm = 50):
    print("="*40)
    print("IMAGE PATH:", image_path)
    print("POINTS PATH:", points_path)
    print("SAVE PATH:", save_path if save_path is not None else "reformatted_image.nii.gz")
    print("FIELD OF VIEW (mm):", fov_mm)
    print("="*40)

    # Step 1: Load the image and points
    
    # Medpy loads the image in this orientation : (x, y, z)
    # where, the coordinates are with respect to the patient axis
    # x -> LEFT to RIGHT (SAGITTAL slices)
    # y -> POSTERIOR to ANTERIOR (CORONAL slices)
    # z -> INFERIOR to SUPERIOR (AXIAL slices)
    
    image, h0 = mio.load(image_path) 
    points = np.load(points_path) # Array of shape (N, 3)

    pixel_spacing = h0.get_voxel_spacing()

    diff_points = points[1:] - points[:-1]
    actual_distances = np.sum(np.linalg.norm((diff_points) * np.array(pixel_spacing), axis = 1))
    pixel_distances = np.insert(np.cumsum(np.linalg.norm(diff_points, axis = 1)), 0, 0)
    total_pixel_length = pixel_distances[-1]

    print(f"Actual Length of the vessel: {actual_distances} mm")
    print(f"Pixel Length of the vessel: {total_pixel_length} pixels")

    # To use Cubic Spline interpoleation to get smoothness in the image
    spline = CubicSpline(pixel_distances, points, bc_type='natural')

    # So num_steps tell the number of output slices that we will have in the reformatted image
    # For example, if the length of the vessel was 100 pixel 
    # and needed output spacing was 2 pixels worth of 1 mm
    # that means 100 / 2 = 50 output slices
    num_steps = int(total_pixel_length)

    # So gives a distance array from 0 to total lenfth with skips
    # so somewhat like [0, 2, 4, 6, ..., total_pixel_length]
    new_dists = np.linspace(0, total_pixel_length, num_steps)

    resampled_points = spline(new_dists) # Shape (num_steps, 3)

    resampled_tangents = spline(new_dists, nu = 1)


    # Time to create the slice grid
    # So you want 50 mm field of view with 1 mm spacing, 
    # so 50 pixels on width and breadth
    pixels_per_side = int(fov_mm)
    half_size = pixels_per_side // 2

    grid_range = np.arange(-half_size, half_size + 1)

    # u_grid -> copying of the grid_range row wise
    # v_grid -> copying of the grid_range column wise
    u_grid, v_grid = np.meshgrid(grid_range, grid_range)

    u_flat = u_grid.flatten()
    v_flat = v_grid.flatten()

    output_slices = []
    prev_U = None
    # reference vector for constructing the orthogonal plane
    ref_vec = np.array([0, 1, 0]) # Z axis

    for i in range(len(resampled_points)):
        # The resampled point at which the slice is to be taken
        p_curr = resampled_points[i]

        # This tangent is the vector pointing along the vessel direction
        tangent = resampled_tangents[i]

        # Normalize the tangent
        t_norm = tangent / np.linalg.norm(tangent)

        # Compute orthogonal vectors between the tangent and reference vector
        u_vec = np.cross(t_norm, ref_vec)

        # if the u_vec is close to zero vecotr, happens only when both are parallel or coincident
        # Suppose the t_vec = [0, 1, 0] and ref_vec = [0, 1, 0]
        # Cross product will give [0, 0, 0] or undefined direction
        # So we have to use another vector for cross product
    
        if np.linalg.norm(u_vec) < 1e-6:
            # Hence the use of X axis that is [1, 0, 0]
            u_vec = np.cross(t_norm, np.array([1, 0, 0]))

        # Once we got the orthogonal vector, normalize it
        u_norm = u_vec / np.linalg.norm(u_vec)

        # IF prev_U is not None, that means it is not the first slice
        # and also the direction of the first point should be consistent and along with the same direction for the subsequent points
        if prev_U is not None:
            # To ensure smoothness of the plane orientation
            # If the current u_norm is opposite to the previous u_norm, flip it
            # Cause coss product some times leads to direction flipping
            # cant help it!!
            if np.dot(u_norm, prev_U) < 0:
                u_norm = -u_norm

        # now again if is not the starting slice
        if prev_U is not None:
            # That can be done by using this method
            # that is use 90% of the previous slice and 10% of the current slice to ensure smootheness
            u_norm = (0.9 * prev_U) + (0.1 * u_norm)
            # and NORMALIIIIZEEEE
            u_norm = u_norm / np.linalg.norm(u_norm)

        # And update with the current u_norm to prev_U
        prev_U = u_norm

        # Once done, 
        v_vec = np.cross(t_norm, u_norm)
        v_norm = v_vec / np.linalg.norm(v_vec)

        center_mm = p_curr * np.array(pixel_spacing)

        slice_coords_mm = (
            center_mm +
            np.outer(u_flat, u_norm) +
            np.outer(v_flat, v_norm)
        )

        slice_coords_pix = slice_coords_mm / np.array(pixel_spacing)
        coords_for_scipy = slice_coords_pix.T

        slice_pixels = map_coordinates(
            image,
            coords_for_scipy,
            order=1,
        )

        slice_2d = slice_pixels.reshape(len(grid_range), len(grid_range))
        output_slices.append(slice_2d)

    mio.save(np.array(output_slices), save_path if save_path is not None else "reformatted_image.nii.gz", h0)

    print(f"Saved reformatted image at {save_path if save_path is not None else 'reformatted_image.nii.gz'}")

def visualize_multi_mip(binary_data, skeleton_data):
    """
    显示三个方向的最大强度投影（MIP）

    参数:
    -----------
    binary_data : numpy.ndarray
        二值化的体积数据
    skeleton_data : numpy.ndarray
        骨架化的体积数据
    """
    import matplotlib
    from matplotlib.patches import Patch
    # 设置字体为英文字体，避免中文显示问题
    matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 计算三个方向的MIP
    # XY平面（横断面）的MIP
    mip_xy_skel = np.max(skeleton_data, axis=2)
    mip_xy_vol = np.max(binary_data, axis=2)

    # XZ平面（冠状面）的MIP
    mip_xz_skel = np.max(skeleton_data, axis=1)
    mip_xz_vol = np.max(binary_data, axis=1)

    # YZ平面（矢状面）的MIP
    mip_yz_skel = np.max(skeleton_data, axis=0)
    mip_yz_vol = np.max(binary_data, axis=0)

    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. 显示骨架的MIP
    axes[0, 0].imshow(mip_xy_skel, cmap='Reds', origin='lower')
    axes[0, 0].set_title('XY Plane - Centerline MIP (Axial)')
    axes[0, 0].set_xlabel('X (voxels)')
    axes[0, 0].set_ylabel('Y (voxels)')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].imshow(mip_xz_skel, cmap='Reds', origin='lower')
    axes[0, 1].set_title('XZ Plane - Centerline MIP (Coronal)')
    axes[0, 1].set_xlabel('X (voxels)')
    axes[0, 1].set_ylabel('Z (voxels)')
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].imshow(mip_yz_skel, cmap='Reds', origin='lower')
    axes[0, 2].set_title('YZ Plane - Centerline MIP (Sagittal)')
    axes[0, 2].set_xlabel('Y (voxels)')
    axes[0, 2].set_ylabel('Z (voxels)')
    axes[0, 2].grid(True, alpha=0.3)

    # 2. 显示叠加的MIP（骨架为红色，血管体积为绿色）
    # XY平面
    rgb_xy = np.zeros(mip_xy_vol.shape + (3,))
    rgb_xy[..., 0] = mip_xy_skel  # 红色通道：骨架
    rgb_xy[..., 1] = mip_xy_vol  # 绿色通道：原始体积
    rgb_xy[..., 2] = 0  # 蓝色通道

    axes[1, 0].imshow(rgb_xy, origin='lower')
    axes[1, 0].set_title('XY Plane - Overlay MIP\nRed: Centerline, Green: Vessel')
    axes[1, 0].set_xlabel('X (voxels)')
    axes[1, 0].set_ylabel('Y (voxels)')
    axes[1, 0].grid(True, alpha=0.3)

    # XZ平面
    rgb_xz = np.zeros(mip_xz_vol.shape + (3,))
    rgb_xz[..., 0] = mip_xz_skel  # 红色通道：骨架
    rgb_xz[..., 1] = mip_xz_vol  # 绿色通道：原始体积
    rgb_xz[..., 2] = 0  # 蓝色通道

    axes[1, 1].imshow(rgb_xz, origin='lower')
    axes[1, 1].set_title('XZ Plane - Overlay MIP\nRed: Centerline, Green: Vessel')
    axes[1, 1].set_xlabel('X (voxels)')
    axes[1, 1].set_ylabel('Z (voxels)')
    axes[1, 1].grid(True, alpha=0.3)

    # YZ平面
    rgb_yz = np.zeros(mip_yz_vol.shape + (3,))
    rgb_yz[..., 0] = mip_yz_skel  # 红色通道：骨架
    rgb_yz[..., 1] = mip_yz_vol  # 绿色通道：原始体积
    rgb_yz[..., 2] = 0  # 蓝色通道

    axes[1, 2].imshow(rgb_yz, origin='lower')
    axes[1, 2].set_title('YZ Plane - Overlay MIP\nRed: Centerline, Green: Vessel')
    axes[1, 2].set_xlabel('Y (voxels)')
    axes[1, 2].set_ylabel('Z (voxels)')
    axes[1, 2].grid(True, alpha=0.3)

    # 添加颜色说明
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Centerline'),
        Patch(facecolor='green', alpha=0.7, label='Vessel Volume')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2,
               bbox_to_anchor=(0.5, 1.02), framealpha=0.8)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # 为图例留出空间
    plt.show()

    # 显示MIP统计信息
    # print("\nMIP Statistics:")
    # print(f"XY Plane (Axial):")
    # print(f"  Centerline MIP max value: {mip_xy_skel.max()}")
    # print(f"  Vessel MIP max value: {mip_xy_vol.max()}")
    # print(f"  Centerline MIP non-zero voxels: {np.sum(mip_xy_skel > 0)}")
    # print(f"  Vessel MIP non-zero voxels: {np.sum(mip_xy_vol > 0)}")
    #
    # print(f"\nXZ Plane (Coronal):")
    # print(f"  Centerline MIP max value: {mip_xz_skel.max()}")
    # print(f"  Vessel MIP max value: {mip_xz_vol.max()}")
    # print(f"  Centerline MIP non-zero voxels: {np.sum(mip_xz_skel > 0)}")
    # print(f"  Vessel MIP non-zero voxels: {np.sum(mip_xz_vol > 0)}")
    #
    # print(f"\nYZ Plane (Sagittal):")
    # print(f"  Centerline MIP max value: {mip_yz_skel.max()}")
    # print(f"  Vessel MIP max value: {mip_yz_vol.max()}")
    # print(f"  Centerline MIP non-zero voxels: {np.sum(mip_yz_skel > 0)}")
    # print(f"  Vessel MIP non-zero voxels: {np.sum(mip_yz_vol > 0)}")

def extract_coordinates(seg_path, save_path=None, skip=10):

    print("="*40)
    print("SEGMENTATION PATH:", seg_path)
    print("SAVE_PATH:", save_path if save_path is not None else "./points.npy")
    print("SKIP FACTOR:", skip)
    print("="*40)

    # 1. Load and skeletonize
    seg, _ = mio.load(seg_path)
    seg_bin = (seg > 0)

    skeleton = skeletonize(seg_bin)
    coords = np.array(np.nonzero(skeleton)).T  # (N, 3)

    print("Total skeleton points:", coords.shape[0])
    # 可视化
    #visualize_multi_mip(seg, skeleton)

    if coords.shape[0] < 2:
        print("Not enough sementation points to extract centerline!!")
        print("Extraction failed.")
        return False

    coord_set = set(map(tuple, coords))

    neighbors = defaultdict(list)

    directions = list(product([-1, 0, 1], repeat=3))
    directions.remove((0, 0, 0))

    for c in coord_set:
        for d in directions:
            n = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
            if n in coord_set:
                neighbors[c].append(n)

    degrees = {k: len(v) for k, v in neighbors.items()}

    if any(deg > 2 for deg in degrees.values()):
        print("Warning::There is a branch in the skeleton, cannot extract a single centerline.")
        # print("Extraction failed.")
        # return False

    endpoints = [k for k, deg in degrees.items() if deg == 1]

    if len(endpoints) != 2:
        print("Warning::The skeleton does not have exactly two endpoints.")
        # print("Extraction failed.")
        # return False

    start = endpoints[0]

    ordered = []
    visited = set()

    curr = start
    prev = None

    while True:
        ordered.append(curr)
        visited.add(curr)

        next_candidates = [
            n for n in neighbors[curr]
            if n != prev
        ]

        if not next_candidates:
            break

        next_node = next_candidates[0]
        prev, curr = curr, next_node

        if curr in visited:
            break

    ordered = np.array(ordered)

    if skip > 1:
        ordered = ordered[::skip]

    np.save(save_path if save_path is not None else "points.npy", ordered)

    return ordered
