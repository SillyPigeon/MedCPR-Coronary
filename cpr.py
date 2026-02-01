import numpy as np
from scipy.ndimage import map_coordinates
from scipy.interpolate import CubicSpline
import medpy.io as mio
from skimage.morphology import skeletonize
from itertools import product
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx

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


def extract_multi_branch_centerlines(seg_path, save_path=None, skip=10, max_branches=3):
    """
    提取多分支中心线，保存最长的若干条路径

    参数:
    -----------
    seg_path : str
        分割图像的路径
    save_path : str or None
        保存点集的路径
    skip : int
        采样间隔
    max_branches : int
        最大保留的路径数量

    返回:
    -----------
    List[np.ndarray]
        提取的中心线路径列表，按长度排序
    """
    print("=" * 40)
    print("SEGMENTATION PATH:", seg_path)
    print("SAVE_PATH:", save_path if save_path is not None else "./points.npy")
    print("SKIP FACTOR:", skip)
    print("MAX BRANCHES:", max_branches)
    print("=" * 40)

    # 1. 加载并骨架化
    seg, _ = mio.load(seg_path)
    seg_bin = (seg > 0).astype(np.uint8)

    skeleton = skeletonize(seg_bin)
    coords = np.array(np.nonzero(skeleton)).T  # (N, 3)

    print(f"Total skeleton points: {coords.shape[0]}")

    # 可视化
    #visualize_multi_mip(seg, skeleton)

    if coords.shape[0] < 2:
        print("Not enough skeleton points to extract centerline!")
        return []

    # 2. 构建图结构
    coord_set = set(map(tuple, coords))
    G = nx.Graph()
    G.add_nodes_from(coord_set)

    # 定义26邻域
    directions = list(product([-1, 0, 1], repeat=3))
    directions.remove((0, 0, 0))

    for node in coord_set:
        for d in directions:
            neighbor = (node[0] + d[0], node[1] + d[1], node[2] + d[2])
            if neighbor in coord_set:
                G.add_edge(node, neighbor)

    print(f"Graph nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

    # 3. 找到所有端点（度数为1的节点）
    endpoints = [node for node, degree in dict(G.degree()).items() if degree == 1]
    print(f"Found {len(endpoints)} endpoints")

    if len(endpoints) < 2:
        print("Not enough endpoints to form a path!")
        return []

    # 4. 使用BFS找到所有可能的路径
    all_paths = []

    for start in endpoints:
        for end in endpoints:
            if start != end:
                try:
                    path = nx.shortest_path(G, start, end)
                    if len(path) > 2:  # 排除非常短的路径
                        all_paths.append(path)
                except nx.NetworkXNoPath:
                    continue

    # 5. 去重和排序
    unique_paths = []
    seen_paths = set()

    for path in all_paths:
        # 转换为元组并排序以确保方向一致性
        path_tuple = tuple(sorted([path[0], path[-1]]))
        if path_tuple not in seen_paths:
            seen_paths.add(path_tuple)
            unique_paths.append(path)

    # 按长度排序
    unique_paths.sort(key=lambda x: len(x), reverse=True)

    # 6. 限制路径数量
    top_paths = unique_paths[:max_branches]

    print(f"Found {len(top_paths)} distinct centerline paths")
    for i, path in enumerate(top_paths):
        print(f"  Path {i + 1}: {len(path)} points")

    # 7. 转换为numpy数组并下采样
    result_paths = []
    for i, path in enumerate(top_paths):
        path_array = np.array(path)
        if skip > 1:
            path_array = path_array[::skip]

        # 保存每条路径
        if save_path:
            base_name = save_path.replace('.npy', '')
            path_save_name = f"{base_name}_path_{i + 1}.npy"
            np.save(path_save_name, path_array)
            print(f"Saved path {i + 1} to {path_save_name}")

        result_paths.append(path_array)

    # 8. 可视化提取的路径
    visualize_extracted_paths(seg, result_paths)

    return result_paths

def visualize_extracted_paths(volume, paths, title="Extracted Centerline Paths"):
    """
    可视化提取的中心线路径

    参数:
    -----------
    volume : numpy.ndarray
        原始体积数据
    paths : List[np.ndarray]
        中心线路径列表
    title : str
        图像标题
    """
    # 创建颜色映射
    colors = ['red', 'green', 'blue', 'orange', 'purple']

    fig = plt.figure(figsize=(15, 5))

    # 创建三个方向的投影
    mip_xy = np.max(volume, axis=2)
    mip_xz = np.max(volume, axis=1)
    mip_yz = np.max(volume, axis=0)

    # 绘制XY平面
    ax1 = fig.add_subplot(131)
    ax1.imshow(mip_xy, cmap='gray', alpha=0.7, origin='lower')
    ax1.set_title('XY Plane (Axial)')
    ax1.set_xlabel('X (voxels)')
    ax1.set_ylabel('Y (voxels)')

    # 绘制XZ平面
    ax2 = fig.add_subplot(132)
    ax2.imshow(mip_xz, cmap='gray', alpha=0.7, origin='lower')
    ax2.set_title('XZ Plane (Coronal)')
    ax2.set_xlabel('X (voxels)')
    ax2.set_ylabel('Z (voxels)')

    # 绘制YZ平面
    ax3 = fig.add_subplot(133)
    ax3.imshow(mip_yz, cmap='gray', alpha=0.7, origin='lower')
    ax3.set_title('YZ Plane (Sagittal)')
    ax3.set_xlabel('Y (voxels)')
    ax3.set_ylabel('Z (voxels)')

    # 在三个平面上绘制所有路径
    for i, path in enumerate(paths):
        color = colors[i % len(colors)]

        if len(path) > 0:
            # XY投影
            ax1.scatter(path[:, 1], path[:, 0], s=5, c=color, alpha=0.7,
                        label=f'Path {i + 1} ({len(path)} pts)')
            ax1.plot(path[:, 1], path[:, 0], color=color, alpha=0.5, linewidth=1)

            # XZ投影
            ax2.scatter(path[:, 1], path[:, 2], s=5, c=color, alpha=0.7)
            ax2.plot(path[:, 1], path[:, 2], color=color, alpha=0.5, linewidth=1)

            # YZ投影
            ax3.scatter(path[:, 0], path[:, 2], s=5, c=color, alpha=0.7)
            ax3.plot(path[:, 0], path[:, 2], color=color, alpha=0.5, linewidth=1)

    ax1.legend(loc='upper right', fontsize=8)
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

    # 3D可视化
    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    # 绘制体积的边界框
    z, y, x = np.where(volume > 0)
    if len(x) > 0:
        ax_3d.scatter(x, y, z, c='gray', alpha=0.1, s=1)

    # 绘制每条路径
    for i, path in enumerate(paths):
        if len(path) > 0:
            color = colors[i % len(colors)]
            # 注意：numpy的坐标顺序是 (z, y, x)，但3D绘图通常用 (x, y, z)
            ax_3d.plot(path[:, 1], path[:, 0], path[:, 2],
                       color=color, linewidth=2, alpha=0.8,
                       label=f'Path {i + 1} ({len(path)} pts)')
            ax_3d.scatter(path[:, 1], path[:, 0], path[:, 2],
                          color=color, s=20, alpha=0.6)

    ax_3d.set_xlabel('X (voxels)')
    ax_3d.set_ylabel('Y (voxels)')
    ax_3d.set_zlabel('Z (voxels)')
    ax_3d.set_title('3D View of Extracted Centerlines')
    ax_3d.legend()
    ax_3d.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def extract_coordinates(seg_path, save_path=None, skip=10):
    """
    向后兼容的单一中心线提取函数
    """
    print("=" * 40)
    print("SEGMENTATION PATH:", seg_path)
    print("SAVE_PATH:", save_path if save_path is not None else "./points.npy")
    print("SKIP FACTOR:", skip)
    print("=" * 40)

    # 1. Load and skeletonize
    seg, _ = mio.load(seg_path)
    seg_bin = (seg > 0)
    skeleton = skeletonize(seg_bin)
    coords = np.array(np.nonzero(skeleton)).T  # (N, 3)

    print("Total skeleton points:", coords.shape[0])

    if coords.shape[0] < 2:
        print("Not enough skeleton points to extract centerline!!")
        print("Extraction failed.")
        return False

    # 使用新的多分支提取函数，但只取第一条路径
    paths = extract_multi_branch_centerlines(seg_path, save_path, skip, max_branches=1)

    if len(paths) > 0:
        ordered = paths[0]
        if save_path:
            np.save(save_path, ordered)
        return ordered
    else:
        return False