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


def curve_planar_reformat(image_path, points_path, save_path, fov_mm=50, rotation_angle=0):
    """
    曲线平面重建，支持旋转角度

    参数:
    -----------
    image_path : str
        原始图像路径
    points_path : str
        中心线点路径
    save_path : str
        保存路径
    fov_mm : int
        视野大小（毫米）
    rotation_angle : float or str or list
        旋转角度，支持：
        - float: 固定角度（度）
        - "multi": 生成多角度（0°, 45°, 90°, 135°）
        - list: 自定义角度列表 [angle1, angle2, ...]
    """
    print("=" * 40)
    print("IMAGE PATH:", image_path)
    print("POINTS PATH:", points_path)
    print("SAVE PATH:", save_path if save_path is not None else "reformatted_image.nii.gz")
    print("FIELD OF VIEW (mm):", fov_mm)
    print("ROTATION ANGLE:", rotation_angle)
    print("=" * 40)

    # Step 1: 加载图像和点
    image, h0 = mio.load(image_path)
    points = np.load(points_path)  # 形状 (N, 3)

    pixel_spacing = h0.get_voxel_spacing()

    # 计算长度
    diff_points = points[1:] - points[:-1]
    actual_distances = np.sum(np.linalg.norm((diff_points) * np.array(pixel_spacing), axis=1))
    pixel_distances = np.insert(np.cumsum(np.linalg.norm(diff_points, axis=1)), 0, 0)
    total_pixel_length = pixel_distances[-1]

    print(f"Actual Length of the vessel: {actual_distances} mm")
    print(f"Pixel Length of the vessel: {total_pixel_length} pixels")

    # 样条插值
    spline = CubicSpline(pixel_distances, points, bc_type='natural')
    num_steps = int(total_pixel_length)
    new_dists = np.linspace(0, total_pixel_length, num_steps)

    resampled_points = spline(new_dists)  # 形状 (num_steps, 3)
    resampled_tangents = spline(new_dists, nu=1)

    # 处理旋转角度参数
    if rotation_angle == "multi":
        rotation_angles = [0, 45, 90, 135]
    elif isinstance(rotation_angle, list):
        rotation_angles = rotation_angle
    else:
        rotation_angles = [rotation_angle]

    # 为每个角度生成CPR图像
    for angle_idx, base_angle in enumerate(rotation_angles):
        print(f"\nProcessing rotation angle: {base_angle}°")

        # 转换角度为弧度
        angle_rad = np.deg2rad(base_angle)

        # 创建切片网格
        pixels_per_side = int(fov_mm)
        half_size = pixels_per_side // 2
        grid_range = np.arange(-half_size, half_size + 1)
        u_grid, v_grid = np.meshgrid(grid_range, grid_range)
        u_flat = u_grid.flatten()
        v_flat = v_grid.flatten()

        output_slices = []
        prev_U = None

        # 对于旋转角度重建，需要保持参考向量的一致性
        # 使用全局参考向量，但应用旋转
        global_ref_vec = np.array([0, 1, 0])  # 原始参考向量

        for i in range(len(resampled_points)):
            # 当前点
            p_curr = resampled_points[i]

            # 切线方向
            tangent = resampled_tangents[i]
            t_norm = tangent / np.linalg.norm(tangent)

            # 方法1：基于旋转角度计算U向量
            # 先计算基础U向量
            base_u_vec = np.cross(t_norm, global_ref_vec)

            # 处理特殊情况（平行向量）
            if np.linalg.norm(base_u_vec) < 1e-6:
                base_u_vec = np.cross(t_norm, np.array([1, 0, 0]))

            # 归一化
            base_u_norm = base_u_vec / np.linalg.norm(base_u_vec)

            # 计算V向量（完成正交基）
            base_v_vec = np.cross(t_norm, base_u_norm)
            base_v_norm = base_v_vec / np.linalg.norm(base_v_vec)

            # 方法2：使用旋转矩阵绕切线方向旋转
            # 绕切线旋转角度
            if np.abs(base_angle) > 1e-6:
                # 创建旋转矩阵（绕切线轴）
                cos_a = np.cos(angle_rad)
                sin_a = np.sin(angle_rad)

                # 罗德里格斯旋转公式
                # R = I + sinθ * K + (1-cosθ) * K²
                # 其中K是切线向量的叉乘矩阵

                # 更简单的方法：直接旋转U和V向量
                u_norm = cos_a * base_u_norm + sin_a * base_v_norm
                v_norm = -sin_a * base_u_norm + cos_a * base_v_norm
            else:
                u_norm = base_u_norm
                v_norm = base_v_norm

            # 确保方向连续性（与前一片对比）
            if prev_U is not None:
                # 检查并防止方向翻转
                if np.dot(u_norm, prev_U) < 0:
                    u_norm = -u_norm
                    v_norm = -v_norm

                # 平滑过渡
                u_norm = (0.9 * prev_U) + (0.1 * u_norm)
                u_norm = u_norm / np.linalg.norm(u_norm)

                # 重新计算V向量确保正交
                v_norm = np.cross(t_norm, u_norm)
                v_norm = v_norm / np.linalg.norm(v_norm)

            prev_U = u_norm

            # 坐标转换
            center_mm = p_curr * np.array(pixel_spacing)
            slice_coords_mm = (
                    center_mm +
                    np.outer(u_flat, u_norm) +
                    np.outer(v_flat, v_norm)
            )

            slice_coords_pix = slice_coords_mm / np.array(pixel_spacing)
            coords_for_scipy = slice_coords_pix.T

            # 插值
            slice_pixels = map_coordinates(
                image,
                coords_for_scipy,
                order=1,
            )

            slice_2d = slice_pixels.reshape(len(grid_range), len(grid_range))
            output_slices.append(slice_2d)

        # 保存结果
        if len(rotation_angles) == 1:
            save_filename = save_path
        else:
            # 为多角度生成不同文件名
            base_name = save_path.replace('.nii.gz', '').replace('.nii', '')
            save_filename = f"{base_name}_angle{base_angle}.nii.gz"

        # 保存图像
        output_array = np.array(output_slices)
        mio.save(output_array, save_filename, h0)

        print(f"Saved CPR image for angle {base_angle}° at {save_filename}")

        # 可视化角度的结果
        # if angle_idx in [0, 1, 2, 3]:
        #     visualize_cpr_result(output_array, base_angle)

    return len(rotation_angles)


def visualize_cpr_result(cpr_volume, angle):
    """
    可视化CPR结果

    参数:
    -----------
    cpr_volume : numpy.ndarray
        CPR体积数据，形状 (depth, height, width)
    angle : float
        旋转角度
    """
    # 创建三个方向的视图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 纵向切面（沿着血管长度）
    ax1 = axes[0, 0]
    longitudinal_view = np.max(cpr_volume, axis=2)  # 沿宽度方向投影
    ax1.imshow(longitudinal_view, cmap='gray', aspect='auto', origin='lower')
    ax1.set_title(f'Longitudinal View (Angle={angle}°)')
    ax1.set_xlabel('Vessel Length (slices)')
    ax1.set_ylabel('Cross-section Height')
    ax1.grid(True, alpha=0.3)

    # 2. 横截面（选几个位置）
    ax2 = axes[0, 1]
    num_slices = cpr_volume.shape[0]
    sample_indices = [
        0,
        num_slices // 4,
        num_slices // 2,
        3 * num_slices // 4,
        num_slices - 1
    ]

    for i, idx in enumerate(sample_indices):
        if idx < num_slices:
            cross_section = cpr_volume[idx, :, :]
            # 稍微偏移每个横截面以便区分
            offset = i * 5
            ax2.imshow(cross_section, cmap='gray',
                       extent=[offset, offset + cpr_volume.shape[2],
                               0, cpr_volume.shape[1]],
                       alpha=0.7, origin='lower')

    ax2.set_title(f'Cross-sections at Different Positions (Angle={angle}°)')
    ax2.set_xlabel('Cross-section Width')
    ax2.set_ylabel('Cross-section Height')
    ax2.set_xlim([0, 5 * cpr_volume.shape[2]])
    ax2.grid(True, alpha=0.3)

    # 3. 3D CPR体积视图
    ax3 = axes[1, 0]
    volume_mip = np.max(cpr_volume, axis=0)
    ax3.imshow(volume_mip, cmap='gray', origin='lower')
    ax3.set_title(f'CPR Volume MIP (Angle={angle}°)')
    ax3.set_xlabel('Cross-section Width')
    ax3.set_ylabel('Cross-section Height')
    ax3.grid(True, alpha=0.3)

    # 4. 角度说明
    ax4 = axes[1, 1]
    ax4.axis('off')

    # 绘制角度示意图
    circle = plt.Circle((0.5, 0.5), 0.3, fill=False, linewidth=2)
    ax4.add_patch(circle)

    # 绘制参考线
    ax4.plot([0.5, 0.8], [0.5, 0.5], 'k--', alpha=0.5, label='0° reference')

    # 绘制角度线
    angle_rad = np.deg2rad(angle)
    end_x = 0.5 + 0.3 * np.cos(angle_rad)
    end_y = 0.5 + 0.3 * np.sin(angle_rad)
    ax4.plot([0.5, end_x], [0.5, end_y], 'r-', linewidth=3,
             label=f'Angle = {angle}°')

    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1])
    ax4.set_aspect('equal')
    ax4.set_title('Rotation Angle Visualization')
    ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))

    plt.suptitle(f'Curved Planar Reformation Results (Rotation Angle: {angle}°)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

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
    提取多分支中心线，根据冠状动脉的解剖特点（两条主分支）选择路径，
    优先选择最长且互相不重复的路径

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

    # 3. 找到所有端点（度数为1的节点，或者度数大于3的节点）
    degree_dict = dict(G.degree())

    # 修改点1：degree == 1 或 degree > 3 都视为端点
    endpoints = [node for node, degree in degree_dict.items() if degree == 1 or degree > 3]
    print(f"Found {len(endpoints)} endpoints (degree=1 or degree>3)")

    if len(endpoints) < 2:
        print("Not enough endpoints to form a path!")
        return []

    # 4. 按Z坐标值（第3个元素，索引2）对端点排序（降序）
    endpoints.sort(key=lambda x: x[2], reverse=True)

    # 5. 选取主分支1的起点（Z坐标最大的端点）
    start1 = endpoints[0]
    print(f"Main branch 1 start (highest Z): {start1}")

    # 6. 为主分支1生成所有可能路径（从start1到其他所有端点）
    branch1_paths = []
    for end in endpoints[1:]:  # 排除起点自身
        try:
            path = nx.shortest_path(G, start1, end)
            if len(path) > 2:  # 排除非常短的路径
                # 计算路径的"独特性"指标：端点距离 + 路径长度
                # 更高的Z坐标差可能意味着更大的分支角度
                z_diff = abs(start1[2] - end[2])
                path_info = {
                    'path': path,
                    'length': len(path),
                    'z_diff': z_diff,
                    'endpoint': end,
                    'start': start1
                }
                branch1_paths.append(path_info)
        except nx.NetworkXNoPath:
            continue

    print(f"Found {len(branch1_paths)} possible paths for branch 1")

    # 7. 选取主分支2的起点（剩余端点中Z坐标最大的）
    # 首先找出branch1_paths中使用的端点
    branch1_endpoints = set([start1])
    for path_info in branch1_paths:
        branch1_endpoints.add(path_info['endpoint'])

    remaining_endpoints = [ep for ep in endpoints if ep not in branch1_endpoints]

    if not remaining_endpoints:
        print("No remaining endpoints for branch 2 after branch 1 endpoints selection")

        # 修改点2：修改这里的选择逻辑
        # 从所有端点中选取距离start1大于10且不在branch1_paths中的点
        valid_candidates = []
        for ep in endpoints[1:]:  # 排除start1
            # 计算欧氏距离
            distance = np.linalg.norm(np.array(start1) - np.array(ep))

            # 检查是否在branch1_paths中
            in_branch1 = any(ep == path_info['endpoint'] for path_info in branch1_paths)

            if distance > 10 and not in_branch1:
                valid_candidates.append((ep, distance))

        if valid_candidates:
            # 按Z坐标降序排序
            valid_candidates.sort(key=lambda x: x[0][2], reverse=True)
            start2 = valid_candidates[0][0]
            print(
                f"Using valid candidate with Z={start2[2]} as branch 2 start (distance to start1: {valid_candidates[0][1]:.2f})")
        else:
            print("Cannot find suitable branch 2 start (no endpoints with distance > 10 and not in branch1)")
            branch2_paths = []
    else:
        # 修改点2：修改这里的选择逻辑
        # 从剩余端点中选取距离start1大于10且不在branch1_paths中的点
        valid_candidates = []
        for ep in remaining_endpoints:
            # 计算欧氏距离
            distance = np.linalg.norm(np.array(start1) - np.array(ep))

            if distance > 10:
                valid_candidates.append((ep, distance))

        if valid_candidates:
            # 按Z坐标降序排序
            valid_candidates.sort(key=lambda x: x[0][2], reverse=True)
            start2 = valid_candidates[0][0]
            print(
                f"Main branch 2 start (highest Z in valid candidates): {start2}, distance to start1: {valid_candidates[0][1]:.2f}")
        else:
            # 如果没有满足距离条件的端点，使用原逻辑
            start2 = remaining_endpoints[0]
            print(f"Warning: No endpoints with distance > 10. Using highest Z in remaining: {start2}")

    # 8. 为主分支2生成所有可能路径（从start2到其他所有端点）
    # 只在这里添加branch2_paths的初始化，避免undefined错误
    branch2_paths = []

    # 如果找到了有效的start2
    if 'start2' in locals() and start2:
        # 排除start2自身和start1（如果start2 == start1）
        available_ends = [ep for ep in endpoints if ep != start2]

        for end in available_ends:
            try:
                path = nx.shortest_path(G, start2, end)
                if len(path) > 2:  # 排除非常短的路径
                    z_diff = abs(start2[2] - end[2])
                    path_info = {
                        'path': path,
                        'length': len(path),
                        'z_diff': z_diff,
                        'endpoint': end,
                        'start': start2
                    }
                    branch2_paths.append(path_info)
            except nx.NetworkXNoPath:
                continue

        print(f"Found {len(branch2_paths)} possible paths for branch 2")
    else:
        print("No valid start2 found, skipping branch 2 path generation")

    # 9. 合并所有路径信息
    all_paths_info = branch1_paths + branch2_paths

    if not all_paths_info:
        print("No valid paths found!")
        return []

    # 10. 路径选择和排序策略：优先选择长且不重复的路径
    selected_paths = []
    all_paths_info.sort(key=lambda x: x['length'], reverse=True)  # 按长度降序排序

    # 计算路径之间的重叠度
    def calculate_overlap(path1_set, path2_set):
        intersection = path1_set.intersection(path2_set)
        if len(path1_set) == 0 or len(path2_set) == 0:
            return 0
        return len(intersection) / min(len(path1_set), len(path2_set))

    # 优先选择主分支1中最长的路径
    if branch1_paths:
        # 先为主分支1选择路径
        branch1_paths.sort(key=lambda x: x['length'], reverse=True)
        best_branch1 = branch1_paths[0]
        selected_paths.append(best_branch1)
        print(f"Selected branch 1 path: length={best_branch1['length']}, Z difference={best_branch1['z_diff']}")

    # 从所有路径中选择与已选路径重叠度最低的路径
    selected_sets = []
    if selected_paths:
        selected_sets.append(set(selected_paths[0]['path']))

    # 选择其他路径（考虑长度和重叠度）
    remaining_paths = [p for p in all_paths_info if p not in selected_paths]

    while len(selected_paths) < max_branches and remaining_paths:
        # 为每个剩余路径计算"得分"
        path_scores = []
        for path_info in remaining_paths:
            path_set = set(path_info['path'])

            # 计算与所有已选路径的最大重叠度
            max_overlap = 0
            if selected_sets:
                for selected_set in selected_sets:
                    overlap = calculate_overlap(path_set, selected_set)
                    max_overlap = max(max_overlap, overlap)

            # 得分公式：长度权重 * (1 - 重叠度)
            # 给予长度更高的权重，但惩罚重叠
            length_weight = path_info['length'] / 100.0  # 归一化
            score = length_weight * (1.0 - max_overlap)

            # 额外奖励Z坐标差异大的路径（可能代表不同分支）
            z_bonus = path_info['z_diff'] / 100.0

            path_scores.append({
                'path_info': path_info,
                'score': score + z_bonus,
                'overlap': max_overlap,
                'length': path_info['length']
            })

        # 按得分排序
        path_scores.sort(key=lambda x: x['score'], reverse=True)

        if path_scores:
            best_candidate = path_scores[0]
            selected_paths.append(best_candidate['path_info'])
            selected_sets.append(set(best_candidate['path_info']['path']))

            # 从剩余路径中移除
            remaining_paths = [p for p in remaining_paths if p != best_candidate['path_info']]

            print(f"Selected additional path: length={best_candidate['length']}, "
                  f"overlap={best_candidate['overlap']:.2f}, score={best_candidate['score']:.3f}")
        else:
            break

    # 11. 转换为路径列表并确保唯一性
    final_paths = []
    seen_endpoint_pairs = set()

    for path_info in selected_paths:
        path = path_info['path']
        # 确保路径方向唯一性
        endpoint_pair = tuple(sorted([path[0], path[-1]]))
        if endpoint_pair not in seen_endpoint_pairs:
            seen_endpoint_pairs.add(endpoint_pair)
            final_paths.append(path)
        else:
            print(f"Skipping duplicate path with endpoints {endpoint_pair}")

    print(f"\nFinal selection: {len(final_paths)} paths")
    for i, path in enumerate(final_paths):
        print(f"  Path {i + 1}: {len(path)} points, "
              f"start={path[0]}, end={path[-1]}")

    # 计算路径之间的重叠统计
    if len(final_paths) > 1:
        print("\nPath overlap statistics:")
        for i in range(len(final_paths)):
            for j in range(i + 1, len(final_paths)):
                set_i = set(map(tuple, final_paths[i]))
                set_j = set(map(tuple, final_paths[j]))
                overlap = set_i.intersection(set_j)
                overlap_ratio = len(overlap) / min(len(set_i), len(set_j))
                print(f"  Path {i + 1} vs Path {j + 1}: {len(overlap)} overlapping points "
                      f"(ratio: {overlap_ratio:.2f})")

    # 12. 转换为numpy数组并下采样
    result_paths = []
    for i, path in enumerate(final_paths):
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

    # 13. 可视化提取的路径
    visualize_extracted_paths(seg, result_paths)

    return result_paths

def visualize_extracted_paths(volume, paths, title="Extracted Centerline Paths"):
    """
    可视化提取的中心线路径

    参数:
    -----------
    volume : numpy.ndarray
        原始体积数据，形状顺序为 (x, y, z)
    paths : List[np.ndarray]
        中心线路径列表，每个路径形状为 (n_points, 3)，顺序为 [x, y, z]
    title : str
        图像标题
    """
    # 创建颜色映射
    colors = ['red', 'green', 'blue', 'orange', 'purple']

    fig = plt.figure(figsize=(15, 5))

    # 获取体积数据的形状（注意：现在volume是(x, y, z)顺序）
    vol_shape = volume.shape  # 顺序为 (x, y, z)

    # 修正：创建三个方向的投影（根据新的坐标顺序）
    # 原来的逻辑需要调整：
    # XY平面（沿着Z轴投影）：np.max(volume, axis=2)
    # XZ平面（沿着Y轴投影）：np.max(volume, axis=1)
    # YZ平面（沿着X轴投影）：np.max(volume, axis=0)

    # 根据你的要求交换XY和YZ平面：
    mip_xy = np.max(volume, axis=2)  # XY平面，沿着Z轴投影（X-Y平面，显示Z方向投影）
    mip_xz = np.max(volume, axis=1)  # XZ平面，沿着Y轴投影（X-Z平面，显示Y方向投影）
    mip_yz = np.max(volume, axis=0)  # YZ平面，沿着X轴投影（Y-Z平面，显示X方向投影）

    # 绘制XY平面 (Z轴投影) - 现在这是第一个子图
    ax1 = fig.add_subplot(131)
    ax1.imshow(mip_xy, cmap='gray', alpha=0.7, origin='lower',
               extent=[0, vol_shape[1], 0, vol_shape[0]])  # 注意：X对应宽度，Y对应高度
    ax1.set_title('XY Plane (Axial View)')
    ax1.set_xlabel('Y (voxels)')
    ax1.set_ylabel('X (voxels)')  # 注意：这里交换了标签
    ax1.grid(True, alpha=0.3, linestyle='--')

    # 绘制XZ平面 (Y轴投影) - 保持中间位置
    ax2 = fig.add_subplot(132)
    ax2.imshow(mip_xz, cmap='gray', alpha=0.7, origin='lower',
               extent=[0, vol_shape[2], 0, vol_shape[0]])  # X对应宽度，Z对应高度
    ax2.set_title('XZ Plane (Coronal View)')
    ax2.set_xlabel('Z (voxels)')
    ax2.set_ylabel('X (voxels)')  # 注意：这里调整了标签
    ax2.grid(True, alpha=0.3, linestyle='--')

    # 绘制YZ平面 (X轴投影) - 现在这是第三个子图，与原来的XY平面交换
    ax3 = fig.add_subplot(133)
    ax3.imshow(mip_yz, cmap='gray', alpha=0.7, origin='lower',
               extent=[0, vol_shape[2], 0, vol_shape[1]])  # Y对应宽度，Z对应高度
    ax3.set_title('YZ Plane (Sagittal View)')
    ax3.set_xlabel('Z (voxels)')
    ax3.set_ylabel('Y (voxels)')  # 注意：这里调整了标签
    ax3.grid(True, alpha=0.3, linestyle='--')

    # 在三个平面上绘制所有路径
    for i, path in enumerate(paths):
        if len(path) == 0:
            continue

        color = colors[i % len(colors)]

        # 路径坐标：path 的形状为 (n_points, 3)，顺序为 [x, y, z]
        x_coords = path[:, 0]
        y_coords = path[:, 1]
        z_coords = path[:, 2]

        # XY平面 (Axial): 显示X和Y坐标
        # 注意：imshow显示时，第一个维度是行（Y），第二个维度是列（X）
        # 但在scatter/plot中，x参数对应列（imshow的第二个维度），y参数对应行（imshow的第一个维度）
        ax1.scatter(y_coords, x_coords, s=15, c=color, alpha=0.8,
                    label=f'Path {i + 1} ({len(path)} pts)')
        ax1.plot(y_coords, x_coords, color=color, alpha=0.6, linewidth=1.5)

        # XZ平面 (Coronal): 显示X和Z坐标
        ax2.scatter(z_coords, x_coords, s=15, c=color, alpha=0.8)
        ax2.plot(z_coords, x_coords, color=color, alpha=0.6, linewidth=1.5)

        # YZ平面 (Sagittal): 显示Y和Z坐标
        ax3.scatter(z_coords, y_coords, s=15, c=color, alpha=0.8)
        ax3.plot(z_coords, y_coords, color=color, alpha=0.6, linewidth=1.5)

    # 添加图例
    ax1.legend(loc='best', fontsize=9)

    # 设置统一的坐标范围
    ax1.set_xlim([0, vol_shape[1]])  # Y轴范围
    ax1.set_ylim([0, vol_shape[0]])  # X轴范围
    ax2.set_xlim([0, vol_shape[2]])  # Z轴范围
    ax2.set_ylim([0, vol_shape[0]])  # X轴范围
    ax3.set_xlim([0, vol_shape[2]])  # Z轴范围
    ax3.set_ylim([0, vol_shape[1]])  # Y轴范围

    # 调整布局
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # 3D可视化（可选） - 也需要相应调整
    plot_3d_view(volume, paths)


def plot_3d_view(volume, paths):
    """
    3D可视化函数 - 需要相应调整
    """
    fig_3d = plt.figure(figsize=(12, 10))
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    # 获取体积边界（注意：volume现在是(x, y, z)顺序）
    vol_shape = volume.shape

    # 创建颜色映射
    colors = ['red', 'green', 'blue', 'orange', 'purple']

    # 绘制体积的边界框（可选）
    if np.sum(volume) < 1000000:  # 如果体积不是太大
        # 注意：现在volume是(x, y, z)顺序，np.where返回的也是这个顺序
        x_idx, y_idx, z_idx = np.where(volume > 0)
        if len(x_idx) > 0:
            sample_size = min(10000, len(x_idx))
            indices = np.random.choice(len(x_idx), sample_size, replace=False)
            ax_3d.scatter(x_idx[indices], y_idx[indices], z_idx[indices],
                          c='lightgray', alpha=0.05, s=1, marker='.')

    # 绘制每条路径
    for i, path in enumerate(paths):
        if len(path) == 0:
            continue

        color = colors[i % len(colors)]
        x_coords = path[:, 0]  # X坐标
        y_coords = path[:, 1]  # Y坐标
        z_coords = path[:, 2]  # Z坐标

        # 绘制3D路径
        ax_3d.plot(x_coords, y_coords, z_coords,
                   color=color, linewidth=3, alpha=0.8,
                   label=f'Path {i + 1} ({len(path)} pts)')

        # 绘制路径点
        ax_3d.scatter(x_coords, y_coords, z_coords,
                      color=color, s=30, alpha=0.8, depthshade=True)

        # 标记起点和终点
        if len(path) > 0:
            # 起点
            ax_3d.scatter(x_coords[0], y_coords[0], z_coords[0],
                          color='yellow', s=100, marker='o', edgecolors='black',
                          label=f'Start {i + 1}' if i == 0 else "")

            # 终点
            ax_3d.scatter(x_coords[-1], y_coords[-1], z_coords[-1],
                          color='cyan', s=100, marker='s', edgecolors='black',
                          label=f'End {i + 1}' if i == 0 else "")

    # 设置坐标轴标签（保持(x, y, z)顺序）
    ax_3d.set_xlabel('X (voxels)', fontsize=12)
    ax_3d.set_ylabel('Y (voxels)', fontsize=12)
    ax_3d.set_zlabel('Z (voxels)', fontsize=12)

    # 设置坐标范围
    ax_3d.set_xlim([0, vol_shape[0]])  # X轴范围
    ax_3d.set_ylim([0, vol_shape[1]])  # Y轴范围
    ax_3d.set_zlim([0, vol_shape[2]])  # Z轴范围

    # 设置视角
    ax_3d.view_init(elev=20, azim=45)

    ax_3d.set_title('3D View of Extracted Centerlines', fontsize=14, fontweight='bold')
    ax_3d.legend(loc='upper right', fontsize=9)
    ax_3d.grid(True, alpha=0.3)

    # 添加坐标轴刻度
    ax_3d.set_xticks(np.linspace(0, vol_shape[0], 5))
    ax_3d.set_yticks(np.linspace(0, vol_shape[1], 5))
    ax_3d.set_zticks(np.linspace(0, vol_shape[2], 5))

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