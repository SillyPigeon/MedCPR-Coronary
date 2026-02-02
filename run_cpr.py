import os
import numpy as np
from cpr import extract_multi_branch_centerlines, curve_planar_reformat, extract_axial_slice
import glob


def process_coronary_data_simple(ct_folder, seg_folder, output_folder, skip=5, fov_mm=65):
    """
    简化版批量处理，只处理第一个中心线路径和0°角度

    参数:
    -----------
    ct_folder : str
        CT原图文件夹路径
    seg_folder : str
        冠脉标签数据文件夹路径
    output_folder : str
        输出文件夹路径
    skip : int
        中心线下采样间隔
    fov_mm : int
        CPR视野大小（毫米）
    """

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取CT文件列表
    ct_files = sorted(glob.glob(os.path.join(ct_folder, "*.nii.gz")) +
                      glob.glob(os.path.join(ct_folder, "*.nii")))

    # 获取分割文件列表
    seg_files = sorted(glob.glob(os.path.join(seg_folder, "*.nii.gz")) +
                       glob.glob(os.path.join(seg_folder, "*.nii")))

    print(f"找到 {len(ct_files)} 个CT文件")
    print(f"找到 {len(seg_files)} 个分割文件")

    # 处理每个文件对
    for i, (ct_file, seg_file) in enumerate(zip(ct_files, seg_files)):
        print("\n" + "=" * 60)
        print(f"处理文件对 {i + 1}/{len(ct_files)}")

        # 提取基础文件名
        ct_basename = os.path.basename(ct_file)
        if ct_basename.endswith('.nii.gz'):
            ct_name = ct_basename[:-7]
        elif ct_basename.endswith('.nii'):
            ct_name = ct_basename[:-4]
        else:
            ct_name = ct_basename

        try:
            # 步骤1: 提取中心线（只取第一条路径）
            print(f"1. 提取中心线...")
            temp_dir = os.path.join(output_folder, "temp")
            os.makedirs(temp_dir, exist_ok=True)

            centerline_save_path = os.path.join(temp_dir, "temp_centerline.npy")

            paths = extract_multi_branch_centerlines(
                seg_path=seg_file,
                save_path=centerline_save_path,
                skip=skip,
                max_branches=1  # 只取第一个分支
            )

            if not paths:
                print(f"警告: 中心线提取失败，跳过此文件")
                continue

            # 步骤2: 生成CPR图像（只生成0°角度）
            print(f"2. 生成CPR图像...")
            cpr_save_path = os.path.join(temp_dir, "temp_cpr.nii.gz")
            points_path = os.path.join(temp_dir, "temp_centerline_path_1.npy")

            if not os.path.exists(points_path):
                print(f"警告: 中心线点文件不存在，跳过此文件")
                continue

            curve_planar_reformat(
                image_path=ct_file,
                points_path=points_path,
                save_path=cpr_save_path,
                fov_mm=fov_mm,
                rotation_angle=0  # 只生成0°角度
            )

            # 步骤3: 提取关键切片
            print(f"3. 提取关键切片...")

            # 构建输出路径
            output_path = os.path.join(output_folder, ct_name)

            saved_files = extract_axial_slice(
                cpr_nifti_path=cpr_save_path,
                save_path=output_path
            )

            if saved_files:
                print(f"成功保存关键切片: {output_path}_depth*.png")

        except Exception as e:
            print(f"处理时发生错误: {str(e)}")
            continue

    # 清理临时文件
    temp_dir = os.path.join(output_folder, "temp")
    if os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir)

    print("\n" + "=" * 60)
    print("批量处理完成!")
    print(f"所有关键切片保存在: {output_folder}")
    print("=" * 60)


if __name__ == "__main__":
    # 使用示例
    process_coronary_data_simple(
        ct_folder="./ct_data",
        seg_folder="./ct_label",
        output_folder="./output",
        skip=5,
        fov_mm=65
    )