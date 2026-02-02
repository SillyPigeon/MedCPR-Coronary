import os
import numpy as np
from cpr import extract_multi_branch_centerlines, curve_planar_reformat, extract_axial_slice
import glob


def process_coronary_data_multi(ct_folder, seg_folder, output_folder,
                                skip=5, max_branches=3, fov_mm=65,
                                rotation_angles=[0, 45, 90, 135]):
    """
    批量处理冠脉CT图像，提取三条中心线，生成多个角度的CPR图像，并提取关键切片

    参数:
    -----------
    ct_folder : str
        CT原图文件夹路径（包含.nii.gz文件）
    seg_folder : str
        冠脉标签数据文件夹路径（包含.nii.gz文件）
    output_folder : str
        输出文件夹路径
    skip : int
        中心线下采样间隔
    max_branches : int
        最大分支数量（设置为3，提取三条中心线）
    fov_mm : int
        CPR视野大小（毫米）
    rotation_angles : list
        旋转角度列表，默认为[0, 45, 90, 135]四个角度
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

    # 确保文件数量匹配
    if len(ct_files) != len(seg_files):
        print("警告: CT文件和分割文件数量不匹配!")
        print("将尝试按文件名排序进行匹配")

    # 按文件名排序以确保匹配
    ct_files.sort()
    seg_files.sort()

    # 处理每个文件对
    for i, (ct_file, seg_file) in enumerate(zip(ct_files, seg_files)):
        print("\n" + "=" * 60)
        print(f"处理文件对 {i + 1}/{len(ct_files)}")
        print(f"CT文件: {os.path.basename(ct_file)}")
        print(f"分割文件: {os.path.basename(seg_file)}")
        print("=" * 60)

        # 提取基础文件名（不含扩展名）
        ct_basename = os.path.basename(ct_file)
        seg_basename = os.path.basename(seg_file)

        # 去除扩展名
        ct_name = ct_basename.replace('.nii.gz', '').replace('.nii', '')
        seg_name = seg_basename.replace('.nii.gz', '').replace('.nii', '')

        # 为当前文件创建临时工作目录
        temp_dir = os.path.join(output_folder, "temp", f"case_{i + 1:03d}_{ct_name}")
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # 步骤1: 提取三条中心线
            print(f"\n1. 提取三条中心线...")
            centerline_save_path = os.path.join(temp_dir, f"{seg_name}_centerline.npy")

            # 提取最多3条中心线
            paths = extract_multi_branch_centerlines(
                seg_path=seg_file,
                save_path=centerline_save_path,
                skip=skip,
                max_branches=max_branches  # 设置为3，提取三条中心线
            )

            if not paths:
                print(f"警告: {seg_name} 中心线提取失败，跳过此文件")
                continue

            actual_branches = len(paths)
            print(f"成功提取 {actual_branches} 条中心线路径")

            # 处理每条中心线路径
            for path_idx in range(actual_branches):
                print(f"\n--- 处理中心线路径 {path_idx + 1}/{actual_branches} ---")

                # 检查中心线点文件是否存在
                points_path = os.path.join(temp_dir, f"{seg_name}_centerline_path_{path_idx + 1}.npy")

                if not os.path.exists(points_path):
                    print(f"警告: 中心线点文件 {points_path} 不存在，跳过此路径")
                    continue

                # 步骤2: 为当前中心线生成四个角度的CPR图像
                print(f"2. 为中心线路径 {path_idx + 1} 生成四个角度的CPR图像...")

                # 为当前路径创建子目录
                path_temp_dir = os.path.join(temp_dir, f"path_{path_idx + 1}")
                os.makedirs(path_temp_dir, exist_ok=True)

                cpr_save_path = os.path.join(path_temp_dir, f"{ct_name}_cpr.nii.gz")

                # 生成四个角度的CPR图像
                num_angles = curve_planar_reformat(
                    image_path=ct_file,
                    points_path=points_path,
                    save_path=cpr_save_path,
                    fov_mm=fov_mm,
                    rotation_angle=rotation_angles  # 四个角度
                )

                print(f"为中心线路径 {path_idx + 1} 成功生成 {num_angles} 个角度的CPR图像")

                # 步骤3: 为每个角度的CPR图像提取关键切片
                print(f"3. 为中心线路径 {path_idx + 1} 提取关键切片...")

                # 收集生成的CPR文件
                cpr_files = []
                if num_angles == 1:
                    # 如果只有一个角度，直接使用保存的文件
                    if os.path.exists(cpr_save_path):
                        cpr_files.append(cpr_save_path)
                else:
                    # 如果有多个角度，收集所有角度的文件
                    for angle in rotation_angles:
                        cpr_file = cpr_save_path.replace('.nii.gz', f'_angle{angle}.nii.gz')
                        if os.path.exists(cpr_file):
                            cpr_files.append(cpr_file)

                print(f"找到 {len(cpr_files)} 个CPR文件用于提取切片")

                # 为每个CPR文件提取关键切片
                for cpr_file in cpr_files:
                    # 提取角度信息
                    cpr_filename = os.path.basename(cpr_file)

                    # 解析角度值
                    if "_angle" in cpr_filename:
                        # 提取角度值
                        angle_part = cpr_filename.split("_angle")[1]
                        angle_value = angle_part.split(".")[0]
                    else:
                        angle_value = "0"

                    # 构建输出文件名
                    output_name = f"{ct_name}_path{path_idx + 1}_angle{angle_value}"
                    slice_output_path = os.path.join(output_folder, output_name)

                    print(f"  提取角度 {angle_value}° 的关键切片...")

                    # 提取轴向切片
                    try:
                        saved_files = extract_axial_slice(
                            cpr_nifti_path=cpr_file,
                            save_path=slice_output_path
                        )

                        if saved_files:
                            print(f"  成功保存关键切片: {output_name}_depth*.png")
                        else:
                            print(f"  警告: 角度 {angle_value}° 的关键切片提取失败")

                    except Exception as e:
                        print(f"  提取角度 {angle_value}° 的关键切片时出错: {str(e)}")
                        continue

            print(f"\n✓ 文件 {ct_name} 处理完成")

        except Exception as e:
            print(f"处理 {ct_name} 时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 60)
    print("批量处理完成!")
    print(f"所有输出文件保存在: {output_folder}")

    # 显示输出文件统计
    print("\n输出文件统计:")
    png_files = glob.glob(os.path.join(output_folder, "*.png"))
    if png_files:
        print(f"共生成 {len(png_files)} 个PNG关键切片文件")

        # 按文件前缀分组统计
        file_groups = {}
        for png_file in png_files:
            filename = os.path.basename(png_file)
            # 提取文件前缀（去掉_depthXXX.png部分）
            if "_depth" in filename:
                prefix = filename.split("_depth")[0]
                if prefix not in file_groups:
                    file_groups[prefix] = 0
                file_groups[prefix] += 1

        print("\n文件分组统计:")
        for prefix, count in file_groups.items():
            print(f"  {prefix}: {count} 个切片")
    else:
        print("未找到PNG输出文件")

    print("=" * 60)

    # 清理临时文件（可选，注释掉以保留中间文件用于调试）
    # temp_dir = os.path.join(output_folder, "temp")
    # if os.path.exists(temp_dir):
    #     import shutil
    #     shutil.rmtree(temp_dir)
    #     print(f"已清理临时目录: {temp_dir}")


def main():
    """主函数：配置参数并运行批量处理"""

    # 配置参数
    CT_FOLDER = "./ct_data"  # CT原图文件夹路径
    SEG_FOLDER = "./ct_label"  # 冠脉标签数据文件夹路径
    OUTPUT_FOLDER = "./output"  # 输出文件夹路径

    # 处理参数
    SKIP = 5  # 中心线下采样间隔
    MAX_BRANCHES = 3  # 提取三条中心线
    FOV_MM = 90  # CPR视野大小
    ROTATION_ANGLES = [0, 45, 90, 135]  # 四个旋转角度

    print("冠脉CT图像批量处理脚本")
    print(f"CT文件夹: {CT_FOLDER}")
    print(f"分割文件夹: {SEG_FOLDER}")
    print(f"输出文件夹: {OUTPUT_FOLDER}")
    print(f"提取中心线数量: {MAX_BRANCHES}")
    print(f"旋转角度: {ROTATION_ANGLES}")
    print("=" * 60)

    # 检查输入文件夹是否存在
    if not os.path.exists(CT_FOLDER):
        print(f"错误: CT文件夹不存在: {CT_FOLDER}")
        return

    if not os.path.exists(SEG_FOLDER):
        print(f"错误: 分割文件夹不存在: {SEG_FOLDER}")
        return

    # 运行批量处理
    process_coronary_data_multi(
        ct_folder=CT_FOLDER,
        seg_folder=SEG_FOLDER,
        output_folder=OUTPUT_FOLDER,
        skip=SKIP,
        max_branches=MAX_BRANCHES,
        fov_mm=FOV_MM,
        rotation_angles=ROTATION_ANGLES
    )


if __name__ == "__main__":
    main()