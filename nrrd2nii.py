import os
import numpy as np
import nibabel as nib
import medpy.io as mio


def batch_nrrd_to_nii(input_dir, output_dir=None):
    """
    批量将NRRD文件转换为NIfTI格式

    参数:
    -----------
    input_dir : str
        包含NRRD文件的输入目录
    output_dir : str or None
        输出NIfTI文件的目录，默认为None（在输入目录下创建nii_output子目录）
    """
    import os

    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(input_dir, "nii_output")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有nrrd文件
    nrrd_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith('.nrrd'):
            nrrd_files.append(file)
        elif file.lower().endswith('.nhdr'):  # 也支持NHDR文件
            nrrd_files.append(file)

    if not nrrd_files:
        print(f"在目录 {input_dir} 中未找到NRRD文件")
        return []

    print(f"找到 {len(nrrd_files)} 个NRRD文件:")
    for file in nrrd_files:
        print(f"  - {file}")

    # 批量转换
    converted_files = []
    for nrrd_file in nrrd_files:
        try:
            # 构建完整路径
            input_path = os.path.join(input_dir, nrrd_file)

            # 生成输出文件名（将.nrrd/.nhdr替换为.nii.gz）
            base_name = os.path.splitext(nrrd_file)[0]
            if base_name.endswith('.nrrd'):
                base_name = base_name[:-5]
            output_file = base_name + ".nii.gz"
            output_path = os.path.join(output_dir, output_file)

            print(f"\n处理: {nrrd_file}")
            print(f"  输入: {input_path}")

            # 加载NRRD文件
            image_data, header = mio.load(input_path)

            # 创建NIfTI图像
            # 注意: medpy保存的NRRD通常是(x, y, z)顺序
            nifti_img = nib.Nifti1Image(image_data, np.eye(4))

            # 设置体素间距（如果header中有）
            if hasattr(header, 'get_voxel_spacing'):
                voxel_spacing = header.get_voxel_spacing()
                if len(voxel_spacing) >= 3:
                    # 设置NIfTI的pixdim
                    nifti_img.header['pixdim'][1:4] = voxel_spacing[:3]

            # 保存为NIfTI
            nib.save(nifti_img, output_path)

            print(f"  输出: {output_path}")
            print(f"  形状: {image_data.shape}")

            converted_files.append({
                'nrrd_file': nrrd_file,
                'nii_file': output_file,
                'shape': image_data.shape,
                'input_path': input_path,
                'output_path': output_path
            })

        except Exception as e:
            print(f"  错误: 转换 {nrrd_file} 失败 - {e}")

    print(f"\n转换完成!")
    print(f"共转换 {len(converted_files)}/{len(nrrd_files)} 个文件")
    print(f"输出目录: {output_dir}")

    return converted_files


def batch_nrrd_to_nii_simple(input_dir, output_suffix="_nii"):
    """
    简化版批量转换，直接在原目录生成.nii.gz文件

    参数:
    -----------
    input_dir : str
        包含NRRD文件的输入目录
    output_suffix : str
        输出文件的名称后缀
    """
    import os

    # 获取所有nrrd文件
    nrrd_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith('.nrrd'):
            nrrd_files.append(file)
        elif file.lower().endswith('.nhdr'):
            nrrd_files.append(file)

    if not nrrd_files:
        print(f"在目录 {input_dir} 中未找到NRRD文件")
        return []

    print(f"找到 {len(nrrd_files)} 个NRRD文件")

    # 批量转换
    for nrrd_file in nrrd_files:
        try:
            # 构建完整路径
            input_path = os.path.join(input_dir, nrrd_file)

            # 生成输出文件名
            base_name = os.path.splitext(nrrd_file)[0]
            if base_name.endswith('.nrrd'):
                base_name = base_name[:-5]
            output_file = f"{base_name}{output_suffix}.nii.gz"
            output_path = os.path.join(input_dir, output_file)

            # 加载并转换
            image_data, header = mio.load(input_path)
            nifti_img = nib.Nifti1Image(image_data, np.eye(4))

            # 设置体素间距
            if hasattr(header, 'get_voxel_spacing'):
                voxel_spacing = header.get_voxel_spacing()
                if len(voxel_spacing) >= 3:
                    nifti_img.header['pixdim'][1:4] = voxel_spacing[:3]

            # 保存
            nib.save(nifti_img, output_path)

            print(f"✓ 已转换: {nrrd_file} -> {output_file} ({image_data.shape})")

        except Exception as e:
            print(f"✗ 转换失败: {nrrd_file} - {e}")

    print(f"\n完成! 共处理 {len(nrrd_files)} 个文件")

    return nrrd_files


# 使用示例
if __name__ == "__main__":
    # 示例1: 将nrrd文件转换到单独的输出目录
    input_folder = "nrrd_label"
    batch_nrrd_to_nii(input_folder)

    # 示例2: 指定输出目录
    # batch_nrrd_to_nii(input_folder, "/path/to/output/nifti")

    # 示例3: 简化版，在原目录生成.nii.gz文件
    # batch_nrrd_to_nii_simple(input_folder)