import os

# 指定图片文件夹路径
folder_path = '../data/new/RP'

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件是否为.jpg格式且文件名中包含" (1)"
    if filename.endswith('.jpg') and " (1)" in filename:
        # 构建新的文件名，移除" (1)"
        new_filename = filename.replace(" (1)", "")
        # 定义旧文件和新文件的完整路径
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, new_filename)
        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f'Renamed: {filename} -> {new_filename}')