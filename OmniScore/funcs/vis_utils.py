from matplotlib import pyplot as plt 
import os 
def plot_vbench_diff_results(folder,data,dimension):
    diff_results = [entry['best']['video_results'] - entry['worst']['video_results']for entry in data]
    diff_results  = sorted(diff_results,reverse=True)
    # worst_results = [entry['worst']['video_results'] for entry in data]
    plt.figure(figsize=(10, 6))
    plt.plot(diff_results, label='Diff Video Results')
    # plt.plot(worst_results, label='Worst Video Results')
    plt.xlabel('Index')
    plt.ylabel('Video Results')
    plt.title(f'{dimension} Diff Video Results')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(folder,f"{dimension}_video_results.png"))
    
import os
from PIL import Image
import math

def create_image_grid(input_folder, output_path):
    # 获取并按文件名排序所有PNG文件
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])
    images = [Image.open(os.path.join(input_folder, f)) for f in image_files]
    
    # 确定每个小图像的大小（假设所有图像大小相同）
    image_width, image_height = images[0].size
    
    # 计算正方形的边长（列数和行数）
    num_images = len(images)
    grid_size = math.ceil(math.sqrt(num_images))
    
    # 创建一个新的大图像
    grid_width = grid_size * image_width
    grid_height = grid_size * image_height
    grid_image = Image.new('RGB', (grid_width, grid_height))
    
    # 将每个小图像粘贴到大图像中
    for idx, image in enumerate(images):
        x = (idx % grid_size) * image_width
        y = (idx // grid_size) * image_height
        grid_image.paste(image, (x, y))
    
    # 保存大图像
    grid_image.save(output_path)

# 示例用法
if __name__=="__main__":
    input_folder = './vbench_feedback/validation_data'
    output_path = 'all_diff_results.png'
    create_image_grid(input_folder, output_path)
