import os
import json
from pathlib import Path
from model_utils import process_image
from PIL import Image
import glob

def save_caption(img_path, caption):
    """保存 caption 到同名的 .caption 文件"""
    try:
        caption_path = str(img_path).rsplit('.', 1)[0] + '.caption'
        with open(caption_path, 'w', encoding='utf-8') as f:
            f.write(caption.strip())
        return True, caption_path
    except Exception as e:
        return False, str(e)

def list_folders(current_path):
    """列出指定路径下的所有文件夹"""
    try:
        items = os.listdir(current_path)
        full_paths = [os.path.join(current_path, item) for item in items]
        folders = [p for p in full_paths if os.path.isdir(p)]
        if current_path not in folders:
            folders.insert(0, current_path)
        return folders
    except Exception as e:
        print(f"读取文件夹出错: {e}")
        return [current_path]

def json_folder_process(folder_path):
    try:
        if not folder_path:
            return "请选择文件夹"
            
        images_data = []
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            caption_path = os.path.join(folder_path, os.path.splitext(img_file)[0] + '.caption')
            
            if os.path.exists(caption_path):
                with open(caption_path, 'r', encoding='utf-8') as f:
                    caption_text = f.read().strip()
            else:
                caption_text = ""
                
            images_data.append({
                "file_name": img_file,
                "text": caption_text
            })
        
        output_path = os.path.join(folder_path, 'metadata.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in images_data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
                
        return f"成功处理 {len(images_data)} 张图片，已保存至 {output_path}"
        
    except Exception as e:
        return f"处理过程中出错: {str(e)}" 
    
def update_folder_path(folder_path):
    return list_folders(folder_path)

def process_folder(folder_path, task_type, user_prompt):
    """处理文件夹中的所有图片"""
    try:
        # 确保模型已加载
        load_model()
        
        if not folder_path:
            return "请选择文件夹"
        
        image_files = glob.glob(f"{folder_path}/*.jpg") + glob.glob(f"{folder_path}/*.png")
        if not image_files:
            return "文件夹中没有找到图片文件"
        
        results = []
        total = len(image_files)
        results.append(f"找到 {total} 张图片，开始处理...\n")
        yield "\n".join(results)
        
        for idx, img_path in enumerate(image_files, 1):
            try:
                progress_msg = f"[{idx}/{total}] 正在处理: {Path(img_path).name}"
                results.append(progress_msg)
                yield "\n".join(results)
                
                image = Image.open(img_path)
                caption = process_image(image, task_type, user_prompt)
                
                success, result = save_caption(img_path, caption)
                if success:
                    save_msg = f"已保存: {Path(result).name}"
                else:
                    save_msg = f"保存失败: {result}"
                
                results.append(f"Caption: {caption}")
                results.append(f"{save_msg}\n")
                yield "\n".join(results)
                
            except Exception as e:
                error_msg = f"处理 {Path(img_path).name} 时出错: {str(e)}\n"
                results.append(error_msg)
                yield "\n".join(results)
        
        results.append(f"处理完成！共处理 {total} 张图片。")
        yield "\n".join(results)
    except Exception as e:
        yield f"处理出错: {str(e)}"
