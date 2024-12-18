import os
from pathlib import Path
import glob
from PIL import Image
import gradio as gr
import json

from model_utils import process_image, list_models
from file_utils import list_folders, json_folder_process, update_folder_path, process_folder

# 设置环境变量
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# 设置本文件所在文件夹为工作文件夹
os.chdir(Path(__file__).parent)
current_path = os.path.dirname(os.path.abspath(__file__))

# 在文件开头添加自定义样式
custom_css = """
.no-background {
    border: 1px solid #e5e5e5 !important;
    background: transparent !important;
}
"""

def save_caption(img_path, caption):
    """保存 caption 到同名的 .caption 文件"""
    try:
        caption_path = str(img_path).rsplit('.', 1)[0] + '.caption'
        with open(caption_path, 'w', encoding='utf-8') as f:
            f.write(caption.strip())
        return True, caption_path
    except Exception as e:
        return False, str(e)

def list_folders(current_path=current_path):
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

def update_folder_path(current_path):
    """更新文件夹路径选项"""
    return gr.Dropdown(choices=list_folders(current_path), value=current_path)

def process_folder(folder_path, task_type, user_prompt):
    """处理文件夹中的所有图片"""
    try:
        # 确保模型已加载
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
        
def json_folder_process(folder_path):
    try:
        if not folder_path:
            return "请选择文件夹"
            
        # 初始化数据列表
        images_data = []
        
        # 获取所有图片文件
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in image_files:
            # 获取图片完整路径
            img_path = os.path.join(folder_path, img_file)
            # 获取对应的caption文件路径
            caption_path = os.path.join(folder_path, os.path.splitext(img_file)[0] + '.caption')
            
            # 如果caption文件存在，读取内容
            if os.path.exists(caption_path):
                with open(caption_path, 'r', encoding='utf-8') as f:
                    caption_text = f.read().strip()
            else:
                caption_text = ""
                
            # 构建数据字典
            image_data = {
                "file_name": img_file,
                "text": caption_text
            }
            images_data.append(image_data)
        
        # 将数据保存为JSONL文件
        output_path = os.path.join(folder_path, 'metadata.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in images_data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
                
        return f"成功处理 {len(images_data)} 张图片，已保存至 {output_path}"
        
    except Exception as e:
        return f"处理过程中出错: {str(e)}"
    

def update_prompt(choice):
    """根据选择更新提示词"""
    if choice == "caption":
        return "make caption for the image, within 20 words"
    else:
        return "generate detailed tags for the image, separate with comma"

# 创建 Gradio 界面
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("## 图像处理演示")
    
    with gr.Tabs():
        # 单张图片处理标签页
        with gr.TabItem("单张图片处理"):
            with gr.Row():
                # 左侧列：提示词和输出结果
                with gr.Column(scale=4):
                    model_path = gr.Dropdown(
                        choices=list_models(),
                        value="llama3.2-vision",
                        label="模型选择",
                        interactive=True
                    )
                    
                    task_type = gr.Radio(
                        choices=["caption", "tag"],
                        value="caption",
                        label="处理类型",
                        interactive=True
                    )
                    
                    single_prompt = gr.Textbox(
                        lines=2, 
                        placeholder="输入提示词...", 
                        value="make caption for the image, within 20 words",
                        label="提示词"
                    )
                    
                    single_button = gr.Button("处理图片", variant="primary")
                    single_output = gr.Textbox(
                        label="输出结果",
                        lines=3,
                        show_copy_button=True
                    )
                
                # 右侧列：图片上传
                with gr.Column(scale=4):
                    image_input = gr.Image(
                        type="filepath",
                        label="上传图片",
                        height=400
                    )
                
                single_button.click(
                    fn=process_image,
                    inputs=[image_input, model_path, task_type, single_prompt],
                    outputs=single_output
                )
            
        
        # 文件夹处理标签页
        with gr.TabItem("文件夹处理"):
            with gr.Row():
                # 左侧列：控制区域
                with gr.Column(scale=6):
                    # 路径选择
                    with gr.Group():
                        with gr.Row():
                            folder_input = gr.Dropdown(
                                choices=list_folders(current_path),
                                value=current_path,
                                label="图片文件夹路径",
                                # allow_custom_value=True,
                                container=False,
                                scale=20
                            )
                            refresh_btn = gr.Button(
                                value="🔄",
                                scale=1,
                                min_width=10
                            )
                    
                    task_type_folder = gr.Radio(
                        choices=["caption", "tag"],
                        value="caption",
                        label="处理类型",
                        interactive=True
                    )
                    
                    folder_prompt = gr.Textbox(
                        lines=2,
                        placeholder="输入提示词...",
                        value="make caption for the image, within 20 words",
                        label="提示词"
                    )
                    
                    folder_button = gr.Button("处理文件夹", variant="primary")
                
                # 右侧列：处理结果
                with gr.Column(scale=7):
                    folder_output = gr.Textbox(
                        label="批量处理结果",
                        lines=15,
                        show_copy_button=True
                    )
            
            # 事件处理
            folder_input.change(
                fn=update_folder_path,
                inputs=[folder_input],
                outputs=[folder_input]
            )
            
            refresh_btn.click(
                fn=update_folder_path,
                inputs=[folder_input],
                outputs=[folder_input]
            )
            
            task_type_folder.change(
                fn=update_prompt,
                inputs=[task_type_folder],
                outputs=[folder_prompt]
            )
            
            folder_button.click(
                fn=process_folder,
                inputs=[folder_input, task_type_folder, folder_prompt],
                outputs=[folder_output],
                show_progress=True
            )

        # json process
        with gr.TabItem("JSONL文件处理"):
            with gr.Row():
                # left side: input
                with gr.Column(scale=4):
                    with gr.Group():
                        with gr.Row():
                            jsonl_input = gr.Dropdown(
                                choices = list_folders(current_path),
                                value=current_path,
                                label="Caption文件夹路径",
                                allow_custom_value=True,
                                interactive=True,
                                container=False,
                                scale=20
                            )
                            refresh_btn = gr.Button(
                                value="🔄",
                                scale=1,
                                min_width=10
                            )
                    task_type_jsonl = gr.Radio(
                        choices=["caption"],
                        value="caption",
                        label="处理类型",
                        interactive=True
                    )

                    folder_button = gr.Button(
                        value="批量处理",
                        variant='primary'
                    )
                    
                # right side: output    
                with gr.Column(scale=5):
                    folder_output = gr.Textbox(
                        label="批量处理结果",
                        lines=15,
                        show_copy_button=True
                    )

            # events functions
            refresh_btn.click(
                fn=update_folder_path,
                inputs=[jsonl_input],
                outputs=[jsonl_input]
            )

            folder_button.click(
                fn=json_folder_process,
                inputs=[jsonl_input],
                outputs=[folder_output],
                show_progress=True
            )
        
if __name__ == "__main__":
    demo.launch() 