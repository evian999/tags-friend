import torch
import logging
import ollama

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelError(Exception):
    """自定义模型相关错误"""
    pass

def process_image(image, model, user_prompt):
    """处理单张图片"""
    try:
        
        # 验证输入
        if image is None:
            raise ValueError("未提供图片")
        if not user_prompt:
            raise ValueError("未提供提示词")
            
        response = ollama.chat(
        model=model,
        messages=[{
            'role': 'user',
            'content': user_prompt,
            'images': [image]
        }]
    )
        
        # 返回结果
        return response['message']['content']
        
    except ModelError as e:
        logger.error(f"模型错误: {str(e)}")
        return f"模型错误: {str(e)}"
    except ValueError as e:
        logger.error(f"输入参数错误: {str(e)}")
        return f"输入错误: {str(e)}"
    except torch.cuda.OutOfMemoryError:
        logger.error("GPU内存不足")
        return "GPU内存不足，请稍后重试"
    except Exception as e:
        logger.error(f"处理图片时发生未知错误: {str(e)}")
        return f"处理出错: {str(e)}" 
    
def list_models():
    """列出所有可用的模型"""
    models = ollama.list()
    model_list = []
    for i in models['models']:
        model_list.append(i['name'])
        
    return model_list