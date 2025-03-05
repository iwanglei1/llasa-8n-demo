from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from xcodec2.modeling_xcodec2 import XCodec2Model

class TTSModels:
    def __init__(self):
        self.llasa_8b = '/mnt/desk/allan/allan_model/Llasa-8B'
        self.model_path = "/mnt/desk/allan/allan_model/xcodec2"
        
        # 初始化tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(self.llasa_8b)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.llasa_8b,
            torch_dtype=torch.float16
        )
        self.model.eval()
        self.model.to('cuda')
        
        # 初始化Codec模型
        self.codec_model = XCodec2Model.from_pretrained(self.model_path)
        self.codec_model.eval().cuda()
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model(self):
        return self.model
    
    def get_codec_model(self):
        return self.codec_model

# 创建全局模型实例
tts_models = TTSModels()