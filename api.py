from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from starlette.background import BackgroundTask
from fastapi.responses import FileResponse
import torch
import soundfile as sf
import io
import numpy as np
from models import tts_models
import librosa
import os
import uuid
import logging

app = FastAPI()

def ids_to_speech_tokens(speech_ids):
    speech_tokens_str = []
    for speech_id in speech_ids:
        speech_tokens_str.append(f"<|s_{speech_id}|>")
    return speech_tokens_str

def extract_speech_ids(speech_tokens_str):
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            num_str = token_str[4:-2]
            num = int(num_str)
            speech_ids.append(num)
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids

def split_text(text):
    """将文本按照句号、分号等标点符号分割成短句"""
    import re
    # 匹配中英文句号、分号、感叹号、问号
    pattern = r'[。。；;！!？?]'
    sentences = re.split(pattern, text)
    # 过滤空字符串
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

import logging
from fastapi import HTTPException

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 临时文件路径配置
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/tts/with_reference")
async def tts_with_reference(file: UploadFile = File(...), input_text: str = Form(...), reference_text: str = Form(None)):
    try:
        if not input_text.strip():
            raise HTTPException(status_code=400, detail="输入文本不能为空")

        if not file.filename.lower().endswith('.wav'):
            raise HTTPException(status_code=400, detail="只支持WAV格式的音频文件")

        logger.info(f"开始处理参考音频文件: {file.filename}")
        
        # 读取上传的音频文件
        content = await file.read()
        try:
            audio_data = sf.read(io.BytesIO(content))
        except Exception as e:
            logger.error(f"音频文件读取失败: {str(e)}")
            raise HTTPException(status_code=400, detail="音频文件格式错误或已损坏")

        # 转换音频数据格式
        prompt_wav = torch.from_numpy(audio_data[0]).float().unsqueeze(0)  

        # 编码参考音频
        with torch.no_grad():
            vq_code_prompt = tts_models.get_codec_model().encode_code(input_waveform=prompt_wav)
            vq_code_prompt = vq_code_prompt[0,0,:]
            speech_ids_prefix = ids_to_speech_tokens(vq_code_prompt)

            # 如果提供了参考文本，将其与目标文本拼接
            if reference_text and reference_text.strip():
                combined_text = reference_text.strip() + input_text
            else:
                combined_text = input_text

            formatted_text = f"<|TEXT_UNDERSTANDING_START|>{combined_text}<|TEXT_UNDERSTANDING_END|>"

            # 准备输入
            chat = [
                {"role": "user", "content": "Convert the text to speech:" + formatted_text},
                {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + ''.join(speech_ids_prefix)}
            ]

            input_ids = tts_models.get_tokenizer().apply_chat_template(
                chat, 
                tokenize=True, 
                return_tensors='pt', 
                continue_final_message=True
            )
            input_ids = input_ids.to('cuda')
            speech_end_id = tts_models.get_tokenizer().convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')

            # 生成语音
            outputs = tts_models.get_model().generate(
                input_ids,
                max_length=2048,
                eos_token_id=speech_end_id,
                do_sample=True,
                top_p=1,
                temperature=0.8,
            )

            generated_ids = outputs[0][input_ids.shape[1]-len(speech_ids_prefix):-1]
            speech_tokens = tts_models.get_tokenizer().batch_decode(generated_ids, skip_special_tokens=True)
            speech_tokens = extract_speech_ids(speech_tokens)
            speech_tokens = torch.tensor(speech_tokens).cuda().unsqueeze(0).unsqueeze(0)
            gen_wav = tts_models.get_codec_model().decode_code(speech_tokens)
            
            final_wav = gen_wav[0, 0, :].cpu().numpy()

            # 生成唯一的输出文件名
            output_filename = f"{uuid.uuid4()}.wav"
            output_path = os.path.join(TEMP_DIR, output_filename)
            
            # 保存生成的音频
            sf.write(output_path, final_wav, 16000)
            logger.info(f"音频生成完成，保存至: {output_path}")
            
            return FileResponse(
                output_path,
                media_type="audio/wav",
                headers={"Content-Disposition": f"attachment; filename={output_filename}"},
                background=BackgroundTask(lambda: os.unlink(output_path))
            )
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts/direct")
async def tts_direct(input_text: str):
    try:
        if not input_text.strip():
            raise HTTPException(status_code=400, detail="输入文本不能为空")
            
        logger.info(f"开始处理文本: {input_text[:50]}...")
        
        with torch.no_grad():
            formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"

            chat = [
                {"role": "user", "content": "Convert the text to speech:" + formatted_text},
                {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"}
            ]

            input_ids = tts_models.get_tokenizer().apply_chat_template(
                chat, 
                tokenize=True, 
                return_tensors='pt', 
                continue_final_message=True
            )
            input_ids = input_ids.to('cuda')
            speech_end_id = tts_models.get_tokenizer().convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')

            outputs = tts_models.get_model().generate(
                input_ids,
                max_length=4096,
                eos_token_id=speech_end_id,
                do_sample=True,
                top_p=1,
                temperature=1,
            )

            generated_ids = outputs[0][input_ids.shape[1]:-1]
            speech_tokens = tts_models.get_tokenizer().batch_decode(generated_ids, skip_special_tokens=True)
            speech_tokens = extract_speech_ids(speech_tokens)
            speech_tokens = torch.tensor(speech_tokens).cuda().unsqueeze(0).unsqueeze(0)
            gen_wav = tts_models.get_codec_model().decode_code(speech_tokens)
            
            final_wav = gen_wav[0, 0, :].cpu().numpy()

            # 生成唯一的输出文件名
            output_filename = f"{uuid.uuid4()}.wav"
            output_path = os.path.join(TEMP_DIR, output_filename)
            
            # 保存生成的音频
            sf.write(output_path, final_wav, 16000)
            logger.info(f"音频生成完成，保存至: {output_path}")
            
            return FileResponse(
                output_path,
                media_type="audio/wav",
                headers={"Content-Disposition": f"attachment; filename={output_filename}"},
                background=BackgroundTask(lambda: os.unlink(output_path))
            )
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))