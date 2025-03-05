import streamlit as st
import requests
import io
import soundfile as sf

st.title('语音合成系统')

# 文本输入
input_text = st.text_area('请输入要合成的文本：', height=100)

# 音频文件上传
reference_audio = st.file_uploader('上传参考音频（可选）', type=['wav'])

# 参考音频文本输入
reference_text = st.text_area('请输入参考音频的文本内容（可选）：', height=50)

if st.button('开始合成'):
    if input_text:
        try:
            if reference_audio:
                # 使用参考音频进行合成
                files = {'file': ('reference.wav', reference_audio.getvalue(), 'audio/wav')}
                response = requests.post(
                    'http://192.168.4.69:8001/tts/with_reference',
                    files=files,
                    data={
                        'input_text': input_text,
                        'reference_text': reference_text
                    }
                )
            else:
                # 直接合成
                response = requests.post(
                    'http://192.168.4.69:8001/tts/direct',
                    params={'input_text': input_text}
                )
            
            if response.status_code == 200:
                # 将响应内容转换为音频数据
                audio_data = response.content
                
                # 使用st.audio显示音频
                st.audio(audio_data, format='audio/wav')
                
                # 提供下载按钮
                st.download_button(
                    label='下载音频文件',
                    data=audio_data,
                    file_name='generated_audio.wav',
                    mime='audio/wav'
                )
            else:
                st.error('语音合成失败，请重试')
        except Exception as e:
            st.error(f'发生错误：{str(e)}')
    else:
        st.warning('请输入要合成的文本')

# 添加使用说明
st.markdown('''
### 使用说明
1. 在文本框中输入要合成的文本
2. 如果需要模仿特定声音，可以上传一段参考音频（WAV格式）
3. 如果上传了参考音频，请输入该音频对应的文本内容
4. 点击"开始合成"按钮进行语音合成
5. 合成完成后可以直接播放或下载音频文件
''')