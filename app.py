import streamlit as st
import whisper
import tempfile
import os
from moviepy.editor import VideoFileClip, AudioFileClip
import plotly.graph_objects as go
from dotenv import load_dotenv
import json
import numpy as np
import soundfile as sf
from scipy import signal
import traceback
import requests
import time
import re

# Загрузка переменных окружения
load_dotenv()

# Инициализация IO Intelligence клиента
IO_API_KEY = "io-v2-eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJvd25lciI6IjcyMjNkMzhiLWY0NDItNGRlZS1hYmQ0LThlZDM0NjhlZjI5NiIsImV4cCI6NDkwMDY0MTM3M30.Nw5GXlaeH0u9dcTsFgUaPoT-2fYo6UlBoqPeV6TjKqZMyLKLrqZkN2FfIOUo-9kKyT5_mw1EiyFww52LwrtGvQ"
IO_API_BASE = "https://api.intelligence.io.solutions/api/v1"

SAMPLE_RATE = 16000  # Whisper ожидает частоту дискретизации 16kHz

# Загрузка модели Whisper
@st.cache_resource
def load_whisper_model():
    """Загрузка модели Whisper с кэшированием"""
    return whisper.load_model("tiny")  # Используем tiny модель для быстрой работы

def process_audio(file_path, status_placeholder):
    """Обработка аудио файла"""
    try:
        status_placeholder.text("🎵 Чтение аудио файла...")
        
        # Преобразование пути в правильную кодировку
        file_path = os.path.abspath(os.path.normpath(file_path))
        
        # Читаем аудио файл с помощью soundfile
        try:
            data, samplerate = sf.read(file_path)
        except Exception as e:
            print(f"Ошибка при чтении файла через soundfile: {str(e)}")
            # Пробуем альтернативный метод через pydub
            from pydub import AudioSegment
            audio = AudioSegment.from_file(file_path)
            samples = audio.get_array_of_samples()
            data = np.array(samples).astype(np.float32) / 32768.0  # нормализация
            samplerate = audio.frame_rate
        
        # Сохраняем длительность аудио в секундах
        st.session_state.audio_duration = len(data) / samplerate
        
        # Если стерео, конвертируем в моно
        if len(data.shape) > 1:
            status_placeholder.text("🎵 Конвертация стерео в моно...")
            data = data.mean(axis=1)
        
        # Убеждаемся, что данные в формате float32
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
        # Ресемплинг до 16kHz если нужно
        if samplerate != SAMPLE_RATE:
            status_placeholder.text(f"🎵 Преобразование частоты дискретизации с {samplerate}Hz до {SAMPLE_RATE}Hz...")
            number_of_samples = round(len(data) * float(SAMPLE_RATE) / samplerate)
            data = signal.resample(data, number_of_samples)
        
        # Нормализация
        if np.abs(data).max() > 1.0:
            status_placeholder.text("🎵 Нормализация аудио...")
            data = data / np.abs(data).max()
            
        return data
            
    except Exception as e:
        st.error(f"Ошибка при обработке аудио файла: {str(e)}")
        print(f"Детали ошибки: {traceback.format_exc()}")
        raise

def extract_audio_from_video(video_path):
    """Извлечение аудио из видео"""
    try:
        # Преобразование пути в правильную кодировку
        video_path = os.path.abspath(os.path.normpath(video_path))
        
        video = VideoFileClip(video_path)
        # Сохраняем длительность видео
        st.session_state.audio_duration = video.duration
        
        # Создаем временный файл с ASCII именем
        temp_dir = tempfile.mkdtemp()
        temp_audio_path = os.path.join(temp_dir, 'temp_audio.wav')
        
        try:
            video.audio.write_audiofile(temp_audio_path, fps=SAMPLE_RATE, codec='pcm_s16le')
        finally:
            video.close()
        
        return temp_audio_path
        
    except Exception as e:
        st.error(f"Ошибка при извлечении аудио из видео: {str(e)}")
        print(f"Детали ошибки: {traceback.format_exc()}")
        raise

def convert_to_wav(file_path):
    """Конвертация аудио в WAV"""
    try:
        # Преобразование пути в правильную кодировку
        file_path = os.path.abspath(os.path.normpath(file_path))
        
        if file_path.endswith('.mp3'):
            # Создаем временный файл с ASCII именем
            temp_dir = tempfile.mkdtemp()
            temp_wav = os.path.join(temp_dir, 'temp_audio.wav')
            
            try:
                audio = AudioFileClip(file_path)
                audio.write_audiofile(temp_wav, fps=SAMPLE_RATE, codec='pcm_s16le')
            finally:
                if 'audio' in locals():
                    audio.close()
            
            return temp_wav
        return file_path
        
    except Exception as e:
        st.error(f"Ошибка при конвертации аудио: {str(e)}")
        print(f"Детали ошибки: {traceback.format_exc()}")
        raise

def transcribe_audio(audio_path, model, status_placeholder, progress_bar):
    """Транскрибация аудио в текст"""
    try:
        # Конвертируем в WAV если нужно
        wav_path = convert_to_wav(audio_path)
        
        try:
            # Обработка аудио
            audio = process_audio(wav_path, status_placeholder)
            
            if audio is None:
                return None
                
            # Проверяем длину аудио
            if len(audio) == 0:
                raise ValueError("Аудио файл пустой или поврежден")
            
            # Оценка времени выполнения
            audio_duration = len(audio) / SAMPLE_RATE
            estimated_time = audio_duration * 0.5
            
            status_placeholder.text(f"🎙️ Распознавание речи... (примерное время: {estimated_time:.1f} сек)")
            
            # Транскрибация с получением временных меток
            start_time = time.time()
            result = model.transcribe(
                audio,
                language="ru",
                verbose=True
            )
            
            elapsed_time = time.time() - start_time
            status_placeholder.text(f"✅ Распознавание завершено за {elapsed_time:.1f} сек")
            
            # Формируем структурированный результат
            transcript_data = {
                "full_text": result["text"],
                "segments": []
            }
            
            # Добавляем сегменты с временными метками
            for segment in result["segments"]:
                transcript_data["segments"].append({
                    "text": segment["text"].strip(),
                    "start": float(segment["start"]),
                    "end": float(segment["end"])
                })
            
            return transcript_data
            
        finally:
            if wav_path != audio_path:
                try:
                    os.remove(wav_path)
                except:
                    pass
                    
    except Exception as e:
        st.error(f"Ошибка при транскрибации: {str(e)}")
        print(f"Детали ошибки: {traceback.format_exc()}")
        return None

def clean_json_string(s):
    """Очистка строки JSON от недопустимых символов"""
    # Удаление управляющих символов, кроме разрешенных в JSON
    # Оставляем только разрешенные управляющие символы: \b \f \n \r \t
    return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', s)

def format_timestamp(seconds):
    """Форматирование времени в формат MM:SS"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def analyze_content(transcript_data):
    """Анализ текста с помощью Intelligence.io"""
    try:       
        # Создаем структурированный текст с метками времени
        text_with_timestamps = ""
        current_time = 0
        
        # Добавляем информацию о сегментах с их временными метками
        for segment in transcript_data["segments"]:
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            text_with_timestamps += f"[{start_time} - {end_time}] {segment['text']}\n"
        
        print("\n=== Текст с метками времени ===")
        print(text_with_timestamps)
        
        messages = [
            {
                "role": "system",
                "content": """Ты - эксперт по анализу текста. Анализируй текст на русском языке.
                ВАЖНО: Твой ответ должен быть строго в следующем формате JSON, без дополнительного текста:
                {
                    "segments": [
                        {
                            "title": "Название темы на русском",
                            "description": "Краткое описание основных моментов темы",
                            "start_time": "MM:SS",
                            "end_time": "MM:SS",
                            "percentage": число от 0 до 100
                        }
                    ]
                }
                
                Инструкции по анализу:
                1. Используй временные метки из исходного текста ([MM:SS - MM:SS])
                2. Объединяй близкие по смыслу части в один сегмент
                3. Для объединенных сегментов используй:
                   - start_time первого сегмента группы
                   - end_time последнего сегмента группы
                4. Percentage вычисляй на основе длительности сегмента
                5. В description пиши только краткое содержание темы, БЕЗ временных меток"""
            },
            {
                "role": "user",
                "content": f"Проанализируй следующий текст с временными метками и раздели его на тематические сегменты. Используй временные метки из текста для определения start_time и end_time:\n\n{text_with_timestamps}"
            }
        ]
        
        data = {
            "model": "deepseek-ai/DeepSeek-R1",
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 2000,
            "response_format": {"type": "json_object"}
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {IO_API_KEY}",
            "Accept": "application/json",
            "Accept-Language": "ru"
        }

        response = requests.post(
            f"{IO_API_BASE}/chat/completions",
            headers=headers,
            json=data,
            verify=False,
            timeout=30
        )


        if response.status_code != 200:
            raise Exception(f"API вернул ошибку {response.status_code}: {response.text}")

        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        segments_data = json.loads(content)

        # Проверяем и нормализуем проценты
        if "segments" in segments_data:
            total = sum(segment["percentage"] for segment in segments_data["segments"])
            if total != 100:
                scale = 100 / total
                for segment in segments_data["segments"]:
                    segment["percentage"] = round(segment["percentage"] * scale, 1)

        return segments_data

    except Exception as e:
        print("\n=== Детали ошибки ===")
        print(traceback.format_exc())
        st.error(f"Произошла ошибка при анализе текста: {str(e)}")
        raise

def visualize_segments(segments):
    """Создание визуализации сегментов"""
    if not segments or "segments" not in segments:
        st.error("Нет данных для визуализации")
        return None
        
    fig = go.Figure()
    
    y_position = 0
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    start_pos = 0
    for i, segment in enumerate(segments["segments"]):
        try:
            width = float(segment['percentage'])
            time_info = f" [{segment.get('start_time', '00:00')} - {segment.get('end_time', '00:00')}]"
            fig.add_trace(go.Bar(
                x=[width],
                y=[y_position],
                orientation='h',
                name=segment['title'],
                text=f"{segment['title']} ({width}%){time_info}",
                marker_color=colors[i % len(colors)],
                hovertext=segment['description'],
                textposition='inside',
            ))
            start_pos += width
        except (KeyError, ValueError) as e:
            st.warning(f"Пропущен сегмент из-за ошибки в данных: {str(e)}")
            continue
    
    fig.update_layout(
        title="Распределение тематических сегментов",
        showlegend=False,
        height=150,
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis=dict(showticklabels=False),
        xaxis=dict(title="Процент от общей длительности")
    )
    
    return fig

def main():
    st.set_page_config(initial_sidebar_state="collapsed")
    
    st.title("Анализатор Медиаконтента")
    
    if 'audio_duration' not in st.session_state:
        st.session_state.audio_duration = 0
    
    model_option = st.sidebar.selectbox(
        'Выберите модель Whisper:',
        ('tiny', 'base', 'small'),
        help="tiny - быстрая (менее точная); base - средняя; small - медленная (более точная)"
    )
    
    st.write("Загрузите аудио или видео файл для анализа")
    st.info("💡 Рекомендуемый размер файла: до 10 минут для лучшей производительности")
    
    uploaded_file = st.file_uploader("Выберите файл", type=['mp3', 'wav', 'mp4'])
    
    if uploaded_file is not None:
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.write(f"📁 Размер файла: {file_size_mb:.1f} MB")
        
        try:
            # Этап 1: Сохранение файла
            status_placeholder.text("📥 Загрузка файла...")
            progress_bar.progress(10)
            
            # Создаем временный файл с ASCII именем
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, f"temp_input{os.path.splitext(uploaded_file.name)[1]}")
            
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            progress_bar.progress(20)
            
            try:
                # Этап 2: Обработка аудио
                if uploaded_file.name.endswith('.mp4'):
                    status_placeholder.text("🎬 Извлечение аудио из видео...")
                    audio_path = extract_audio_from_video(temp_path)
                else:
                    status_placeholder.text("🎵 Подготовка аудио...")
                    audio_path = temp_path
                
                progress_bar.progress(40)
                
                # Этап 3: Загрузка модели
                status_placeholder.text("🤖 Загрузка модели Whisper...")
                model = load_whisper_model()
                progress_bar.progress(50)
                
                # Этап 4: Транскрибация
                transcript_data = transcribe_audio(audio_path, model, status_placeholder, progress_bar)
                progress_bar.progress(70)
                
                if not transcript_data or not transcript_data["full_text"].strip():
                    status_placeholder.text("❌ Ошибка: Не удалось получить текст")
                    st.error("Не удалось получить текст из аудио. Возможно, файл не содержит речи.")
                    return
                
                st.subheader("Транскрипция")
                st.text_area("Текст", transcript_data["full_text"], height=200)
                
                # Этап 5: Анализ контента
                status_placeholder.text("🧠 Анализ содержания...")
                progress_bar.progress(80)
                analysis = analyze_content(transcript_data)
                
                if analysis and "segments" in analysis:
                    progress_bar.progress(90)
                    
                    # Этап 6: Визуализация
                    status_placeholder.text("📊 Создание визуализации...")
                    st.subheader("Тематические сегменты")
                    fig = visualize_segments(analysis)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Этап 7: Вывод результатов
                    st.subheader("Детальная информация по сегментам")
                    for segment in analysis["segments"]:
                        with st.expander(f"{segment['title']} ({segment['percentage']}%) - [{segment.get('start_time', '00:00')} - {segment.get('end_time', '00:00')}]"):
                            st.write(segment['description'])
                    
                    # Завершение
                    progress_bar.progress(100)
                    status_placeholder.text("✅ Анализ завершен!")
                else:
                    status_placeholder.text("❌ Ошибка: Не удалось выделить сегменты")
                    st.warning("Не удалось выделить тематические сегменты в тексте.")
            
            finally:
                # Очистка временных файлов
                try:
                    os.remove(temp_path)
                    if uploaded_file.name.endswith('.mp4') and 'audio_path' in locals():
                        os.remove(audio_path)
                except Exception as e:
                    st.warning(f"Не удалось удалить временные файлы: {str(e)}")
        
        except Exception as e:
            status_placeholder.text("❌ Произошла ошибка")
            st.error(f"Произошла ошибка при обработке файла: {str(e)}")
            st.error(traceback.format_exc())

if __name__ == "__main__":
    main() 