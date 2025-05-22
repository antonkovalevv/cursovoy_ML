import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import whisper
import tempfile
import os
from moviepy.editor import VideoFileClip, AudioFileClip
import json
import numpy as np
import soundfile as sf
from scipy import signal
import traceback
import requests
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

load_dotenv()

IO_API_KEY = "io-v2-eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJvd25lciI6IjcyMjNkMzhiLWY0NDItNGRlZS1hYmQ0LThlZDM0NjhlZjI5NiIsImV4cCI6NDkwMDY0MTM3M30.Nw5GXlaeH0u9dcTsFgUaPoT-2fYo6UlBoqPeV6TjKqZMyLKLrqZkN2FfIOUo-9kKyT5_mw1EiyFww52LwrtGvQ"
IO_API_BASE = "https://api.intelligence.io.solutions/api/v1"
SAMPLE_RATE = 16000

class AudioAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализатор Медиаконтента")
        self.root.geometry("1000x800")
        
        # Переменные
        self.audio_duration = 0
        self.model = None
        self.current_file = None
        
        # Создание интерфейса
        self.create_widgets()
        
        # Загрузка модели в отдельном потоке
        threading.Thread(target=self.load_model, daemon=True).start()

    def create_widgets(self):
        # Верхняя панель
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)
        
        ttk.Label(top_frame, text="Выберите файл для анализа:").pack(side=tk.LEFT)
        ttk.Button(top_frame, text="Открыть файл", command=self.select_file).pack(side=tk.LEFT, padx=5)
        
        # Выбор модели
        model_frame = ttk.Frame(self.root, padding="10")
        model_frame.pack(fill=tk.X)
        
        ttk.Label(model_frame, text="Модель Whisper:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value="tiny")
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, values=["tiny", "base", "small"])
        model_combo.pack(side=tk.LEFT, padx=5)
        
        # Прогресс
        self.progress_frame = ttk.Frame(self.root, padding="10")
        self.progress_frame.pack(fill=tk.X)
        
        self.status_label = ttk.Label(self.progress_frame, text="Готов к работе")
        self.status_label.pack(fill=tk.X)
        
        self.progress = ttk.Progressbar(self.progress_frame, mode='determinate')
        self.progress.pack(fill=tk.X)
        
        # Текст транскрипции
        transcript_frame = ttk.LabelFrame(self.root, text="Транскрипция", padding="10")
        transcript_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.transcript_text = scrolledtext.ScrolledText(transcript_frame, height=10)
        self.transcript_text.pack(fill=tk.BOTH, expand=True)
        
        # Результаты анализа
        results_frame = ttk.LabelFrame(self.root, text="Результаты анализа", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=10)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Фрейм для графика
        self.plot_frame = ttk.Frame(self.root, padding="10")
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

    def load_model(self):
        try:
            self.update_status("Загрузка модели Whisper...")
            self.model = whisper.load_model("tiny")
            self.update_status("Модель загружена. Готов к работе")
        except Exception as e:
            self.show_error("Ошибка загрузки модели", str(e))

    def select_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Аудио/Видео файлы", "*.mp3 *.wav *.mp4")]
        )
        if file_path:
            self.current_file = file_path
            threading.Thread(target=self.process_file, args=(file_path,), daemon=True).start()

    def process_file(self, file_path):
        try:
            self.update_status("Обработка файла...")
            self.update_progress(10)
            
            # Создаем временный файл
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, f"temp_input{os.path.splitext(file_path)[1]}")
            
            with open(file_path, 'rb') as src, open(temp_path, 'wb') as dst:
                dst.write(src.read())
            
            self.update_progress(20)
            
            # Обработка аудио/видео
            if file_path.endswith('.mp4'):
                self.update_status("Извлечение аудио из видео...")
                audio_path = self.extract_audio_from_video(temp_path)
            else:
                audio_path = temp_path
            
            self.update_progress(40)
            
            # Транскрибация
            self.update_status("Распознавание речи...")
            transcript = self.transcribe_audio(audio_path)
            
            if not transcript:
                self.show_error("Ошибка", "Не удалось получить текст из аудио")
                return
            
            self.update_progress(70)
            self.transcript_text.delete(1.0, tk.END)
            self.transcript_text.insert(tk.END, transcript)
            
            # Анализ
            self.update_status("Анализ содержания...")
            analysis = self.analyze_content(transcript)
            
            if analysis and "segments" in analysis:
                self.update_progress(90)
                self.visualize_segments(analysis)
                self.show_segments_info(analysis)
                self.update_status("Анализ завершен!")
                self.update_progress(100)
            else:
                self.show_error("Ошибка", "Не удалось проанализировать текст")
            
        except Exception as e:
            self.show_error("Ошибка обработки", str(e))
        finally:
            try:
                os.remove(temp_path)
                if file_path.endswith('.mp4') and 'audio_path' in locals():
                    os.remove(audio_path)
            except:
                pass

    def extract_audio_from_video(self, video_path):
        video = VideoFileClip(video_path)
        self.audio_duration = video.duration
        
        temp_dir = tempfile.mkdtemp()
        temp_audio_path = os.path.join(temp_dir, 'temp_audio.wav')
        
        try:
            video.audio.write_audiofile(temp_audio_path, fps=SAMPLE_RATE, codec='pcm_s16le')
        finally:
            video.close()
        
        return temp_audio_path

    def process_audio_data(self, file_path):
        try:
            data, samplerate = sf.read(file_path)
        except Exception as e:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(file_path)
            samples = audio.get_array_of_samples()
            data = np.array(samples).astype(np.float32) / 32768.0
            samplerate = audio.frame_rate
        
        self.audio_duration = len(data) / samplerate
        
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
        if samplerate != SAMPLE_RATE:
            number_of_samples = round(len(data) * float(SAMPLE_RATE) / samplerate)
            data = signal.resample(data, number_of_samples)
        
        if np.abs(data).max() > 1.0:
            data = data / np.abs(data).max()
            
        return data

    def transcribe_audio(self, audio_path):
        try:
            audio = self.process_audio_data(audio_path)
            if len(audio) == 0:
                raise ValueError("Аудио файл пустой или поврежден")
            
            result = self.model.transcribe(audio)
            return result["text"]
        except Exception as e:
            self.show_error("Ошибка транскрибации", str(e))
            return None

    def analyze_content(self, transcript):
        try:
            messages = [
                {
                    "role": "system",
                    "content": """Ты - эксперт по анализу текста. Анализируй текст на русском языке.
                    ВАЖНО: Твой ответ должен быть строго в следующем формате JSON, без дополнительного текста:
                    {
                        "segments": [
                            {
                                "title": "Название темы на русском",
                                "description": "Описание темы на русском",
                                "percentage": число от 0 до 100
                            }
                        ]
                    }"""
                },
                {
                    "role": "user",
                    "content": f"Проанализируй следующий текст и раздели его на тематические сегменты с указанием их примерной длительности в процентах:\n\n{transcript}"
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
                raise Exception(f"API вернул ошибку {response.status_code}")

            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            segments_data = json.loads(content)

            # Нормализация процентов
            total = sum(segment["percentage"] for segment in segments_data["segments"])
            if total != 100:
                scale = 100 / total
                for segment in segments_data["segments"]:
                    segment["percentage"] = round(segment["percentage"] * scale, 1)

            # Добавление временных меток
            current_time = 0
            for segment in segments_data["segments"]:
                segment_duration = (segment["percentage"] / 100) * self.audio_duration
                segment["start_time"] = self.format_timestamp(current_time)
                current_time += segment_duration
                segment["end_time"] = self.format_timestamp(current_time)

            return segments_data

        except Exception as e:
            self.show_error("Ошибка анализа", str(e))
            return None

    def visualize_segments(self, segments):
        # Очищаем предыдущий график
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        # Создаем новый график
        fig, ax = plt.subplots(figsize=(10, 3))
        
        y_position = 0
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, segment in enumerate(segments["segments"]):
            width = float(segment['percentage'])
            ax.barh(y_position, width, color=colors[i % len(colors)])
            ax.text(width/2, y_position, 
                   f"{segment['title']} ({width}%)\n[{segment['start_time']} - {segment['end_time']}]",
                   ha='center', va='center')
        
        ax.set_title("Распределение тематических сегментов")
        ax.set_xlabel("Процент от общей длительности")
        ax.set_yticks([])
        
        # Встраиваем график в tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def show_segments_info(self, analysis):
        self.results_text.delete(1.0, tk.END)
        for segment in analysis["segments"]:
            self.results_text.insert(tk.END, 
                f"Тема: {segment['title']}\n"
                f"Время: {segment['start_time']} - {segment['end_time']} ({segment['percentage']}%)\n"
                f"Описание: {segment['description']}\n\n"
            )

    def format_timestamp(self, seconds):
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def update_status(self, message):
        self.root.after(0, lambda: self.status_label.config(text=message))

    def update_progress(self, value):
        self.root.after(0, lambda: self.progress.configure(value=value))

    def show_error(self, title, message):
        messagebox.showerror(title, message)

def main():
    root = tk.Tk()
    app = AudioAnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 