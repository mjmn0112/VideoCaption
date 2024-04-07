import moviepy.editor as mp
import whisper
import numpy as np
import torch

video_path = r"your video path"
output_path = r"sub location captions.srt"

# Load the video and extract the audio
video = mp.VideoFileClip(video_path)
audio = video.audio

# Write the audio to a file
audio_path = r"a path for audio"
audio.write_audiofile(audio_path)

try:
    model = whisper.load_model("small")
    audio_array = audio.to_soundarray(fps=video.fps)
    audio_tensor = torch.from_numpy(audio_array).float()
    result = model.transcribe(audio_tensor, fp16=False)

    # Generate captions in SRT format
    captions = []
    for segment in result["segments"]:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]
        captions.append(f"{len(captions) + 1}\n{start_time:.2f} --> {end_time:.2f}\n{text}\n\n")

    # Write captions to an SRT file
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(captions)

    print(f"Captions saved to: {output_path}")
except Exception as e:
    print(f"Error during transcription: {e}")
