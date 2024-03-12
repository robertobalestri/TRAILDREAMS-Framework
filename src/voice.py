import logging
import os
from pathlib import Path
from collections import Counter
from pydub import AudioSegment
import glob
import torch
from TTS.api import TTS
from MovieInfo import MovieInfo
from common import get_project_dir, logger, VOICES_DIR, get_paths

# Function to add silence padding
def add_silence_padding(audio_path, padding_duration):
    audio = AudioSegment.from_wav(audio_path)
    silence = AudioSegment.silent(duration=padding_duration)
    padded_audio = silence + audio + silence
    padded_audio.export(audio_path, format="wav")

# Function to generate voices
def generate_voices(project_dir: Path, tts: TTS, voice_padding: int, voice_type: str) -> None:
    audio_dir = get_paths()["audios_dir"]
    if not audio_dir.exists():
        audio_dir.mkdir(parents=True)

    trailer_speech = []
    with open(project_dir / "trailer_speech_gpt_generated.txt", "r") as file:
        trailer_speech = file.read().split("\n")

    pattern = os.path.join(VOICES_DIR, f"{voice_type}_*.wav")
    speaker_wavs = glob.glob(pattern)
    speaker_wavs = [wav.replace("\\", "/") for wav in speaker_wavs]
    
    for idx, speech in enumerate(trailer_speech):
        audio_path = audio_dir / f"trailer_speech_{idx+1}.wav"
        logger.info(f"Generating audio {idx+1}")
        tts.tts_to_file(
            speech,
            speaker_wav=speaker_wavs[0],  # Assuming we're using the first match
            language="en",
            file_path=str(audio_path),
            split_sentences=False
        )
        if voice_padding > 0:
            add_silence_padding(str(audio_path), voice_padding * 1000)

# Updated function to map genres to voice types
def map_genres_to_voice_type(genres):
    genre_to_voice = {
        'Action': 'solemn',
        'Adventure': 'solemn',
        'Animation': 'calm',
        'Biography': 'informative',
        'Comedy': 'calm',
        'Crime': 'narrative',
        'Documentary': 'informative',
        'Drama': 'solemn',
        'Family': 'calm',
        'Fantasy': 'solemn',
        'Film-Noir': 'solemn',
        'History': 'informative',
        'Horror': 'horrorific',
        'Music': 'narrative',
        'Musical': 'narrative',
        'Mystery': 'solemn',
        'Romance': 'calm',
        'Sci-Fi': 'solemn',
        'Short': 'calm',
        'Sport': 'narrative',
        'Thriller': 'horrorific',
        'War': 'solemn',
        'Western': 'narrative'
    }

    voice_counts = Counter([genre_to_voice[genre] for genre in genres if genre in genre_to_voice])

    most_common_voice = voice_counts.most_common(1)[0][0] if voice_counts else 'calm'
    return most_common_voice

# Main function to generate voice based on the most suitable voice type
def voice_main(movie_info: MovieInfo, tts_model: str, voice_padding: int) -> None:
    logger.info("##### Starting step 5 voice generation #####")
    logger.info(f"Using TTS model: {tts_model}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS(model_name=tts_model).to(device)
    
    genres = movie_info.get_genres()
    
    voice_type = map_genres_to_voice_type(genres)
    
    logger.info(f"Selected voice type: {voice_type} for genres: {genres}")
    
    project_dir = get_project_dir()
    generate_voices(project_dir, tts, voice_padding, voice_type)