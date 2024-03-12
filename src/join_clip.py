import itertools
import logging
from glob import glob
from pathlib import Path
from moviepy.editor import VideoFileClip, concatenate_videoclips
from common import get_scenes_dir, logger, get_paths
import re
from MovieInfo import MovieInfo
from moviepy.editor import AudioFileClip
import os
import subprocess

def extract_timestamps(filename):
    # This function extracts the start and end timestamps from the filename
    match = re.search(r'clip_(\d+\.\d+)_(\d+\.\d+)_\d+\.mp4', filename)
    return (float(match.group(1)), float(match.group(2))) if match else (0, 0)



def join_clips(combination: list[str], trailer_dir: Path, quote_clips: list[str], trailer_number: int, fade_duration: float = 0.1):
    logger.info(f"Generating trailer {trailer_number}")
    trailer_path = Path(f"{trailer_dir}/trailer_{trailer_number}.mp4")

    print("COMBINATION=" + str(combination))
    
    # Sort the clips in the combination by the numerical part of the filename
    sorted_combination_clips = sorted(combination, key=lambda x: int(Path(x).stem.split("_")[3]))

    print("sorted_combination_clips=" + str(sorted_combination_clips))
    
    # Extract start and end times for quote clips, making sure to only pass the filename, not the full path
    # Extract start and end times for quote clips
    quote_times = [(extract_timestamps(Path(clip).name)) for clip in quote_clips]

    print("QUOTE TIMES: " + str(quote_times))

    # Filter out overlapping clips from the combination
    non_overlapping_clips = []
    for clip in sorted_combination_clips:
        filename = Path(clip).name
        parts = filename.split("_")
        if len(parts) >= 4:
            try:
                start = float(parts[1])
                end = float(parts[2])
                overlap = any(q_start < end and q_end > start for q_start, q_end in quote_times)
                if not overlap:
                    non_overlapping_clips.append(clip)
            except ValueError as e:
                print(str(parts))
                logger.error(f"Invalid filename format for clip: {filename}")
    
    print("SORTED_CLIPS= " + str(non_overlapping_clips))

    total_clips = len(non_overlapping_clips)
    interval = total_clips // (len(quote_clips) + 1)
    insert_positions = [interval * (i + 1) for i in range(len(quote_clips))]

    clips = []
    timestamp_log = []
    current_duration = 0.0
    quote_clip_index = 0

    for i, clip_path in enumerate(non_overlapping_clips):
        if quote_clip_index < len(quote_clips) and i in insert_positions:
            quote_clip = VideoFileClip(quote_clips[quote_clip_index])
            quote_clip = quote_clip.audio_fadein(fade_duration)
            quote_clip = quote_clip.audio_fadeout(fade_duration)
            clips.append(quote_clip.set_audio(quote_clip.audio))
            timestamp_log.append((current_duration, current_duration + quote_clip.duration))
            current_duration += quote_clip.duration
            quote_clip_index += 1

        clip = VideoFileClip(clip_path)
        clips.append(clip)
        current_duration += clip.duration

    trailer = concatenate_videoclips(clips)
    trailer.write_videofile(str(trailer_path))

    with open(f"{trailer_dir}/trailer_{trailer_number}_quotes_timestamps.txt", 'w') as f:
        for start, end in timestamp_log:
            f.write(f"{start:.2f} - {end:.2f}\n")


def separate_vocals(quote_clip, temp_dir, model='htdemucs_ft'):
    """
    Extract the audio from a video clip, separate the vocals using Demucs, and return the path to the separated vocals.
    
    :param quote_clip: The video clip from which vocals should be separated.
    :param temp_dir: Temporary directory to store intermediate files.
    :param model: The Demucs model to use (e.g., 'htdemucs', 'htdemucs_ft').
    :return: Path to the separated vocals audio file.
    """
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # Extract audio from the video clip and save it as an MP3 file
    audio_file_path = os.path.join(temp_dir, 'audio.mp3')
    quote_clip.write_audiofile(audio_file_path, codec='mp3')

    # Separate vocals from the audio file using Demucs
    output_directory = temp_dir
    command = f"python -m demucs.separate -n {model} --mp3 --mp3-bitrate 320 {audio_file_path} -o {output_directory} --two-stems=vocals"
    try:
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while separating audio: {e}")
        return None
    else:
        print(f"Separation completed. Check the directory {output_directory} for results.")

    # Return the path to the separated vocals file
    separated_vocals_path = os.path.join(output_directory, 'htdemucs_ft', 'audio', 'vocals.mp3')
    return separated_vocals_path


def replace_audio_with_vocals(quote_clip_path, temp_dir, output_dir, model='htdemucs_ft'):
    filename = os.path.basename(quote_clip_path)
    output_path = os.path.join(output_dir, filename)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    quote_clip = VideoFileClip(str(quote_clip_path))
    separated_vocals_path = separate_vocals(quote_clip.audio, temp_dir, model)

    if separated_vocals_path:
        vocals_audio = AudioFileClip(separated_vocals_path)
        quote_clip = quote_clip.set_audio(vocals_audio)
        # Ensure the output path is a string and directory exists
        quote_clip.write_videofile(str(output_path), codec='libx264', audio_codec='mp3')
        vocals_audio.close()
        os.remove(separated_vocals_path)

    return output_path


def join_clip_main(movie_info: MovieInfo):
    logger.info("##### Starting trailer creation #####")
    
    trailer_dir = get_paths()["trailer_dir"]
    if not trailer_dir.exists():
        trailer_dir.mkdir(parents=True, exist_ok=True)

    scenes_dir = get_scenes_dir()
    quote_clips_dir = get_paths()["quote_clips_dir"]
    quote_clips_dir = Path(quote_clips_dir)
    quote_clips = sorted(glob(f"{quote_clips_dir}/quote_clip_*.mp4"), key=lambda x: float(Path(x).stem.split("_")[2]))
    
    temp_dir = "temporary_audio_separation"  # Temporary directory for audio processing
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    
    only_vocals_quote_clip_dir = get_paths()["only_vocals_quote_clips_dir"]
    # Process each quote clip to replace its audio with the separated vocals
    processed_quote_clips = []
    for quote_clip_path in quote_clips:
        processed_clip = replace_audio_with_vocals(quote_clip_path, temp_dir, only_vocals_quote_clip_dir)
        processed_quote_clips.append(processed_clip)
        
    non_empty_scenes = []
    for scene in scenes_dir:
        scene_clips = glob(f"{scene}/clips/*.mp4")
        if scene_clips:
            non_empty_scenes.append(scene_clips)

    if non_empty_scenes:
        clip_combinations = list(itertools.product(*non_empty_scenes))
        for idx, combination in enumerate(clip_combinations):
            join_clips(list(combination), trailer_dir, processed_quote_clips, idx + 1)

    else:
        logger.warning("No clips found in any scene directories. Trailer creation aborted.")