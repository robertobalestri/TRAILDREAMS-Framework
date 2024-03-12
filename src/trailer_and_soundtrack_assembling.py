import subprocess
import logging
from glob import glob
from pathlib import Path
from moviepy.editor import VideoFileClip, AudioFileClip

from common import get_paths, logger
from pathlib import Path
from MovieInfo import MovieInfo

def get_audio_duration(audio_path):
    """
    Get the duration of an audio file.
    """
    audio_clip = AudioFileClip(audio_path)
    return audio_clip.duration

def run_ffmpeg_command(command):
    """
    Run an ffmpeg command with subprocess and log any errors.

    Args:
        command (list): The ffmpeg command to run as a list of arguments.
    """
    print("Running command:", " ".join(command))
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logger.info("ffmpeg command executed successfully.")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg command failed with return code {e.returncode}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        raise

def apply_dynamic_ducking(trailer_video_path, soundtrack_audio_path, output_video_path, music_volume, original_audio_volume):
    """
    Function to dynamically duck the soundtrack based on the trailer audio and merge it with the original video.
    """
    # Paths for intermediate files
    adjusted_trailer_audio_path = 'temp_adjusted_trailer_audio.wav'
    adjusted_soundtrack_path = 'temp_adjusted_soundtrack.wav'
    ducked_audio_path = 'temp_ducked_audio.wav'
    
    fade_in_duration = 2
    fade_out_duration = 6
    
    # Get the duration of the soundtrack for fade-out calculation
    soundtrack_duration = get_audio_duration(soundtrack_audio_path)
    trailer_duration = get_audio_duration(trailer_video_path)
    
    minimum_duration = min(soundtrack_duration, trailer_duration)
    
    fade_out_start = max(minimum_duration - fade_out_duration, 0)  # Ensuring not negative
    
    

    # Adjust the volumes of the trailer audio and the soundtrack
    run_ffmpeg_command([
        'ffmpeg', '-i', trailer_video_path,
        '-filter:a', f'volume={original_audio_volume}dB',
        '-ac', '2',  # Ensure stereo output for compatibility
        '-y', adjusted_trailer_audio_path
    ])
    
    # Apply fade in and fade out to the soundtrack, in addition to adjusting its volume
    run_ffmpeg_command([
        'ffmpeg', '-i', soundtrack_audio_path,
        '-filter:a', f'volume={music_volume}dB,afade=t=in:st=0:d={fade_in_duration},afade=t=out:st={fade_out_start}:d={fade_out_duration}',
        '-y', adjusted_soundtrack_path
    ])

    # Apply dynamic ducking to mix the adjusted trailer audio and soundtrack
    # Using the improved filter_complex chain for dynamic ducking
    run_ffmpeg_command([
        'ffmpeg',
        '-i', adjusted_trailer_audio_path,  # Input 0: Adjusted trailer audio
        '-i', adjusted_soundtrack_path,     # Input 1: Adjusted soundtrack
        '-filter_complex',
        "[0:a]asplit=2[sc][mix];"\
        "[1:a][sc]sidechaincompress=threshold=0.015:ratio=1.5:level_sc=0.8:release=600:attack=25[compr];"\
        "[compr][mix]amix=inputs=2:duration=first:dropout_transition=3[a]",
        '-map', '[a]',                      # Map the processed audio stream
        '-y', ducked_audio_path
    ])

    # Merge the processed (ducked) audio with the original video and add fade in and out effects
    fade_in_duration_video = 1  # Duration of the fade in from black effect in seconds
    fade_out_duration_video = 5  # Duration of the fade to black effect in seconds
    video_duration = get_audio_duration(trailer_video_path)  # Assuming video and its audio have the same duration

    run_ffmpeg_command([
        'ffmpeg',
        '-i', trailer_video_path,          # Original video
        '-i', ducked_audio_path,           # Processed (ducked) audio
        '-filter_complex', 
        f'[0:v]fade=t=in:st=0:d={fade_in_duration_video},fade=t=out:st={video_duration-fade_out_duration_video}:d={fade_out_duration_video}[v]',  # Apply fade in at start and fade to black at the end
        '-map', '[v]',                     # Map the processed video stream
        '-map', '1:a:0',                   # Map audio from the processed audio
        '-c:v', 'libx264',                 # Specify video codec
        '-preset', 'medium',               # Encoding preset
        '-crf', '23',                      # Constant Rate Factor for quality
        '-c:a', 'libmp3lame',              # Specify audio codec
        '-shortest',                       # Finish encoding when the shortest input stream ends
        '-y', output_video_path
    ])

    # Cleanup temporary files
    Path(adjusted_trailer_audio_path).unlink(missing_ok=True)
    Path(adjusted_soundtrack_path).unlink(missing_ok=True)
    Path(ducked_audio_path).unlink(missing_ok=True)

    logger.info(f"Ducked video saved as {output_video_path}")


def composite_trailer_soundtrack(trailers_dir: str, music_volume: float, original_audio_volume: float, project_name: str):
    trailers_dir_path = Path(trailers_dir)
    final_dir_path = trailers_dir_path / "final"
    final_dir_path.mkdir(parents=True, exist_ok=True)

    # Use glob pattern to match all trailers with voices
    for trailer_path in trailers_dir_path.glob("trailer_*_with_voices.mp4"):
        trailer_name = trailer_path.stem
        trailer_number = trailer_name.split('_')[-3]  # Adjusted index based on naming convention
        soundtrack_name = f"soundtrack_{trailer_number}.wav"
        soundtrack_path = trailers_dir_path / soundtrack_name

        if soundtrack_path.exists():
            logger.info(f"Processing {trailer_name}")
            output_video_path = final_dir_path / f"{project_name}_{trailer_name}_with_soundtrack.mp4"
            
            apply_dynamic_ducking(str(trailer_path), str(soundtrack_path), str(output_video_path), music_volume, original_audio_volume)
            logger.info(f"Final video saved to {output_video_path}")
        else:
            logger.warning(f"Soundtrack not found for {trailer_name}: Expected path {soundtrack_path}")

def trailer_and_soundtrack_assembling_main(movie_info: MovieInfo, music_volume_in_dB: float, trailer_no_soundtrack_volume_in_dB: float, project_name: str):
    logger.info("##### Starting step 10 trailer and soundtrack compositing #####")
    trailer_dir = get_paths()["trailer_dir"]
    composite_trailer_soundtrack(trailer_dir, music_volume_in_dB, original_audio_volume=trailer_no_soundtrack_volume_in_dB, project_name=project_name)