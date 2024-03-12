

from concurrent.futures import ProcessPoolExecutor
import logging
from pathlib import Path
import subprocess
from moviepy.editor import VideoFileClip
from common import get_paths, logger, FRAME_EXTRACTION_TIME_INTERVAL
from moviepy.editor import VideoFileClip
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from MovieInfo import MovieInfo


def extract_frame(movie_path, timestamp, frame_number, output_path):
    """
    Extracts a single frame from a video file at the specified timestamp.

    Args:
        movie_path (str): The path to the video file.
        timestamp (float): The timestamp in seconds where the frame should be extracted.
        frame_number (int): The frame number.
        output_path (Path): The path to save the extracted frame.

    Returns:
        str: A message indicating the result of the extraction.
    """
    ffmpeg_command = f'ffmpeg -ss {timestamp} -i "{movie_path}" -frames:v 1 "{output_path}" -y'
    try:
        subprocess.run(ffmpeg_command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"Extracted frame at {timestamp} seconds (frame {frame_number}) to {output_path}"
    except subprocess.CalledProcessError as e:
        return f"Failed to extract frame: {e}"


from moviepy.editor import VideoFileClip
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

def create_screenshots(movie_path: str, n_frames: int, frame_dir: Path) -> None:
    """
    Create screenshots from a video file.

    Args:
        movie_path (str): The path to the video file.
        n_frames (int): The number of screenshots to create.
        frame_dir (Path): The directory to save the screenshots.

    Returns:
        None
    """
    if not frame_dir.exists():
        frame_dir.mkdir(parents=True, exist_ok=True)

    video = VideoFileClip(movie_path)
    duration = video.duration
    fps = video.fps
    # Adjusting start and end times to exclude the first 1% and last 10% of the video
    start_time = duration * 0.01
    end_time = duration * 0.90
    adjusted_duration = end_time - start_time
    video.reader.close()
    video.audio.reader.close_proc()

    step = adjusted_duration / n_frames
    tasks = []

    with ProcessPoolExecutor(max_workers=4) as executor:
        for i in range(n_frames):
            timestamp = start_time + i * step
            frame_number = int(timestamp * fps)
            output_path = frame_dir / f"frame_{frame_number}.jpg"
            tasks.append(executor.submit(extract_frame, movie_path, timestamp, frame_number, output_path))

        for future in tasks:
            logger.info(future.result())

def frame_main(movie_info: MovieInfo) -> None:
    """
    Main function to create screenshots from a video file using FFmpeg in parallel.

    Returns:
        None
    """
    logger.info("##### Starting step 1 frame sampling with FFmpeg in parallel #####")
    
    number_of_frames = int(movie_info.file_duration / FRAME_EXTRACTION_TIME_INTERVAL)
    
    create_screenshots(movie_info.get_movie_file_path(), number_of_frames, get_paths()["frames_dir"])