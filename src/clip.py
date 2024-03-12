import logging
from glob import glob
from pathlib import Path
import random
import shutil
from moviepy.editor import VideoFileClip
from scenedetect import SceneManager, ContentDetector, open_video
from scenedetect.frame_timecode import FrameTimecode
from common import get_scenes_dir, logger
import subprocess
import tempfile
from MovieInfo import MovieInfo 


def find_scenes(video_path, is_black_and_white, start_time=None, end_time=None):
    """
    Detects scenes in a video file, focusing on a segment defined by start and end times.
    
    Args:
        video_path (str): Path to the video file.
        is_black_and_white (bool): Indicates if the video is in black and white.
        start_time (float): Start time in seconds from which to begin scene detection.
        end_time (float): End time in seconds at which to stop scene detection.
    
    Returns:
        list: A list of start times (in seconds) for each detected scene.
    """
    video = open_video(video_path)
    scene_manager = SceneManager()

    if is_black_and_white:
        scene_manager.add_detector(ContentDetector(luma_only=True, threshold=18))
    else:
        scene_manager.add_detector(ContentDetector(threshold=23))

    # If start time is provided, seek to that timecode.
    if start_time is not None:
        start_time_tc = FrameTimecode(start_time, video.frame_rate)
        video.seek(start_time_tc)

    # Perform scene detection from the current position to the end or to a specified end time.
    if end_time is not None:
        end_time_tc = FrameTimecode(end_time, video.frame_rate)
        scene_manager.detect_scenes(video, end_time=end_time_tc)
    else:
        scene_manager.detect_scenes(video)

    # Retrieve the list of detected scenes as start/end FrameTimecode pairs
    scenes = scene_manager.get_scene_list()
    
    # Convert the scenes list to a list of start times in seconds
    scene_starts = [scene[0].get_seconds() for scene in scenes]

    return scene_starts

def generate_clips(scenes_dir, movie_info: MovieInfo, how_many_seconds_buffer=1, max_seconds_before_chosen_frame=5, max_clip_len=7, min_clip_len=3):
    """
    Generate video clips from scenes, adjusting the start time based on scene detection.
    """
    movie_path = movie_info.get_movie_file_path()

    duration = movie_info.get_movie_file_duration()
    
    fps = movie_info.get_movie_file_fps()

    for idx, scene_dir in enumerate(scenes_dir):
        logger.info(f"Generating clips for scene {idx+1}")
        clip_dir = Path(scene_dir) / "clips"
        clip_dir.mkdir(parents=True, exist_ok=True)

        frame_paths = sorted(glob(f"{scene_dir}/frames/*.jpg"), key=lambda x: int(Path(x).stem.split('_')[-1]))
        for frame_path in frame_paths:
            frame_number = int(Path(frame_path).stem.split('_')[-1])
            chosen_frame_time = frame_number / fps
            buffer_start_time = max(0, chosen_frame_time - how_many_seconds_buffer)
            buffer_end_time = chosen_frame_time + max_clip_len + how_many_seconds_buffer
            
            # Find scenes within the buffer around the chosen frame
            scene_starts = find_scenes(movie_path, movie_info.get_is_black_and_white(), buffer_start_time, buffer_end_time)
            
            # Determine the adjusted start time for the clip
            start_scene = next((start for start in scene_starts if start <= chosen_frame_time), chosen_frame_time)
            clip_start_time = max(start_scene, chosen_frame_time - max_seconds_before_chosen_frame)

            provisional_clip_end = chosen_frame_time + random.uniform(min_clip_len, max_clip_len)
        
            # Ensure the clip starts at or before the chosen frame time but not earlier than allowed by max_seconds_before_chosen_frame
            final_clip_start_time = max(clip_start_time, chosen_frame_time - max_seconds_before_chosen_frame)
            final_clip_start_time = min(final_clip_start_time, chosen_frame_time)  # Clip cannot start after the chosen frame
            
            
            provisional_clip_end = min(provisional_clip_end, duration) # Ensure the provisional clip end does not exceed the movie's duration
            
            adjusted_clip_end = min(provisional_clip_end, final_clip_start_time + max_clip_len) # Ensure the clip does not exceed the maximum length
            
            
            movie = VideoFileClip(movie_path)
            
            clip = movie.subclip(final_clip_start_time, adjusted_clip_end)
            clip_name = f"clip_{final_clip_start_time}_{adjusted_clip_end}_{frame_number}.mp4"
            clip_path = clip_dir / clip_name
            logger.info(f"Creating clip: {clip_path}")
            clip.write_videofile(str(clip_path), verbose=False, logger=None)
            clip.close()

def post_process_clips(clips_dir, is_black_and_white=False):
    """
    Post-process clips in a directory.
    """
    clip_paths = glob(f"{clips_dir}/*.mp4")
    for clip_path in clip_paths:
        remove_short_scenes_from_clip(Path(clip_path), 2, is_black_and_white)

def remove_short_scenes_from_clip(clip_path, min_scene_length=1.5, is_black_and_white=False):
    video_path = str(clip_path)
    
    # Re-encode the video if needed
    reencode_video(video_path, frame_rate=23.98, codec='libx264')
    
    video = open_video(video_path)
    scene_manager = SceneManager()
    
    # Configure the detector
    detector_options = {'min_scene_len': min_scene_length, 'threshold': 12}
    if is_black_and_white:
        detector_options['luma_only'] = True
    scene_manager.add_detector(ContentDetector(**detector_options))

    # Perform scene detection on the opened video
    scene_manager.detect_scenes(frame_source=video)
    
    # Retrieve the list of detected scenes
    scenes = scene_manager.get_scene_list()

    if not scenes or len(scenes) <= 1:
        logger.info(f"No actionable scenes detected or clip too short to modify: {video_path}")
        return  # No modification needed

    # Calculate the duration of the first and last scenes
    first_scene_duration = scenes[0][1].get_seconds() - scenes[0][0].get_seconds()
    last_scene_duration = scenes[-1][1].get_seconds() - scenes[-1][0].get_seconds()

    # Adjust start and end times based on the duration of orphan scenes
    start_time = scenes[0][0].get_seconds() if first_scene_duration >= min_scene_length else scenes[1][0].get_seconds()
    end_time = scenes[-1][1].get_seconds() if last_scene_duration >= min_scene_length else scenes[-2][1].get_seconds()

    trimming_time = 0.1
    
    # Trim 0.1 second from the start and end, ensuring end_time > start_time
    final_start_time = min(start_time + trimming_time, end_time - trimming_time)  # Ensure start_time does not exceed end_time
    final_end_time = max(end_time - trimming_time, final_start_time + trimming_time)  # Ensure there's at least 1 second of video

    # Create a new clip without the short orphan scenes and trimmed start/end
    with VideoFileClip(video_path) as video_clip:
        new_clip = video_clip.subclip(final_start_time, final_end_time)
        new_clip_path = video_path.replace(".mp4", "_adjusted.mp4")
        new_clip.write_videofile(new_clip_path, codec='libx264', audio_codec='libmp3lame')
        new_clip.close()

    # Replace original clip with adjusted clip
    shutil.move(new_clip_path, video_path)
    logger.info(f"Adjusted clip saved: {video_path}")
    
def reencode_video(input_file, frame_rate=23.98, codec='libx264'):
    """
    Re-encode a video file to a specified frame rate and codec using FFmpeg.
    Args:
        input_file (str): Path to the input video file.
        frame_rate (float): Desired frame rate (frames per second). Defaults to 23.98.
        codec (str): Video codec to use for re-encoding. Defaults to 'libx264'.
    """
    
    temp_file_path = tempfile.mktemp(suffix='.mp4')

    command = [
        'ffmpeg', '-i', input_file, '-r', str(frame_rate), '-c:v', codec,
        '-preset', 'faster', '-crf', '18', '-c:a', 'copy', temp_file_path
    ]
    #subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("STDOUT:", result.stdout.decode('utf-8'))
    print("STDERR:", result.stderr.decode('utf-8'))
    
    logging.info(f"Video re-encoded: {input_file}")

    shutil.move(temp_file_path, input_file)

def clip_main(movie_info: MovieInfo, how_many_seconds_buffer, max_seconds_before_chosen_frame, max_clip_len, min_clip_len) -> None:
    """
    Main function to orchestrate clip generation and post-processing.
    """
    logger.info("Starting step 4 clip creation process.")
    
    scenes_dir = get_scenes_dir()
    
    generate_clips(scenes_dir, movie_info, how_many_seconds_buffer, max_seconds_before_chosen_frame, max_clip_len, min_clip_len)
    for scene_dir in scenes_dir:
        clip_dir = Path(scene_dir) / "clips"
        post_process_clips(clip_dir, movie_info.get_is_black_and_white())
        
        
        
        
        
        
        
        
        
