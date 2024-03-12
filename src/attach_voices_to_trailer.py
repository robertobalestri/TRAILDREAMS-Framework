import logging
from pathlib import Path
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
from common import logger, get_paths, get_project_dir
from moviepy.editor import AudioFileClip
from pydub import AudioSegment
import os
import numpy as np
from MovieInfo import MovieInfo

def read_quote_timestamps(trailer_path):
    """Read quote timestamps from a file and return as list of tuples."""
    timestamps_path = trailer_path.with_name(f"{trailer_path.stem}_quotes_timestamps.txt")
    if not timestamps_path.exists():
        logger.warning(f"Timestamps file not found: {timestamps_path}")
        return []

    timestamps = []
    with open(timestamps_path, 'r') as f:
        for line in f:
            start, end = map(float, line.strip().split(' - '))
            timestamps.append((start, end))
    return timestamps

def distribute_voice_clips(voice_clips, available_slots, trailer_duration):
    if not voice_clips:
        return []

    # Sort clips by duration to try placing larger clips first, which might be more restrictive
    voice_clips.sort(key=lambda clip: clip.duration, reverse=True)

    distributed_clips = []
    used_intervals = []

    for clip in voice_clips:
        for start, end in available_slots:
            # Find a slot where the clip fits
            if end - start >= clip.duration and (start, end) not in used_intervals:
                # Calculate start time within the slot, ensuring no overlap with previously placed clips
                clip_start_time = max(start, (start + end - clip.duration) / 2)  # Center clip if possible
                distributed_clips.append((clip, clip_start_time))
                used_intervals.append((start, end))
                break
    else:
        # If loop completes without breaking, not all clips fit.
        logger.warning("Not all voice clips could be distributed among the available slots.")

    if len(distributed_clips) != len(voice_clips):
        logger.error("Failed to fit all voice clips. Adjusting strategy or reducing clip count may be necessary.")

    return distributed_clips

def calculate_available_slots(quotes_intervals, trailer_duration):
    """
    Calculates slots available for placing voice clips, avoiding quote intervals.
    """
    # Start with the full trailer duration as one large available slot
    available_slots = [(0, trailer_duration)]
    
    for start, end in sorted(quotes_intervals):
        for i, (slot_start, slot_end) in enumerate(available_slots):
            # If the quote interval overlaps with the current slot, split or adjust the slot
            if start <= slot_end and end >= slot_start:
                # Remove the current slot and add new slots based on the overlap
                del available_slots[i]
                if slot_start < start:
                    available_slots.insert(i, (slot_start, start))
                if end < slot_end:
                    available_slots.insert(i + 1, (end, slot_end))
                break  # Move on to the next quote interval
    
    return available_slots














import random
def distribute_voice_clips(voice_clips, available_slots, trailer_duration):
    if not voice_clips:
        return []

    random.shuffle(voice_clips)  # Randomize the order of voice clips to distribute
    distributed_clips = []

    for clip in voice_clips:
        # Find suitable slots where the current clip can fit
        suitable_slots = [(start, end) for start, end in available_slots if end - start >= clip.duration]

        if suitable_slots:
            # Randomly choose a slot where the clip will be placed
            chosen_slot = random.choice(suitable_slots)
            # Randomly determine the start time within the chosen slot for the clip
            max_start = chosen_slot[1] - clip.duration
            clip_start_time = random.uniform(chosen_slot[0], max_start)
            distributed_clips.append((clip, clip_start_time))

            # Update available slots by removing the time occupied by the current clip
            available_slots = [(start, end) if end <= clip_start_time or start >= clip_start_time + clip.duration
                               else (start, clip_start_time) if start < clip_start_time
                               else (clip_start_time + clip.duration, end) for start, end in available_slots]

            # Remove any slots that have become too small to fit any remaining clips
            available_slots = [slot for slot in available_slots if slot[1] - slot[0] >= min(voice_clips, key=lambda c: c.duration).duration]

    # If there are voice clips left that have not been placed, log a warning
    if len(distributed_clips) != len(voice_clips):
        logger.warning("Not all voice clips could be distributed among the available slots.")

    return distributed_clips

def calculate_available_slots(quotes_intervals, trailer_duration, buffer=1):
    """
    Calculates slots available for placing voice clips, avoiding quote intervals and the first and last second.
    
    :param quotes_intervals: Intervals where quotes are present in the trailer.
    :param trailer_duration: Total duration of the trailer.
    :param buffer: Duration in seconds to exclude from the start and end of the trailer.
    :return: A list of available slots for placing voice clips.
    """
    # Start with the full trailer duration minus the buffer as one large available slot
    available_slots = [(buffer, trailer_duration - buffer)]
    
    for start, end in sorted(quotes_intervals):
        new_slots = []
        for slot_start, slot_end in available_slots:
            if end <= slot_start or start >= slot_end:
                # No overlap, keep the slot
                new_slots.append((slot_start, slot_end))
            else:
                # Adjust slots to exclude the quote interval
                if start > slot_start:
                    new_slots.append((slot_start, start))
                if end < slot_end:
                    new_slots.append((end, slot_end))
        available_slots = new_slots  # Replace with the updated list of slots

    return available_slots






def calculate_average_volume_of_clips(voice_clips_paths):
    """Calculate the average volume (in dBFS) of a list of voice clips."""
    volumes = []
    for clip_path in voice_clips_paths:
        audio = AudioSegment.from_file(clip_path)
        volumes.append(audio.dBFS)
    return np.mean(volumes) if volumes else None

def adjust_audio_segment_volume(audio_segment, target_volume_dbfs):
    """Adjust the volume of an audio segment to a target dBFS volume."""
    
    DELTA_DBFS = 4  # Adjust by a small delta to avoid clipping
    
    change_in_volume = target_volume_dbfs - audio_segment.dBFS
    return audio_segment.apply_gain(change_in_volume - DELTA_DBFS)

def attach_voices_to_trailer(trailer_dir: Path, audios_dir: Path, clips_volume: float, voice_volume: float):
    
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    trailers = sorted(trailer_dir.glob("*.mp4"))
    voice_clips_paths = sorted(audios_dir.glob("*.wav"))

    average_voice_clip_volume = calculate_average_volume_of_clips(voice_clips_paths)
    
    # Define your fade duration in milliseconds
    FADE_DURATION = 300  # milliseconds fade duration
    
    for trailer_path in trailers:
        logger.info(f"Processing {trailer_path}")
        
        quotes_intervals = read_quote_timestamps(trailer_path)
        
        # Load the trailer's audio as a pydub AudioSegment for manipulation
        trailer_audio = AudioSegment.from_file(str(trailer_path))
        modified_audio_segments = []

        for start, end in quotes_intervals:
            start_ms, end_ms = start * 1000, end * 1000
            # Extract and adjust volume for segments inside quote intervals
            inside_segment = trailer_audio[start_ms:end_ms]
            
            # Apply fade in and fade out
            inside_segment = inside_segment.fade_in(FADE_DURATION).fade_out(FADE_DURATION)
            
            inside_adjusted = adjust_audio_segment_volume(inside_segment, average_voice_clip_volume)
            modified_audio_segments.append((inside_adjusted, start_ms, end_ms))
        
        # Rebuild the trailer's audio, adjusting volumes outside quote intervals
        current_pos = 0
        rebuilt_audio = AudioSegment.silent(duration=0)
        for segment, start_ms, end_ms in modified_audio_segments:
            outside_segment = trailer_audio[current_pos:start_ms]
            outside_adjusted = outside_segment.apply_gain(clips_volume)
            rebuilt_audio += outside_adjusted + segment
            current_pos = end_ms
        
        # Append any remaining audio after the last quote interval
        if current_pos < len(trailer_audio):
            remaining_segment = trailer_audio[current_pos:]
            remaining_adjusted = remaining_segment.apply_gain(clips_volume)
            rebuilt_audio += remaining_adjusted
        
        # Export the rebuilt audio to a temporary file
        temp_audio_path = os.path.join(temp_dir, "temp_audio.wav")
        rebuilt_audio.export(temp_audio_path, format="wav")
        
        # Use the adjusted audio with the original video
        middle_trailer = VideoFileClip(str(trailer_path)).set_audio(AudioFileClip(temp_audio_path))
        
        
        # Load all voice clips and adjust their volume
        loaded_voice_clips = [AudioFileClip(str(voice_clip)).volumex(voice_volume) for voice_clip in voice_clips_paths]

        # Read the quotes timestamps to avoid
        quotes_intervals = read_quote_timestamps(trailer_path)

        # Calculate available slots for voice clips
        available_slots = calculate_available_slots(quotes_intervals, middle_trailer.duration)

        # Distribute voice clips within these slots
        voice_clip_starts = distribute_voice_clips(loaded_voice_clips, available_slots, middle_trailer.duration)

        if not voice_clip_starts:
            logger.error("Could not distribute voice clips appropriately.")
            continue

        # Create the composite audio clip with the distributed voice clips
        audio_clips = [middle_trailer.audio]  # Start with the original audio
        for clip, start_time in voice_clip_starts:
            clip = clip.set_start(start_time)
            audio_clips.append(clip)

        composite_audio = CompositeAudioClip(audio_clips)
        final_trailer = middle_trailer.set_audio(composite_audio)

        # Save the final trailer with voices attached
        final_trailer_path = trailer_path.with_name(f"{trailer_path.stem}_with_voices.mp4")
        final_trailer.write_videofile(str(final_trailer_path), logger=None)
        
        logger.info(f"Saved {final_trailer_path}")

def attach_voices_to_trailer_main(movie_info: MovieInfo, clips_volume: float, voice_volume: float):
    logger.info("##### Attaching voices to trailer #####")
    audios_dir = get_paths()["audios_dir"]
    attach_voices_to_trailer(get_paths()["trailer_dir"], audios_dir, clips_volume, voice_volume)