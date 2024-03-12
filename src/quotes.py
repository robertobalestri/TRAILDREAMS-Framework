import requests
from bs4 import BeautifulSoup
from common import get_paths, PROMPTS_DIR, get_project_dir, filter_violent_words, OPENAI_API_VERSION, logger, MAX_DURATION_QUOTE_CLIP, get_video_file_details
from imdb import Cinemagoer
import logging
from pathlib import Path
import re
from openai import AzureOpenAI
from dotenv import load_dotenv, find_dotenv, get_key
import time
import stable_whisper
import subprocess
from difflib import SequenceMatcher
from datetime import timedelta, datetime
import os
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import json
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import string
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.core import Segment
from pydub import AudioSegment
import spacy
from textblob import TextBlob
from MovieInfo import MovieInfo

def extract_audio(video_path, output_audio_path):
    """
    Extracts the audio from a video file and saves it as a WAV file.

    Args:
        video_path (str): The path to the input video file.
        output_audio_path (str): The path to save the output audio file.

    Returns:
        None
    """
    command = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-vn',
        '-acodec', 'pcm_s16le',
        output_audio_path.with_suffix('.wav')
    ]
    subprocess.run(command)

def extract_clip(video_path, start_time, end_time, output_path):
    """
    Extracts a clip from the given video file between the start and end times.
    
    Args:
        video_path (str): Path to the source video file.
        start_time (str): Start time of the clip in 'HH:MM:SS' format.
        end_time (str): End time of the clip in 'HH:MM:SS' format.
        output_path (Path): Path to save the extracted clip.
    """
    command = [
        'ffmpeg',
        '-y',
        '-i', str(video_path),
        '-ss', start_time,
        '-to', end_time,
        '-c', 'copy',
        str(output_path)
    ]
    subprocess.run(command, check=True)

def split_text_by_hashtags(text):
    """
    Splits the given text by hashtags and returns a list of cleaned phrases.

    Args:
        text (str): The text to be split.

    Returns:
        list: A list of cleaned phrases without hashtags and leading/trailing whitespaces.
    """
    phrases = text.split('#')
    cleaned_phrases = []
    for phrase in phrases:
        phrase = phrase.replace('"', "").strip()
        cleaned_phrases.append(phrase)
    return cleaned_phrases

def is_sentence_complete(nlp, sentence):
    """
    Check if a sentence has a clear subject and predicate, recognizing copular verbs as predicates.
    
    Parameters:
    sentence (str): The sentence to analyze.

    Returns:
    bool: True if the sentence is complete, False otherwise.
    """
    doc = nlp(sentence)
    has_subject = False
    has_predicate = False

    for token in doc:
        # Check for a nominal subject.
        if token.dep_ in ['nsubj', 'nsubjpass']:
            has_subject = True
        # Check for a verb or auxiliary verb as a predicate.
        if token.pos_ in ['VERB', 'AUX']:
            has_predicate = True

    return has_subject and has_predicate

def analyze_sentiment_intensity(sentence):
    if sentence.endswith('!'):
        return 1
    analysis = TextBlob(sentence)
    polarity = analysis.sentiment.polarity
    # The intensity is considered as the absolute value of the polarity
    intensity = abs(polarity)
    
    return intensity


def process_quotes(raw_quotes, is_documentary=False):
    nlp = spacy.load("en_core_web_sm")  # Load the English tokenizer, tagger, parser, NER, and word vectors
    
    processed_quotes = []
    
    
    for raw_quote in raw_quotes:
        raw_quote = (re.sub(r'\[.*?\]', '', raw_quote)).strip()
        
        # Split the quote by speakers
        speaker_quotes = re.split(r'(?m)^\w+:\s*', raw_quote)
        for speaker_quote in speaker_quotes:
            if not speaker_quote:
                continue
            full_speaker_quote = speaker_quote.replace(":", "", 1).replace("*", "").strip()
            

            quote_for_length_check = re.sub(r'[^\w\s]', '', full_speaker_quote).replace(" ", "")

            sentences = re.split(r'(?<=[.!?])\s*', full_speaker_quote)
            
            for sentence in sentences:
                cleaned_sentence = sentence.strip()
                sentence_for_length_check = re.sub(r'[^\w\s]', '', cleaned_sentence).replace(" ", "")
                
                print("Cleaned sentence: " + cleaned_sentence, "Sentiment intensity: " + str(analyze_sentiment_intensity(cleaned_sentence)))
                
                if (sentence_for_length_check and len(sentence_for_length_check) >= 12 and len(sentence_for_length_check) <= 80
                    and is_sentence_complete(nlp, cleaned_sentence)
                    and (analyze_sentiment_intensity(cleaned_sentence) > 0.20 or is_documentary)
                    ):

                    cleaned_sentence = re.sub(r'([,!.?])', r' \1', cleaned_sentence)
                    
                    if len(quote_for_length_check) <= 80:
                        processed_quotes.append(full_speaker_quote)
                        break
                    else:
                        processed_quotes.append(cleaned_sentence)

    sorted_quotes = sorted(processed_quotes, key=lambda q: len(re.sub(r'[^\w\s]', '', q).replace(" ", "")))[:200]

    print("Length sorted_quotes = " + str(len(sorted_quotes)))

    for quote in sorted_quotes:
        print(quote)
    
    return sorted_quotes

def call_gpt_relevant_quotes(quotes, number_of_quote_clips):
    load_dotenv(find_dotenv())
    azure_endpoint = get_key(".env", "AZURE_OPENAI_ENDPOINT")
    api_key = get_key(".env", "AZURE_OPENAI_KEY")
    client = AzureOpenAI(azure_endpoint=azure_endpoint, api_key=api_key, api_version=OPENAI_API_VERSION)  # Adjust API version if necessary.

    filtered_quotes = [filter_violent_words(quote, substitute_with_redacted=False) for quote in quotes]

    
    print("Length = " + str(len(filtered_quotes)))
    
    def generate_quotes(input_quotes):
        
        prompt_path = str(PROMPTS_DIR / 'prompt_relevant_quotes.txt').replace("\\", "/")
        
        with open(prompt_path, 'r') as file:
            prompt = file.read()
            response = client.chat.completions.create(
                model="gpt-35-turbo",
                #model="gpt-4-1106",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "#NUMBER OF PHRASES TO CHOOSE: " + str(number_of_quote_clips) + '#\n'},
                    {"role": "user", "content": "#QUOTES: " + str(input_quotes) + '#\n'}
                ],
                max_tokens=1000
            )
            
            print(str(response.choices[0]))
            return response.choices[0].message.content
        
    try:
        return generate_quotes(filtered_quotes)
    except Exception as e:
        half_index = len(filtered_quotes) // 2
        if half_index > 0:
            # Try first half
            try:
                return generate_quotes(filtered_quotes[:half_index])
            except Exception as e:
                # If first half fails, try second half
                return generate_quotes(filtered_quotes[half_index:])
        else:
            raise Exception("Unable to generate quote with the given input.") from e

def transcribe_audio_to_srt(model, audio_path, output_path, quotes_selected):
    """
    Transcribes the audio file at the given `audio_path` and saves the transcription to an SRT file at the `output_path`.

    Args:
        audio_path (str): The path to the audio file to be transcribed.
        output_path (str): The path to save the SRT file.
        quotes_selected (list): A list of quotes to be included in the transcription.

    Returns:
        None
    """


    # Transcribe audio with adjusted parameters
    result = model.transcribe(
        str(audio_path),
        language='en',
        suppress_silence=False,  # Consider turning off silence suppression
        word_timestamps=True,  # Ensure word-level timestamps are enabled
        regroup=True,  # Keep default regrouping or experiment with this
        suppress_word_ts=False,  # Disable word timestamp suppression
        use_word_position=True,  # Use word position for timestamp adjustments
        vad=True,  # Voice Activity Detection to help identify speech segments
        #initial_prompt="I need to find the following quotes: " + str(quotes_selected),
        only_voice_freq = True,
        ts_num = 5
        # You might want to experiment with other parameters based on your specific needs
    ).clamp_max()

    
    #new_result = model.align(str(audio_path), result, language='en')
    
    # Save the transcription to an SRT file
    result.to_srt_vtt(output_path, segment_level=True, word_level=False)  # segment_level = False, word_level = True)
    
    #result.to_srt_vtt(str(get_project_dir() / "word_level.srt"), segment_level=False, word_level=True)

def preprocess_transcript(raw_transcript):
    """
    Preprocesses a raw transcript by splitting it into timestamped entries.

    Args:
        raw_transcript (str): The raw transcript to be processed.

    Returns:
        list: A list of tuples containing timestamp and corresponding text entries.
    """
    entries = []
    lines = raw_transcript.split('\n')
    i = 0
    while i < len(lines):
        if '-->' in lines[i]:
            timestamp = lines[i].strip()
            text = ""
            i += 1
            while i < len(lines) and lines[i].strip() != "":
                text += lines[i].strip() + " "
                i += 1
            entries.append((timestamp, text.strip()))
        i += 1
    return entries

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def find_similar_phrase(entries, target_phrase):
    target_phrase = target_phrase.lower()
    best_match = ("", "", 0)  # (start_timestamp, end_timestamp, similarity score)
    best_individual_match = ("", "", 0)  # To store the best non-expanded match
    start_index, end_index = None, None

    # Iterate over entries to find the best initial match and potential span
    for i, entry in enumerate(entries):
        text = entry[1].lower()
        similarity = SequenceMatcher(None, target_phrase, text).ratio()

        if similarity > best_individual_match[2]:
            best_individual_match = (entry[0].split(' --> ')[0], entry[0].split(' --> ')[1], similarity)

        if similarity > best_match[2]:
            best_match = (entry[0].split(' --> ')[0], entry[0].split(' --> ')[1], similarity)
            start_index, end_index = i, i

            # Extend the search to adjacent entries if the phrase might span multiple entries
            combined_text = text
            for j in range(i + 1, len(entries)):
                combined_text += " " + entries[j][1].lower()
                combined_similarity = SequenceMatcher(None, target_phrase, combined_text).ratio()
                if combined_similarity > best_match[2]:
                    best_match = (entry[0].split(' --> ')[0], entries[j][0].split(' --> ')[1], combined_similarity)
                    end_index = j

    # Check if the best match span is longer than 8 seconds
    if start_index is not None and end_index is not None:
        start_timestamp, end_timestamp = best_match[0], best_match[1]
        duration = calculate_duration(start_timestamp, end_timestamp)

        print(f"Duration for '{target_phrase}': {duration}")
        
        if duration > MAX_DURATION_QUOTE_CLIP:
            # If the duration exceeds MAX_DURATION_QUOTE_CLIP seconds, revert to the best individual match's timestamps
            start_timestamp, end_timestamp = best_individual_match[0], best_individual_match[1]
            
            print(f"Reverting to individual match for '{target_phrase}': start={start_timestamp}, end={end_timestamp}")

        # Extract the exact or closest matching phrase from the combined text
        combined_text = " ".join(entry[1].lower() for entry in entries[start_index:end_index + 1])
        extracted_phrase = get_closest_match(target_phrase, combined_text)

        print(f"Exact/closest match for '{target_phrase}': start={start_timestamp}, end={end_timestamp}, similarity={best_match[2]}, extracted_phrase={extracted_phrase}")

        return [(start_timestamp, end_timestamp, target_phrase)]
    
    return []  # Return empty list if no match is found

def calculate_duration(start, end):
    """Calculate the duration in seconds between two timestamps."""
    FMT = '%H:%M:%S,%f'
    from datetime import datetime
    tdelta = datetime.strptime(end, FMT) - datetime.strptime(start, FMT)
    return tdelta.total_seconds()


def get_closest_match(target_phrase, text):
    best_match = (0, "")
    for i in range(len(text) - len(target_phrase) + 1):
        substring = text[i:i + len(target_phrase)]
        similarity = SequenceMatcher(None, target_phrase, substring).ratio()
        if similarity > best_match[0]:
            best_match = (similarity, substring)
    return best_match[1]


def format_timedelta(td):
    """Converts a timedelta to a string format suitable for FFmpeg."""
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"
    
def cut_buffered_clips_with_vad(video_path, clips, output_folder, start_buffer, end_buffer, audio_path, small_end_gap=0.1):
    vad_pipeline = VoiceActivityDetection(segmentation="pyannote/segmentation")
    vad_model = vad_pipeline.instantiate({
        "onset": 0.5, "offset": 0.5, "min_duration_on": 0.2, "min_duration_off": 0.1
    })

    audio_for_vad = vad_model({'uri': 'audio', 'audio': audio_path})

    adjusted_clips = []
    for i, (start_time, end_time) in enumerate(clips):
        start_td = datetime.strptime(start_time, "%H:%M:%S,%f") - datetime(1900, 1, 1)
        end_td = datetime.strptime(end_time, "%H:%M:%S,%f") - datetime(1900, 1, 1)

        buffered_start_td = max(timedelta(0), start_td - timedelta(seconds=start_buffer))
        buffered_end_td = end_td + timedelta(seconds=end_buffer)

        for speech_segment in audio_for_vad.get_timeline():
            speech_start_td = timedelta(seconds=speech_segment.start)
            speech_end_td = timedelta(seconds=speech_segment.end)

            if speech_start_td <= end_td and speech_end_td > end_td:
                buffered_end_td = max(buffered_end_td, speech_end_td + timedelta(seconds=end_buffer))

        buffered_end_td += timedelta(seconds=small_end_gap)

        adjusted_clips.append((buffered_start_td, buffered_end_td))

    # Remove overlapping clips
    adjusted_clips = sorted(adjusted_clips, key=lambda x: x[0])  # Sort by start time
    non_overlapping_clips = []
    for current_clip in adjusted_clips:
        if not non_overlapping_clips or current_clip[0] >= non_overlapping_clips[-1][1]:
            non_overlapping_clips.append(current_clip)

    # Generate clips
    for i, (start_td, end_td) in enumerate(non_overlapping_clips):
        buffered_start_str = (datetime(1900, 1, 1) + start_td).time().strftime("%H:%M:%S")
        buffered_end_str = (datetime(1900, 1, 1) + end_td).time().strftime("%H:%M:%S")

        
        # Assuming start_td and end_td are timedelta objects
        start_in_seconds = start_td.total_seconds()
        end_in_seconds = end_td.total_seconds()
        
        output_path = Path(output_folder) / f"quote_clip_{start_in_seconds}_{end_in_seconds}_{i+1}.mp4"
        
        command = [
            'ffmpeg',
            '-y',
            '-i', str(video_path),
            '-ss', buffered_start_str,
            '-to', buffered_end_str,
            '-c:v', 'libx264',
            '-c:a', 'libmp3lame',
            str(output_path)
        ]
        
        subprocess.run(command, check=True)

    print("Buffered clips have been successfully cut with VAD adjustment and overlapping clips handled.")



def overlay_black_on_orphan_scenes(movie_info, video_path, min_scene_length=1.0):
    """
    Overlays black video on the start or end orphan scenes of a given video
    while keeping the original audio, based only on scenes shorter than the min_scene_length.
    """
    
    # Set up PySceneDetect managers
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(min_scene_len=min_scene_length, threshold=24))
    
    # Start video manager and detect scenes
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
   
    fps = movie_info.get_movie_file_fps()
    
    # Retrieve the list of detected scenes (as frame numbers)
    scenes = scene_manager.get_scene_list(fps)
    
    #print("Scenes:", scenes)

    # Stop the video manager
    video_manager.release()

    if not scenes:
        logger.info("No scenes detected.")
        return

    scenes_timestamps = [(scene[0].get_seconds(), scene[1].get_seconds()) for scene in scenes]

    # Initialize list for orphan scenes based on min_scene_length
    orphan_scenes = []
    for i, (start, end) in enumerate(scenes_timestamps):
        duration = end - start
        # For the first and last scene, check if they're orphans based on their duration
        if (i == 0 or i == len(scenes_timestamps) - 1) and duration < min_scene_length:
            orphan_scenes.append((start, end))

    # Overlay black screens on orphan scenes
    for start_time, end_time in orphan_scenes:
        #print(f"Overlaying black screen from {start_time} to {end_time}")
        quote_clip_path, temp_path, black_screen_path = overlay_black_screen(video_path, start_time, end_time, fps)
        
        time.sleep(0.5)
        # Clean up the temporary black screen video file
        os.remove(black_screen_path)
        
        # Copy the adjusted video to the old video path
        shutil.move(temp_path, quote_clip_path)
        
    

def overlay_black_screen(video_path, start_time, end_time, fps):
    """
    Overlays a black screen over a specified time range in the video,
    dynamically adjusting to the video's resolution and processing each clip individually.

    Args:
    video_path (str): Path to the video file.
    start_time (float): Start time of the range to overlay with black screen.
    end_time (float): End time of the range to overlay with black screen.
    fps (float): Frame rate of the video.
    """
    # Get the resolution of the video
    width, height = get_video_resolution(video_path)
    
    # Generate a black screen video clip of the required duration and resolution
    duration = end_time - start_time
    
    #print(f"Duration black screen: {duration}")
    
    black_screen_path = "black_screen_temp.mp4"
    command = [
        'ffmpeg',
        '-y',
        '-f', 'lavfi',
        '-i', f'color=c=black:s={width}x{height}:r={fps}',
        '-t', str(duration),
        '-vf', f'fps={fps}',
        black_screen_path
    ]
    subprocess.run(command, check=True)

    # Dynamically construct the output file path based on the input video path
    output_path = video_path.replace(".mp4", "_adjusted.mp4")

    ffmpeg_command = [
        'ffmpeg',
        '-y',  # Overwrite output files without asking
        '-i', video_path,  # Dynamically use the current video path
        '-i', black_screen_path,  # Input black screen video
        '-filter_complex', 
        f"[0:v][1:v]overlay=enable='between(t,{start_time},{end_time})'[video]",  # Dynamically set overlay times
        '-map', '[video]',  # Map the video output from the filter complex
        '-map', '0:a',  # Map the audio from the input file
        '-c:v', 'libx264',  # Video codec
        '-c:a', 'libmp3lame',  # Audio codec, changed to 'aac' for broader compatibility
        '-shortest',  # Finish encoding when the shortest input stream ends
        output_path  # Dynamically set the output file path
    ]

    try:
        subprocess.run(ffmpeg_command, check=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr.decode()}")
        raise

    logger.info(f"Orphan scenes overlaid with black screen in {video_path} using resolution {width}x{height}")
    
    return video_path, output_path, black_screen_path

def get_video_resolution(video_path):
    """
    Uses ffprobe to get the resolution of the video.

    Args:
    video_path (str): Path to the video file.

    Returns:
    tuple: Resolution of the video as (width, height).
    """
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'json',
        video_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    info = json.loads(result.stdout)
    width = info['streams'][0]['width']
    height = info['streams'][0]['height']
    return width, height

def check_and_rename_clips(movie_info, folder_path):
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"The provided path {folder_path} is not a valid directory.")
        return

    # List all mp4 files in the directory
    clips = list(folder.glob('*.mp4'))
    clips_to_keep = []

    for clip_path in clips:
        # Use your existing function to get the duration of the clip
        duration, _ = get_video_file_details(clip_path)

        if duration > MAX_DURATION_QUOTE_CLIP:
            print(f"Deleting {clip_path} as its duration {duration} seconds exceeds the maximum allowed of {MAX_DURATION_QUOTE_CLIP} seconds.")
            os.remove(clip_path)
        else:
            clips_to_keep.append(clip_path)

    # Rename the remaining clips sequentially
    for i, clip_path in enumerate(sorted(clips_to_keep, key=lambda x: float(x.stem.split("_")[2])), start=1):
        start, end = map(float, clip_path.stem.split("_")[2:4]) # Extract the start and end timestamps, for example extracts from quote_clip_2.1_3.5_1.mp4 extract 2.1 and 3.5
        new_path = clip_path.parent / f"quote_clip_{start}_{end}_{i}{clip_path.suffix}"
        os.rename(clip_path, new_path)
        print(f"Renamed {clip_path} to {new_path}")

def timestamp_to_seconds(timestamp):
    hours, minutes, seconds = timestamp.split(':')
    seconds, milliseconds = seconds.split(',')
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000
    return total_seconds

def quotes_main(movie_info: MovieInfo, target_quote_clip_count: int) -> None:
    """
    Main function to find and process quotes from a movie.

    This function performs the following steps:
    1. Find the shortest quotes from IMDb.
    2. Select relevant quotes using Azure OpenAI (or any other GPT model).
    3. Extract audio from the video file.
    4. Transcribe audio to SRT format.
    5. Preprocess the transcript.
    6. Get timestamps for each selected quote.
    7. Cut buffered clips based on timestamps.

    Returns:
        None
    """
    logger.info("Starting process to find and clip quotes.")
    
    # Load the Whisper model
    model = stable_whisper.load_model('base')
    
    # Step 1: Find the shortest quotes from IMDb
    
    raw_quotes = movie_info.get_quotes()
    
    print("Raw quotes: " + str(raw_quotes))
    
    shortest_quotes = process_quotes(raw_quotes, is_documentary=movie_info.is_documentary())
    
    print("Shortest quotes: " + str(shortest_quotes))
    
    # Step 2: Select relevant quotes via Azure OpenAI (or any other GPT model)
    selected_quotes_path = get_project_dir() / "selected_quotes.txt"
    
    quotes_selected = []
    
    if not selected_quotes_path.exists():
        
        while len(quotes_selected) != target_quote_clip_count and len(shortest_quotes) > target_quote_clip_count:
            quotes_selected = split_text_by_hashtags(call_gpt_relevant_quotes(shortest_quotes, target_quote_clip_count))
        
        with open(selected_quotes_path, "w") as file:
            file.write("\n".join(quotes_selected))
    else:
        with open(selected_quotes_path, "r") as file:
            quotes_selected = file.read().strip().split("\n")

    logger.info(f"Selected quotes: {quotes_selected}")
    
    # Step 3: Extract audio from the video file
    audio_path = get_project_dir() / "movie_audio.wav"
    if not audio_path.exists():
        extract_audio(movie_info.get_movie_file_path(), audio_path)
    
    # Step 4: Transcribe audio to SRT
    subs_path = get_project_dir() / "subs.srt"
    
    if not subs_path.exists():
        transcript = transcribe_audio_to_srt(model, audio_path, str(subs_path), quotes_selected)
    
    with open(subs_path, "r", encoding="utf-8") as file:
        transcript = file.read()
        #print(f"Transcript length: {len(transcript)}")
        
    preprocessed_transcript = preprocess_transcript(transcript)
    
    # Step 6: Get timestamps for each selected quote
    all_timestamps = []  # Initialize an empty list to collect all timestamps
    for quote in quotes_selected:
        logger.info(f"Searching for quote: {quote}")
        quote_timestamps = find_similar_phrase(preprocessed_transcript, quote)
        
        all_timestamps.extend(quote_timestamps)  # Collect timestamps for each quote

    # Order timestamps by the minor number in the first element of each tuple
    all_timestamps = sorted(all_timestamps, key=lambda x: x[0])

    duration = movie_info.get_movie_file_duration()
    
    # Calculate the threshold duration for similarity
    threshold_duration = duration * 0.01

    # Remove quotes very near in time to each other
    filtered_timestamps = []
    for i in range(len(all_timestamps)):
        if i == 0:
            filtered_timestamps.append(all_timestamps[i])
        else:
            prev_end_time = timestamp_to_seconds(filtered_timestamps[-1][1])
            curr_start_time = timestamp_to_seconds(all_timestamps[i][0])
            # Check if the current timestamp is too close to the previous one
            if curr_start_time - prev_end_time <= threshold_duration:
                # Compare sentiment intensity of the current quote and the previous quote
                prev_sentiment = analyze_sentiment_intensity(filtered_timestamps[-1][2])
                
                print(f"Prev sentiment: {prev_sentiment} Phrase: {filtered_timestamps[-1][2]}")
                
                curr_sentiment = analyze_sentiment_intensity(all_timestamps[i][2])
                
                print(f"Curr sentiment: {curr_sentiment} Phrase: {all_timestamps[i][2]}")

                # Keep the quote with higher sentiment intensity
                if curr_sentiment > prev_sentiment:
                    filtered_timestamps[-1] = all_timestamps[i]
                else:
                    logger.info(f"Skipping similar quote with lower sentiment: {all_timestamps[i]}")
            else:
                filtered_timestamps.append(all_timestamps[i])

    # Update all_timestamps with the filtered timestamps
    all_timestamps = filtered_timestamps

    logger.info(f"Collected timestamps for all quotes: {all_timestamps}")
    
    # Step 7: Cut buffered clips based on timestamps
    quote_clips_folder = get_paths()["quote_clips_dir"]
    os.makedirs(quote_clips_folder, exist_ok=True)  # Ensure the output folder exists
    clips = [(ts[0], ts[1]) for ts in all_timestamps]  # Correctly prepare clip timestamps

    cut_buffered_clips_with_vad(movie_info.get_movie_file_path(), clips, quote_clips_folder, start_buffer=0.1, end_buffer=0.8, audio_path=audio_path, small_end_gap=0.1)

    #input("Press Enter to continue...")
    
    # Step 8: Check and delete clips that exceed the maximum duration, then rename the remaining clips sequentially
    check_and_rename_clips(movie_info, quote_clips_folder)
    
    # Step 8: Overlay black screens on orphan scenes for each quote clip
    min_scene_length = 1.0  # Minimum length of a scene to not be considered orphan, in seconds
    for clip_path in quote_clips_folder.iterdir():
        if clip_path.suffix == '.mp4':  # Ensure processing only video files
            logger.info(f"Processing {clip_path.name} for orphan scenes...")
            overlay_black_on_orphan_scenes(movie_info, str(clip_path), min_scene_length)
    
    logger.info("Finished processing quotes.")
    
    