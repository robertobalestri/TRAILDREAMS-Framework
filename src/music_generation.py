from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from openai import AzureOpenAI
from dotenv import load_dotenv, find_dotenv, get_key, set_key
import time
import torch
from moviepy.editor import VideoFileClip
from glob import glob
import os
import logging
from common import OPENAI_API_VERSION, logger, get_video_file_details, get_paths, get_project_dir, PROMPTS_DIR
from MovieInfo import MovieInfo

def get_video_durations(directory):
    # Use glob to find all video files in the directory
    # This pattern assumes video files have common video file extensions, add or remove as necessary
    video_files = glob(os.path.join(directory, "*.mp4")) + glob(os.path.join(directory, "*.avi")) + glob(os.path.join(directory, "*.mov"))

    video_durations = {}
    for video_file in video_files:
        duration, _ = get_video_file_details(video_file)
        video_durations[video_file] = duration
    
    return video_durations

def gpt_musical_prompt_create(plot):
    
    prompt_file_path = str(PROMPTS_DIR / 'prompt_gpt_for_music_creation.txt').replace("\\", "/")

    with open(prompt_file_path, 'r') as file:
        prompt = file.read()
    
    client = AzureOpenAI(
        azure_endpoint=get_key(".env", "AZURE_OPENAI_ENDPOINT"),
        api_key=get_key(".env", "AZURE_OPENAI_KEY"),
        api_version=OPENAI_API_VERSION
    )
    
    attempt = 0
    max_attempts = 3  # Maximum number of attempts
    retry_delay = 5  # Delay between retries in seconds

    while attempt < max_attempts:
        try:
            response = client.chat.completions.create(
                model="gpt-4-1106",
                #model="gpt-35-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "#PLOT: " + plot + '#\n'}
                ],
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            time.sleep(retry_delay)  # Wait before retrying
            attempt += 1

    raise Exception("Failed to enhance plot line after multiple attempts.")

def get_or_create_music_prompt(plot, prompt_file_path):
    """Get the existing music prompt from a file, or create it if it doesn't exist."""
    # Check if the prompt file already exists
    if os.path.exists(prompt_file_path):
        with open(prompt_file_path, 'r') as file:
            music_direction_prompt = file.read()
            if music_direction_prompt:
                return music_direction_prompt
    
    # If the file does not exist or is empty, generate a new prompt and save it
    music_direction_prompt = gpt_musical_prompt_create(plot)
    with open(prompt_file_path, 'w') as file:
        file.write(music_direction_prompt)
    
    return music_direction_prompt

def generate_music(music_direction_prompt, duration, trailer_index):
    # Define the output path for the audio file based on the trailer index
    output_audio_path = os.path.join(get_paths()["trailer_dir"], f'soundtrack_{trailer_index}')

    print(torch.cuda.is_available(), torch.cuda.current_device())
    
    # Check if the audio file already exists and is not empty
    if os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 0:
        print(f"Music file already exists: {output_audio_path}")
    else:
        torch.cuda.empty_cache()
        model = MusicGen.get_pretrained("facebook/musicgen-large")
        
        model.set_generation_params(duration=duration)  # Set duration for music generation

        print(f"Generating music for {duration} seconds...")
        
        max_attempts = 3  # Maximum number of attempts
        i=0
        
        while i < max_attempts:
            try:
                wav = model.generate([music_direction_prompt])  # Generate the music based on the prompt
                break
            except:
                i+=1
                if i==max_attempts:
                    print(f"Failed to generate music after {max_attempts} attempts.")
                    return
                continue

        # Assuming only one wav file is generated; no loop is necessary if you're generating a single file
        audio_path = audio_write(output_audio_path, wav[0].cpu(), model.sample_rate, strategy="rms")
        print(f"Generated and saved: {audio_path}")
        #torch.cuda.empty_cache()
        

    
def music_generation_main(movie_info: MovieInfo):
    logger.info("##### Starting step 6 creating music for trailer #####")
    
    plot_path = get_paths()["plot_path"]
    prompt_file_path = os.path.join(get_project_dir(), 'music_direction_prompt.txt')  # Define the path for your prompt file

    # Open the plot file and read its contents
    with open(plot_path, 'r') as file:
        plot = file.read()

    # Get or create the music direction prompt
    music_direction_prompt = get_or_create_music_prompt(plot, prompt_file_path)
    
    print(music_direction_prompt)
    
    durations = get_video_durations(get_paths()["trailer_dir"])
    print(durations)
    
    for clip_path, duration in durations.items():
        print(clip_path)  # This prints the path for each trailer
        trailer_name = os.path.basename(clip_path)  # Extracts the filename from the path
        
        if "with_voices" in trailer_name:
            continue  # Skip trailers with "with_voices" in their filenames
        
        trailer_index = (os.path.splitext(trailer_name)[0]).split('_')[-1]  # Extracts the trailer index from the filename
        print(duration)  # This prints the duration for each trailer
        
        print(music_direction_prompt)
        print(duration)
        print(trailer_index)
        
        # Call generate_music with the correct arguments
        generate_music(music_direction_prompt, duration, trailer_index)