import logging
from pathlib import Path
from imdb import Cinemagoer
import re
from openai import AzureOpenAI
from dotenv import load_dotenv, find_dotenv, get_key
import time
from common import filter_violent_words, OPENAI_API_VERSION, logger, get_project_dir, get_paths, PROMPTS_DIR
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import shutil
import os
from MovieInfo import MovieInfo

def delete_scene_folders(project_dir: Path) -> None:
    logger.info("Deleting scene folders")
    scene_folders = project_dir.glob("scene_*")
    for folder in scene_folders:
        if folder.is_dir():
            shutil.rmtree(folder)


def call_gpt_synopsis_to_plot(synopsis, number_of_clips=5):
    
    load_dotenv(find_dotenv())
    azure_endpoint = get_key(".env", "AZURE_OPENAI_ENDPOINT")
    api_key = get_key(".env", "AZURE_OPENAI_KEY")
    client = AzureOpenAI(azure_endpoint=azure_endpoint, api_key=api_key, 
                         api_version= OPENAI_API_VERSION)
                         
                         #api_version="2024-02-15-preview")

    # Filter violent words from the synopsis
    filtered_synopsis = filter_violent_words(synopsis)
    
    print(filtered_synopsis)
    
    prompt_path = str(PROMPTS_DIR / "prompt_synopsis_to_plot.txt").replace("\\", "/")
    
    with open(prompt_path, 'r') as file:
        prompt = file.read()

    attempt = 0
    max_attempts = 1
    retry_delay = 1

    while attempt < max_attempts:
        print(f"Attempt {attempt + 1}")
        try:
            response = client.chat.completions.create(
                model="gpt-4-1106",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "#PLOT: " + filtered_synopsis + '#\n#NUMBER OF PHRASES TO GENERATE: ' +  str(number_of_clips) + '#\n'}
                ],
                max_tokens=1000
            )
            print(response.choices[0])
            if response.choices[0].message.content:
                return response.choices[0].message.content
            else:
                continue
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            time.sleep(retry_delay)
            attempt += 1

    logger.error("Failed to enhance plot line after multiple attempts.")
    return None

def get_movie_synopsis(movie, index=0):
    logger.info(f'Fetching synopsis for movie ID: "{movie.movieID}"')
    synopsis_list = movie.get('synopsis', [])

    if synopsis_list:
       
        return synopsis_list[index]
    else:
        logger.error("No synopsis found.")
        return None

def split_text_by_hashtags(text):
    phrases = text.split('#')
    cleaned_phrases = []
    for phrase in phrases:
        phrase = phrase.strip()
        
        if len(phrase) < 1:
            continue
        
        if not phrase.endswith(('.', '!', '?')):
            phrase += '.'
        cleaned_phrases.append(phrase)
    return cleaned_phrases

def get_sub_plots(generated_plot: str, project_dir: Path) -> None:
    logger.info("Generating subplots from the generated plot")
    if generated_plot is None:
        logger.error("Generated plot is None. Cannot generate subplots.")
        return

    subplots = split_text_by_hashtags(generated_plot)
    
    for idx, subplot in enumerate(subplots):
        if subplot:
            scene_plot_path = project_dir / f"scene_{idx+1}" / "plot_line.txt"
            if not scene_plot_path.parent.exists():
                scene_plot_path.parent.mkdir(parents=True)
            scene_plot_path.write_text(subplot)
    
    return subplots

def plot_main(movie_info: MovieInfo, target_clip_count: int) -> None:
    logger.info("##### Starting plot generation from synopsis #####")
    project_dir = get_project_dir()
    
    if not project_dir.exists():
        project_dir.mkdir(parents=True, exist_ok=True)
        
    synopsis_path = get_paths()["synopsis_path"]
    if not synopsis_path.exists():
        synopsis = movie_info.get_synopsis()
        synopsis_path.write_text(synopsis)  # Save the original synopsis for reference
    else:
        synopsis = synopsis_path.read_text()
    
    plot_path = get_paths()["plot_path"]
    
    if synopsis:
        if not plot_path.exists():
            subplots = []
            print("target_clip_count) = " + str(target_clip_count))
            print("len(subplots) = " + str(len(subplots)))
            print("target_clip_count - 1) = " + str(target_clip_count - 1))
            
            while len(subplots) != target_clip_count and len(subplots) != target_clip_count - 1 and len(subplots) != target_clip_count + 1:
            # Call the function to delete the scene folders
                delete_scene_folders(project_dir)
                generated_plot = call_gpt_synopsis_to_plot(synopsis, target_clip_count)
                
                subplots = get_sub_plots(generated_plot, project_dir)
                if subplots is None:
                    continue
                
            plot_path.write_text(generated_plot)

        else:
            generated_plot = plot_path.read_text()
            subplots = get_sub_plots(generated_plot, project_dir)   