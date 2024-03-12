import logging
import shutil
from glob import glob
from pathlib import Path
from dotenv import load_dotenv, find_dotenv, get_key
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from openai import AzureOpenAI
from common import get_paths, get_scenes_dir, OPENAI_API_VERSION, logger, PROMPTS_DIR
import time
import requests
from moviepy.editor import VideoFileClip
import cv2
import easyocr
from typing import List
from MovieInfo import MovieInfo

def get_movie_details(movie_path):
    """Get movie duration and frames per second (FPS)."""
    with VideoFileClip(movie_path) as movie:
        duration = movie.duration
        fps = movie.fps
    return duration, fps

def get_image_embeddings(model: SentenceTransformer, img_filepaths: list[str], batch_size: int) -> np.ndarray:
    """Create embeddings from a set of images."""
    img_emb = model.encode(
        [Image.open(img_filepath) for img_filepath in img_filepaths],
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=True,
    )
    return img_emb

def call_gpt_for_keywords(plot_line, previous_keywords=[]):
    """
    Call GPT for keywords creation based on the plot line.

    Args:
        plot_line (str): The plot line for which keywords need to be generated.
        previous_keywords (list, optional): Previously generated keywords to avoid. Defaults to [].

    Returns:
        str: The generated keywords.

    Raises:
        None

    """
    load_dotenv(find_dotenv())
    azure_endpoint = get_key(".env", "AZURE_OPENAI_ENDPOINT")
    api_key = get_key(".env", "AZURE_OPENAI_KEY")
    if not azure_endpoint or not api_key:
        logger.error("Azure endpoint or API key is not set.")
        return None
    client = AzureOpenAI(azure_endpoint=azure_endpoint, api_key=api_key, api_version=OPENAI_API_VERSION)
    prompt_file_path = str(PROMPTS_DIR / 'prompt_for_extract_keywords_from_plot_line.txt').replace("\\", "/")
    try:
        with open(prompt_file_path, 'r') as file:
            prompt = file.read()
    except IOError as e:
        logger.error(f"Failed to read prompt file: {e}")
        return None
    attempt = 0
    max_attempts = 3
    retry_delay = 5
    while attempt < max_attempts:
        try:
            response = client.chat.completions.create(
                model="gpt-35-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"#EXTRACT KEYWORDS FROM: {plot_line}#\n"},
                    {"role": "user", "content": f"#PREVIOUSLY GENERATED KEYWORDS TO AVOID: {previous_keywords}#\n"},
                ],
                max_tokens=1000
            )
            return response.choices[0].message.content
        except requests.exceptions.RequestException as e:
            logger.error(f"Network-related exception on attempt {attempt + 1}: {e}")
        except Exception as e:
            logger.error(f"General exception on attempt {attempt + 1}: {e}")
        time.sleep(retry_delay)
        attempt += 1
    logger.error("Failed to enhance plot line after multiple attempts.")
    return None

def search(query: str, model: SentenceTransformer, img_emb: np.ndarray, top_k: int) -> list:
    """Search the `top_k` most similar embeddings to a text."""
    query_emb = model.encode([query], convert_to_tensor=True, show_progress_bar=False)
    hits = util.semantic_search(query_emb, img_emb, top_k=top_k)[0]
    return hits

def retrieve_frames_with_keywords(scene_keywords, img_filepaths_sorted, model, img_emb, initial_top_k, duration, fps, increment_step=1, max_top_k=10):
    chosen_frames = []
    min_distance = duration * 0.015

    frame_numbers = [int(Path(path).stem.split('_')[1]) for path in img_filepaths_sorted]
    min_frame_number, max_frame_number = min(frame_numbers), max(frame_numbers)
    forty_percent_frame_number = min_frame_number + (max_frame_number - min_frame_number) * 0.4

    logger.info(f"Min frame number: {min_frame_number}, Max frame number: {max_frame_number}")
    logger.info(f"Forty percent frame number: {forty_percent_frame_number}")
    
    total_scenes = len(scene_keywords)
    first_segment_end_scene = int(total_scenes * 0.4)

    for idx, (scene_dir, keywords) in enumerate(scene_keywords):
        logger.info(f"Retrieving images for scene {idx + 1}")
        frames_dir = Path(f"{scene_dir}/frames")
        if not frames_dir.exists():
            frames_dir.mkdir(parents=True, exist_ok=True)

        if idx < first_segment_end_scene:
            act_specific_img_filepaths = [path for path in img_filepaths_sorted if int(Path(path).stem.split('_')[1]) <= forty_percent_frame_number]
        else:
            act_specific_img_filepaths = [path for path in img_filepaths_sorted if int(Path(path).stem.split('_')[1]) > forty_percent_frame_number]

        act_specific_img_emb_indices = [img_filepaths_sorted.index(path) for path in act_specific_img_filepaths]
        act_specific_img_emb = img_emb[act_specific_img_emb_indices]
        
        logger.info(f"Scene {idx + 1}: selected {len(act_specific_img_filepaths)} frames from range")

        top_k = initial_top_k
        found_frame = False
        
        # Before the while loop in the retrieve_frames_with_keywords function
        logger.info(f"Current scene's frame directory is {frames_dir}")
        
        while top_k <= max_top_k and not found_frame:
            hits = search(keywords, model, act_specific_img_emb, top_k=top_k)

            # Inside the while loop in the retrieve_frames_with_keywords function
            for hit in hits:
                img_filepath = act_specific_img_filepaths[hit["corpus_id"]]
                frame_number = int(Path(img_filepath).stem.split('_')[1])
                frame_time = frame_number / fps

                # Check if the frame satisfies the minimum distance condition and if text is not present
                if all(abs(frame_time - cf_time) >= min_distance for cf_time in chosen_frames):
                    chosen_frames.append(frame_time)
                    destination_frame_name = Path(img_filepath).name
                    destination_path = frames_dir / destination_frame_name  # Ensure this is the correct path for the current scene
                    shutil.copyfile(img_filepath, destination_path)
                    logger.info(f"Copied {img_filepath} to {destination_path}")
                    found_frame = True
                    break


            if not found_frame:
                top_k += increment_step

        if not found_frame:
            logger.info(f"No suitable frame found for scene {idx + 1} even after increasing top_k.")


def image_retrieval_main(movie_info: MovieInfo, similarity_model_id: str, similarity_batch_size: int, n_retrieved_images: int) -> None:
    """Main function to orchestrate the frame retrieval process."""
    logger.info("##### Starting step 3 frame retrieval #####")
    
    duration = movie_info.get_movie_file_duration()
    
    fps = movie_info.get_movie_file_fps()

    # Precompute keywords
    scene_keywords = []
    scenes_dir = get_scenes_dir()

    # Ensure the directories are sorted by scene number
    for i, scene_dir in enumerate(scenes_dir):
        plot_path = Path(f"{scene_dir}/plot_line.txt")
        keywords_path = Path(f"{scene_dir}/keywords.txt")
        if keywords_path.exists():
            with open(keywords_path, 'r') as file:
                keywords = file.read()
        else:
            plot = plot_path.read_text()
            keywords = call_gpt_for_keywords(plot)
            with open(keywords_path, 'w') as file:
                file.write(keywords)
                    
        scene_keywords.append((Path(scene_dir), keywords))
        
        
    # Load SentenceTransformer model
    logger.info(f"Loading {similarity_model_id} as the similarity model")
    model = SentenceTransformer(similarity_model_id)
    img_filepaths = glob(f"{get_paths()['frames_dir']}/*.jpg")
    img_filepaths_sorted = sorted(img_filepaths, key=lambda x: int(Path(x).stem.split('_')[-1]))
    
    logger.info(f"Retrieving from {len(img_filepaths_sorted)} images")
    img_emb = get_image_embeddings(model, img_filepaths_sorted, similarity_batch_size)

    # Retrieve frames based on precomputed keywords
    retrieve_frames_with_keywords(scene_keywords, img_filepaths_sorted, model, img_emb, n_retrieved_images, duration, fps, increment_step=1, max_top_k=10)
