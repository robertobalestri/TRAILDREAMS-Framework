import os
from openai import AzureOpenAI
from dotenv import load_dotenv, find_dotenv, get_key, set_key
import re
import time
import logging
from moviepy.editor import VideoFileClip
from MovieInfo import MovieInfo
from common import PROMPTS_DIR, get_project_dir, get_paths, filter_violent_words, OPENAI_API_VERSION, logger, VOICE_OVER_GENERATION_TIME_INTERVAL

def call_gpt_for_trailer_speech(synopsis, director, release_date, number_of_phrases=5):
    load_dotenv(find_dotenv())

    # Filter violent words from the synopsis
    filtered_synopsis = filter_violent_words(synopsis)
    
    print(filtered_synopsis)
    
    prompt_file_path = str(PROMPTS_DIR / 'prompt_trailer_speech.txt').replace("\\", "/")

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
                    
                    {
                        "role": "user", 
                        "content": "#PLOT: " + synopsis + '#\n' 
                                    + "#DIRECTOR: " + director + '#\n' 
                                    + "#RELEASE DATE: " + release_date + '#\n'
                                    + "#NUMBER OF PHRASES TO GENERATE: " + number_of_phrases + '#\n'
                    }
                ],
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            time.sleep(retry_delay)  # Wait before retrying
            attempt += 1

    raise Exception("Failed to enhance plot line after multiple attempts.")

# Function to split the text by hashtags and return the result as a list of phrases
def split_text_by_hashtags(text: str):
    """
    Splits the text by hashtags and returns a list of phrases.
    If a dot is not present at the end of every phrase, it adds one.
    """
    text = text.strip()
    
    phrases = text.split('#')

    cleaned_phrases = []

    for phrase in phrases:
        phrase = phrase.strip()
        if not phrase.endswith(('.', '!', '?')):
            phrase += '.'
        cleaned_phrases.append(phrase)
    return cleaned_phrases


def gpt_trailer_speech_main(movie_info: MovieInfo):
    logger.info("##### Starting step 7 generating phrases for trailer #####")
    
    project_dir = get_project_dir()  # Base directory for the project
    
    synopsis_path = get_paths()["synopsis_path"]  # Path to the synopsis file
    
    trailer_dir = get_paths()["trailer_dir"]  # Path to the directory containing the trailers
    
    # Define the path for the enhanced plot line file
    trailer_speech_path = os.path.join(project_dir, 'trailer_speech_gpt_generated.txt')

    if not os.path.exists(trailer_speech_path):
        
        print("Entered")
        
        # Open the file and read its contents into a variable
        with open(synopsis_path, 'r') as file:
            synopsis = file.read()
        
        trailers = sorted(trailer_dir.glob("*.mp4"))
        
        if len(trailers) == 0:
            raise Exception("No trailers found in the directory.")
        
        for trailer_path in trailers:
            logger.info(f"Processing {trailer_path}")
            trailer = VideoFileClip(str(trailer_path))

            number_of_phrases =  int(trailer.duration // VOICE_OVER_GENERATION_TIME_INTERVAL)
                            
            # Call the function to enhance the plot lines
            trailer_speech = call_gpt_for_trailer_speech(synopsis, movie_info.get_directors(), movie_info.get_release_date(), number_of_phrases=str(number_of_phrases))
            
            print(trailer_speech)
            
            trailer_speech_list = split_text_by_hashtags(trailer_speech)
            
            # Convert the list into a single string with each item on a new line
            trailer_speech_string = '\n'.join(trailer_speech_list)
            
            # Save the enhanced plot line to the file
            with open(trailer_speech_path, 'w') as file:
                file.write(trailer_speech_string.rstrip('\n'))

    else:
        logger.info("Trailer speech already generated.")

