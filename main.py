import src.clip as clip
import src.frame as frame
import src.image_retrieval as image_retrieval
import src.join_clip as join_clip
import src.plot as plot
import src.voice as voice
import src.gpt_trailer_speech as gpt_trailer_speech
import src.music_generation as music_generation
import src.trailer_and_soundtrack_assembling as trailer_and_soundtrack_assembling
import src.attach_voices_to_trailer as attach_voices_to_trailer
import src.quotes as quotes
from common import logger, set_configs_path, VOICE_PADDING, NUMBER_RETRIEVED_IMAGES, MOVIES_DIR, HOW_MANY_SECONDS_BUFFER, MAX_SECONDS_BEFORE_CHOSEN_FRAME, TTS_MODEL, SIMILARITY_MODEL_ID, SIMILARITY_BATCH_SIZE
from MovieInfo import MovieInfo
from pathlib import Path

def process_movie(configs_path):
    """
    This is the main function that executes various modules for generating a trailer.
    """
    configs = set_configs_path(configs_path)
    
    movie_info = MovieInfo(configs['movie_id'], configs['movie_path']).fill_with_imdb_info()
    
    frame.frame_main(movie_info)
    
    target_clip_count = configs["target_standard_clip_count"]
    plot.plot_main(movie_info, target_clip_count = target_clip_count)
    
    target_quote_clip_count = configs["target_quote_clip_count"]
    quotes.quotes_main(movie_info, target_quote_clip_count)
    
    
    similarity_model_id = SIMILARITY_MODEL_ID
    similarity_batch_size = SIMILARITY_BATCH_SIZE
    n_retrieved_images = NUMBER_RETRIEVED_IMAGES
    image_retrieval.image_retrieval_main(movie_info, similarity_model_id, similarity_batch_size, n_retrieved_images)
    
    
    how_many_seconds_buffer = HOW_MANY_SECONDS_BUFFER
    max_seconds_before_chosen_frame = MAX_SECONDS_BEFORE_CHOSEN_FRAME
    max_clip_len = configs["max_clip_len"]
    min_clip_len = configs["min_clip_len"]
    clip.clip_main(movie_info, how_many_seconds_buffer, max_seconds_before_chosen_frame, max_clip_len, min_clip_len)
    
    join_clip.join_clip_main(movie_info)
    
    gpt_trailer_speech.gpt_trailer_speech_main(movie_info)

    tts_model = TTS_MODEL
    voice_padding = VOICE_PADDING
    voice.voice_main(movie_info, tts_model, voice_padding)
    
    
    clips_volume = configs["standard_clips_volume_in_dB"]
    voice_volume = configs["voice_over_volume"]
    attach_voices_to_trailer.attach_voices_to_trailer_main(movie_info, clips_volume, voice_volume)

    music_generation.music_generation_main(movie_info)
    
    music_volume_in_dB = configs["music_volume_in_dB"]
    trailer_no_soundtrack_volume_in_dB = configs["trailer_no_soundtrack_volume_in_dB"]
    project_name = configs["project_name"]
    trailer_and_soundtrack_assembling.trailer_and_soundtrack_assembling_main(movie_info, music_volume_in_dB, trailer_no_soundtrack_volume_in_dB, project_name)
    
def main():
    logger.info("Starting the main function")
    
    configs_directory = MOVIES_DIR  # Directory containing all your YAML configuration files
    
    print(len(list(Path(configs_directory).glob("*"))))
    
    for configs_file in Path(configs_directory).glob("*.yaml"):
        logger.info(f"Processing {configs_file}")
        print(str(configs_file))
        
        #FOR TESTING PURPOSES, ONLY ONE MOVIE IS PROCESSED
        if(str(configs_file) == r"movies\interstellar_configs.yaml"):
            process_movie(str(configs_file))

if __name__ == "__main__":
    main()