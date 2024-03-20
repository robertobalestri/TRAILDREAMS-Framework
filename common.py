import logging
import re
from glob import glob
from pathlib import Path
from moviepy.editor import VideoFileClip
import yaml
import re


def parse_configs(configs_path: str) -> dict:
    """Parse configs from the YAML file.

    Args:
        configs_path (str): Path to the YAML file

    Returns:
        dict: Parsed configs
    """
    configs = yaml.safe_load(open(configs_path, "r"))
    logger.info(f"Configs: {configs}")
    return configs


#CONFIGS_PATH = "movies/interstellar_configs.yaml"
#CONFIGS_PATH = "movies/night_of_the_living_dead_configs.yaml"
#CONFIGS_PATH = "movies/grand_budapest_hotel_configs.yaml"
#CONFIGS_PATH = "movies/300_configs.yaml"
#CONFIGS_PATH = "movies/the_hobbit_an_unexpected_journey_configs.yaml"
#CONFIGS_PATH = "movies/1917_configs.yaml"
#CONFIGS_PATH = "movies/mission_impossible_configs.yaml"
#CONFIGS_PATH = "movies/fight_club_configs.yaml"
#CONFIGS_PATH = "movies/the_lord_of_the_rings_the_return_of_the_king_configs.yaml"
#configs = parse_configs(CONFIGS_PATH)


# Function to set the current configuration path
def set_configs_path(configs_path: str):
    global configs
    configs = parse_configs(configs_path)
    return configs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


def get_project_dir():
    return Path(f"./projects/{configs['project_name']}")

def get_paths():
    project_dir = get_project_dir()
    return {
        "plot_path": Path(f"{project_dir}/plot.txt"),
        "synopsis_path": Path(f"{project_dir}/synopsis.txt"),
        "frames_dir": Path(f"{project_dir}/frames"),
        "trailer_dir": Path(f"{project_dir}/trailers"),
        "quotes_path": Path(f"{project_dir}/quotes.txt"),
        "quote_clips_dir": Path(f"{project_dir}/quote_clips"),
        "audios_dir": Path(f"{project_dir}/audios"),
        "only_vocals_quote_clips_dir": Path(f"{project_dir}/quote_clips/only_vocals_quote_clips"),
    }
    
MOVIES_DIR = Path("movies")
VOICES_DIR = Path("voices")
PROMPTS_DIR = Path("prompts")

### CONFIGURATIONS THAT CAN'T BE CHANGED BY USER
OPENAI_API_VERSION = "2023-09-01-preview"
MAX_DURATION_QUOTE_CLIP = 12
FRAME_EXTRACTION_TIME_INTERVAL = 9
VOICE_OVER_GENERATION_TIME_INTERVAL = 18
HOW_MANY_SECONDS_BUFFER = 10
MAX_SECONDS_BEFORE_CHOSEN_FRAME = 3
TTS_MODEL = "xtts"
SIMILARITY_BATCH_SIZE = 128
SIMILARITY_MODEL_ID = "clip-ViT-L-14"
VOICE_PADDING = 0
NUMBER_RETRIEVED_IMAGES = 1
###

def get_scenes_dir() -> list[str]:
    """Get the list of scene directories."""
    project_dir = get_project_dir()
    scenes_dir = glob(f"{project_dir}/scene_*")
    # Use Path to convert scene_dir from string to Path object before accessing .name
    scenes_dir = sorted(scenes_dir, key=lambda s: int(Path(s).name.split('_')[1]))  # Natural sort
    
    # Convert backslashes to forward slashes for consistency
    scenes_dir = [s.replace("\\", "/") for s in scenes_dir]
    
    return scenes_dir

# List of violent words to filter out
violent_words = [
    'kill', 'murder', 'assault', 'attack', 'beat', 'fight', 'stab', 'shoot', 'massacre', 'terrorize', 'abuse',
    'batter', 'harm', 'injure', 'strangle', 'slaughter', 'maim', 'decapitate', 'mutilate', 'ravage', 'demolish',
    'destroy', 'choke', 'kidnap', 'ambush', 'explode', 'bomb', 'arson', 'torture', 'war', 'conflict', 'aggression',
    'hostility', 'brutality', 'carnage', 'bloodshed', 'homicide', 'manslaughter', 'genocide', 'violence', 'weapon',
    'gun', 'knife', 'missile', 'bullet', 'explosive', 'detonate', 'attacks', 'struggles', 'knocked unconscious',
    'dead', 'pursues', 'drift', 'abandons', 'deserted', 'exploring', 'gruesome find', 'partially devoured', 'panic',
    'flee', 'intercepted', 'smashing skulls', 'hysteria', 'barricade', 'hammer and nails', 'rifle', 'shooting',
    'mayhem', 'defend', 'attackers', 'fight off', 'rescue operations', 'consuming their victims\' flesh', 'gunshot',
    'heavy blow to the head', 'armed men', 'patrolling', 'throw Molotov cocktails', 'flaming torch', 'explodes',
    'punches', 'cowardice', 'zombies eat', 'eating', 'charred remains', 'living dead', 'attacks', 'wrest', 'shoot',
    'dies', 'transformed', 'consuming', 'stabs', 'killing', 'overwhelmed', 'barricades', 'catatonic state', 'battle',
    'mob', 'retreats', 'locks', 'reanimate', 'shoots', 'make his last stand', 'posse', 'kills', 'fights', 'struggle',
    'struggles', 'struggling', 'struggled', 'struggle', 'struggles', 'struggles', 'struggles', 'struggles', 'struggles',
    'violently', 'smashing', 'smashes', 'smashed', 'smash', 'smashes', 'smashes', 'smashes', 'smashes', 'smashes',
    'shot', 'corpse', 'trap', 'fuck', 'fucking', 'fucks', 'death', 'killed', 'gun', 'bomb', 'burn', 'fight', 'hit', 
    'steal', 'threaten', 'assault', 'vandalism', 'sabotage', 'castration', 'suicide', 'explosives', 'nitroglycerin', 
    "gunpoint", "chemical", "burn", "horrific", "threatens", "steals", "soapmaking", "rite of passage", "hazing", 
    "beating", "vandalism", "disfigure", "threat", "demolition", "combat", "sabotage", "crash", "fistfight", "detonates", "explosives",
    'fighting', 'wound', 'destruction', 'destroyed', "gun", "shot", "fight", "blood", "assaulted", "beat", "pistol", "fighting", 
    "shooting", "bomb", "explosives", "detonates", "detonation", 'assassin', 'assassins', 'amputated', 'bodies', 'victims', 'monsters',
    'ripped', 'grabbed', 'cadaver', 'limbs', 'fuckin\'', 'fucking', 'motherfucker', 'negro', 'nigga', 'niggers', 'nigger', 
]

def filter_violent_words(text, substitute_with_redacted=True):
    # Pattern to capture words, considering punctuation, quotes, and brackets
    # This pattern will match words in various contexts, including those surrounded by punctuation or within quotes/brackets
    pattern = r"\b\w+\b|'[\w]+'|\"[\w]+\"|\([\w]+\)|[\w]+[\.,!?;]"

    # Find all matching instances based on the pattern
    words = re.findall(pattern, text)
    
    # Filter and replace violent words
    # When a word is identified, it's checked against the violent_words list after stripping surrounding punctuation/quotes
    if substitute_with_redacted:
        filtered_words = ['REDACTED' if re.sub(r'[\'\".,!?;()]', '', word).lower() in violent_words else word for word in words]
    else:
        filtered_words = [' ' if re.sub(r'[\'\".,!?;()]', '', word).lower() in violent_words else word for word in words]

    # Reconstruct the sentence from the filtered words
    return ' '.join(filtered_words)

def get_video_file_details(movie_path):
    """Get movie duration and frames per second (FPS)."""
    movie_path = str(movie_path)
    with VideoFileClip(movie_path) as movie:
        duration = movie.duration
        fps = movie.fps
    return duration, fps