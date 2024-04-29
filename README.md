# TRAILDREAMS: An LLM-Driven Framework for Automated Trailer Generation

TRAILDREAMS is an innovative framework leveraging a Large Language Model (LLM) to transcend traditional automated trailer production techniques. It employs a comprehensive multimodal strategy, integrating textual analysis with visual and auditory elements, to craft trailers that are not only coherent but deeply engaging.

## Overview

TRAILDREAMS is designed to create captivating previews that entice viewers to watch the full video. Unlike conventional approaches focusing predominantly on visual and auditory elements, TRAILDREAMS emphasizes a multimodal strategy, incorporating narrative context, and semantic insights throughout the trailer production process.

## Installation

1. Clone the TRAILDREAMS repository to your local machine and navigate to the TRAILDREAMS directory.
2. Install Microsoft C++ Build Tools at ```https://visualstudio.microsoft.com/it/visual-cpp-build-tools/```
3. Create a virtual environment. Install the required Python packages using the provided `requirements.txt` file.
4. Due to some package incompatibilies, you should install TTS with ```pip install TTS``` and then you should uninstall PyTorch and all its packages with ```pip uninstall torch``` (confirm with "y" when requested by the terminal).
5. For PyTorch installation, visit the PyTorch website ```https://pytorch.org/get-started/locally/```. For our testing we used CUDA 12.1 and we ran ```pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121``` to install a PyTorch version that works with our requirements.
6. Install FFmpeg on your system if it's not already installed. This can typically be done through your system's package manager or by downloading it from the FFmpeg official website.
7. Download and install the SpaCy English model by executing the SpaCy model download command.
```python -m spacy download en_core_web_sm```
8. Add a .env file to the root folder with two variables: AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY. NOTE: You'll need an Azure OpenAI API key.
9. Type ```huggingface-cli login``` in the terminal (virtual environment should be activated). Generate a token from your HuggingFace account and paste it when requested.

## Preparing the Movie File

1. Place your movie file in the `movies` directory within the TRAILDREAMS project. The movie file should be in MP4 format.
2. An example configuration file, `interstellar_configs.yaml`, is provided in the `movies` folder. Add your "interstellar.mp4" movie file to the same location. If you want to try with other movies, change the configs file accordingly.

## Usage


Inside the `main.py` script, there's a line that specifies the movie to be processed. For testing purposes, the script is currently set to process only the "Interstellar" movie, as defined by the `interstellar_configs.yaml` file in the `movies` directory.
Execute the `main.py` script and the output trailer will be saved inside ```projects\interstellar\trailers\final\interstellar_trailer_1_with_voices_with_soundtrack.mp4```.


## Usage
## Contributing

Contributions to TRAILDREAMS are encouraged. If you have suggestions for improvements or wish to contribute code, please adhere to the standard GitHub pull request process.

## License

This project is licensed under the MIT License. Refer to the LICENSE.md file for details.

## Citation

If you use TRAILDREAMS in your research, please cite our paper.

## Contact

For questions or additional support, please contact roberto.balestri2@unibo.it
