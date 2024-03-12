# TRAILDREAMS: An LLM-Driven Framework for Automated Trailer Generation

TRAILDREAMS is an innovative framework leveraging a Large Language Model (LLM) to transcend traditional automated trailer production techniques. It employs a comprehensive multimodal strategy, integrating textual analysis with visual and auditory elements, to craft trailers that are not only coherent but deeply engaging.

## Overview

TRAILDREAMS is designed to create captivating previews that entice viewers to watch the full video. Unlike conventional approaches focusing predominantly on visual and auditory elements, TRAILDREAMS emphasizes a multimodal strategy, incorporating narrative context, and semantic insights throughout the trailer production process.

## Installation

1. Clone the TRAILDREAMS repository to your local machine and navigate to the TRAILDREAMS directory.
2. Install the required Python packages using the provided `requirements.txt` file.
3. Install FFmpeg on your system if it's not already installed. This can typically be done through your system's package manager or by downloading it from the FFmpeg official website.

For PyTorch installation, visit the PyTorch website to select the installation command that matches your system setup, especially the version that aligns with your CUDA for GPU support if applicable.

After setting up the environment, download and install the SpaCy English model by executing the SpaCy model download command.

```python -m spacy download en_core_web_sm```

## Preparing the Movie File

1. Place your movie file in the `movies` directory within the TRAILDREAMS project. The movie file should be in MP4 format.
2. An example configuration file, `interstellar_configs.yaml`, is provided in the `movies` folder. Add your "interstellar.mp4" movie file to the same location. If you want to try with other movies, change the configs file accordingly.

## Usage

Generate a trailer with TRAILDREAMS by executing the `main.py` script. Ensure your movie file is in the correct directory and format before starting. The output trailer will be saved inside ```projects\interstellar\trailers\final\interstellar_trailer_1_with_voices_with_soundtrack.mp4```.

## Contributing

Contributions to TRAILDREAMS are encouraged. If you have suggestions for improvements or wish to contribute code, please adhere to the standard GitHub pull request process.

## License

This project is licensed under the MIT License. Refer to the LICENSE.md file for details.

## Citation

If you use TRAILDREAMS in your research, please cite our paper.

## Acknowledgments

- Extend gratitude to any collaborators, funding sources, or other forms of support.

## Contact

For questions or additional support, please contact roberto.balestri2@unibo.it
