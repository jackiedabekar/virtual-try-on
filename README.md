# Virtual Try-On Imagen

A virtual try-on solution powered by Google's Gemini 2.5 (Nano Banana Model), enabling advanced garment fitting and image generation.

## Features

- **Virtual Try-On:** Upload your photo and preview clothing on various models.
- **Gemini 2.5 Integration:** Leverages Google's gemini-2.5-flash-preview for rapid, high-quality image synthesis.
- **Intuitive Interface:** Streamlined workflow for easy user interaction.

## Requirements

- Python 3.8 or higher
- [Google Gemini API access](https://ai.google.dev/)
- Dependencies listed in `requirements.txt`

## Installation

```bash
git clone https://github.com/yourusername/virtual-tryon-imagen.git
cd virtual-tryon-imagen
pip install -r requirements.txt
```

## Usage

1. Obtain Gemini 2.5 API credentials.
2. Configure credentials via environment variables or the config file.

## Model Information

- **Model:** gemini-2.5-flash-preview (Nano Banana)
- **Provider:** Google AI
- **Use Case:** Fast, high-fidelity image generation for virtual try-on.

## License

MIT License

## Acknowledgements

- [Google AI Gemini Models](https://ai.google.dev/)
- Open source contributors

## Example Notebooks

- **imagen-v1-gcp.ipynb:** Demonstrates virtual try-on using Google Cloud Platform APIs.
- **imagen-v2-gai.ipynb:** Showcases enhanced image generation with Google AI Studio.