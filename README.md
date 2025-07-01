# Vietnamese ASR with Whisper

This project provides a pipeline for Vietnamese Automatic Speech Recognition (ASR) using OpenAI's Whisper model, including data preprocessing, model fine-tuning, and a Streamlit web app for inference.

## Project Structure

```
VietASR/
├── app.py                  # Streamlit web app for ASR demo
├── scripts/                # Python scripts for data processing, training, etc.
├── notebooks/              # Jupyter notebooks for experiments and documentation
├── models/                 # Saved and fine-tuned model weights
├── data/
│   ├── raw/                # Raw audio and transcript data
│   └── processed/          # Processed audio and CSVs
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # Project documentation
```

## Setup

1. Clone the repository and navigate to the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Install system dependencies for audio processing:
   ```bash
   sudo apt-get update && sudo apt-get install -y ffmpeg
   ```

## Usage

- **Data Preprocessing:**
  - Use scripts or notebooks in `notebooks/` and `scripts/` to preprocess and split your data.
- **Model Training:**
  - Fine-tune Whisper using provided notebooks or scripts.
- **Web Demo:**
  - Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
  - Upload or record audio to get Vietnamese transcriptions.

## Folders
- `scripts/`: Python scripts for automation and batch processing.
- `notebooks/`: Jupyter notebooks for step-by-step experiments.
- `models/`: Store your fine-tuned Whisper models here.
- `data/raw/`: Place your original audio and transcript files here.
- `data/processed/`: Processed audio and CSVs for training/testing.

## Notes
- Make sure to update paths in scripts and notebooks according to this structure.
- Large files and model weights are ignored by git (see `.gitignore`).

## Credits
- Built with [Hugging Face Transformers](https://huggingface.co/docs/transformers/index), [Streamlit](https://streamlit.io/), and [OpenAI Whisper](https://github.com/openai/whisper).
