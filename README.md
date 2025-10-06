TITLE
Building a Speech to Text Transcriptor with PyTorch QuartzNet

INTRODUCTION
This project turns speech into text and can optionally translate the text. It offers two engines. A local PyTorch QuartzNet path when you want control and tunability, and an API path using Whisper for convenience. The app accepts audio and video uploads or a YouTube link, converts media to a standard wav format with ffmpeg, runs transcription, and can compute quality metrics when you provide a reference transcript. It also supports extracting text from documents and translating longer texts in chunks so the workflow stays practical for real tasks. 


WHY PYTORCH AND QUARTZNET
I chose PyTorch with QuartzNet to retain control over the model and features. With PyTorch I can manage audio preprocessing, spectrogram parameters, decoding, and domain adaptation for accents or noise profiles. The QuartzNet flow uses an explicit AudioProcessor for mel spectrograms, CTC friendly text normalization, and greedy decoding, which makes the system transparent and adjustable when accuracy matters or when running in constrained or private environments. 

TOOL KEY FEATURES
Local QuartzNet or Whisper selection. Switch engines in the UI. 



 

streamlit_app


YouTube ingestion and general audio or video uploads with automatic conversion to mono 16 kHz wav using ffmpeg. 



Optional accuracy checks. Compute WER and CER by pasting ground truth. One variant also calculates BLEU when a reference is provided. Results are saved to JSON for later analysis. 



Text utilities. Extract text from txt, md, csv, docx, or pdf. Translate long texts in chunks with a translator prompt that preserves meaning and structure. 




Practical guards. File size checks for the Whisper path and clear messages when audio exceeds API limits. ffmpeg presence checks with a quick version display. 



TECHNOLOGIES USED
PyTorch for model execution and custom audio feature handling. 



QuartzNet architecture for CTC based speech recognition via local checkpoints or NeMo restoration when using a .nemo file. 



Streamlit for a simple end to end UI that orchestrates inputs, transcription, metrics, and translation. 




ffmpeg and ffprobe for robust media handling. 


Optional OpenAI Whisper API as an alternate engine and translation for multi language support. 

app

 


PROJECT WORKFLOW
Prepare media. The app checks ffmpeg and ffprobe, then converts uploads or a YouTube clip into a standard 16 kHz wav. 




Choose engine. Use local QuartzNet with a checkpoint path when you need control, or select Whisper for a hosted option with a size guard. 



Transcribe. For QuartzNet, the AudioProcessor generates log mel spectrograms and handles decoding. For Whisper, the app streams the wav to the API. 

my_utils

 

app


Evaluate. Optionally paste a reference to compute WER and CER. One variant also writes BLEU and runtime details to the results JSON. 

streamlit_app


Translate. If enabled, long texts are split into chunks with a translator prompt that preserves names, numbers, tone, and layout. Outputs are saved and can be downloaded. 

app

 


KEY FINDINGS AND BUSINESS IMPLICATIONS
Two paths cover different needs. A lightweight API route is quick to integrate, while the PyTorch route gives you control, repeatability, and the option to fine tune for domain audio. This mirrors the tradeoffs described in the accompanying write up comparing a RAG and LangChain style orchestration to a heavier QuartzNet flow. The right choice depends on cost, data privacy, and the need for customization. 

Bernard Griffin final report


Measurable quality. The app persists WER and CER results so teams can track improvements over time and attach evidence during model changes or data cleaning. Sample JSON result files are included to show how outputs are stored for review. 

wer_results_20250910_094557

 

wer_results_20250910_094612


Operational practicality. The system handles real inputs such as YouTube and large documents and provides clear error messages for oversize audio in the API path. This lowers friction for analysts and non engineers using the tool day to day. 

app

 

streamlit_app

CONCLUSION
This project balances convenience and control. When speed matters, the Whisper path works out of the box with sensible size checks. When you need transparency, tunability, or to operate in restricted environments, PyTorch with QuartzNet lets you own preprocessing, decoding, and adaptation strategies. The UI and saved metrics make it easy to compare runs, share results, and move toward a production setup. 

app

 

my_utils

 

streamlit_app
