
# 🎯 English Accent Analyzer

A Python-based tool to extract audio from a public video URL (e.g., YouTube, Loom, MP4) and classify the speaker’s English accent. This can help evaluate spoken English for hiring  purposes.



## ✨ Features

- 🎥 Accepts public video URLs (YouTube, Loom, direct MP4 links)
- 🎧 Extracts and processes audio using `yt-dlp` and `librosa`
- 🗣️ Detects spoken language using OpenAI's Whisper (if available)
- 🌍 Classifies English accent: American, British, Australian, Indian, Canadian, or Irish
- 📊 Returns:
  - Detected language and confidence
  - Accent classification with confidence
  - Human-readable summary
- 🖥️ Works via both:
  - Command-Line Interface (CLI)
  - Web Interface via [Streamlit](https://streamlit.io/)



## 🚀 Quickstart

### 1. Clone the Repository

bash
git clone https://github.com/2002bish/accent_analyzer.py.git



### 2. Install Dependencies

It's recommended to use a virtual environment:
bash
pip install -r requirements.txt


### 3. Run the App

#### Web UI (Streamlit):

bash
streamlit run accent_analyzer.py


#### CLI Mode:

bash
python accent_analyzer.py "https://www.youtube.com/watch?v=VIDEO_ID"


To save results to a JSON file:
bash
python acc.py "https://www.youtube.com/watch?v=VIDEO_ID" --output result.json




## 📦 Dependencies

 `streamlit`
 `yt-dlp`
 `librosa`
 `scikit-learn`
 `numpy`, `scipy`
 `openai-whisper` *(optional, for language detection)*
 `joblib`

Install with:

bash
pip install -r requirements.txt




## 📘 Example Output


URL: https://www.youtube.com/watch?v=xyz
Language Detected: en
English Confidence: 87.5%
Accent Classification: British
Accent Confidence: 74.2%

Summary:
• Language: en (English confidence: 87.5%)
• Accent: British (confidence: 74.2%)
• Strong English speaker detected with clear British accent characteristics




## 🛠️ Notes

* Accent classification is based on a mock-trained model for demonstration purposes. For production, use real-world annotated datasets.
* If Whisper is not installed or available, a fallback method is used to estimate speech activity and English confidence.



## 📄 License

MIT License



## 🙋‍♂️ Author

Created by Bishal Niroula.
