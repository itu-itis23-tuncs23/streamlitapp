import streamlit as st
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import torchaudio
import torch

# Streamlit run command: `streamlit run app.py`

# ------------------------------
# Load Whisper Model
# ------------------------------
@st.cache_resource
def load_whisper_model():
    """
    Load the Whisper model and processor for audio transcription.
    Returns:
        tuple: (WhisperProcessor, WhisperForConditionalGeneration)
    """
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    return processor, model


# ------------------------------
# Load NER Model
# ------------------------------
@st.cache_resource
def load_ner_model():
    """
    Load the Named Entity Recognition (NER) model pipeline.
    Returns:
        transformers pipeline
    """
    return pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)


# ------------------------------
# Audio Preprocessing Function
# ------------------------------
def preprocess_audio(uploaded_file):
    """
    Preprocess a .wav file for Whisper transcription.
    Args:
        uploaded_file: Uploaded .wav file.
    Returns:
        torch.Tensor: Preprocessed audio waveform.
        int: Sampling rate (16,000 Hz).
    """
    # Load the audio file using torchaudio
    waveform, sample_rate = torchaudio.load(uploaded_file)

    # Resample to 16kHz if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    # Convert to mono if the audio has multiple channels
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Normalize the audio to the range [-1.0, 1.0]
    waveform = waveform / waveform.abs().max()

    return waveform, sample_rate


# ------------------------------
# Transcription Logic
# ------------------------------
def transcribe_audio(uploaded_file, processor, model):
    """
    Transcribe audio into text using the Whisper model.
    Args:
        uploaded_file: Audio file uploaded by the user.
        processor: Whisper processor.
        model: Whisper model.
    Returns:
        str: Transcribed text from the audio file.
    """
    # Preprocess the audio
    waveform, sample_rate = preprocess_audio(uploaded_file)

    # Prepare input for the Whisper model
    inputs = processor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt")

    # Generate transcription
    transcription_ids = model.generate(**inputs)

    # Decode the transcribed text
    transcribed_text = processor.batch_decode(transcription_ids, skip_special_tokens=True)[0]

    return transcribed_text


# ------------------------------
# Entity Extraction
# ------------------------------
def extract_entities(text, ner_pipeline):
    """
    Extract entities from transcribed text using the NER model.
    Args:
        text (str): Transcribed text.
        ner_pipeline: NER pipeline loaded from Hugging Face.
    Returns:
        dict: Grouped entities (ORGs, LOCs, PERs).
    """
    entities = ner_pipeline(text)
    grouped_entities = {"ORG": [], "LOC": [], "PER": []}
    for entity in entities:
        entity_type = entity["entity_group"]
        if entity_type in grouped_entities:
            grouped_entities[entity_type].append(entity["word"])
    return grouped_entities


# ------------------------------
# Main Streamlit Application
# ------------------------------
def main():
    st.title("Meeting Transcription and Entity Extraction")

    # Replace with student details
    STUDENT_NAME = "Sude Dilay Tunç"
    STUDENT_ID = "150230716"
    st.write(f"**{STUDENT_ID} - {STUDENT_NAME}**")

    # Load models
    processor, whisper_model = load_whisper_model()
    ner_pipeline = load_ner_model()

    # Upload audio file
    uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

    if uploaded_file is not None:
        # Transcribe audio
        with st.spinner("Transcribing audio..."):
            transcription = transcribe_audio(uploaded_file, processor, whisper_model)
        st.success("Transcription completed!")
        st.text_area("Transcribed Text", transcription, height=200)

        # Extract entities
        with st.spinner("Extracting entities..."):
            entities = extract_entities(transcription, ner_pipeline)
        st.success("Entity extraction completed!")

        # Display entities
        st.subheader("Extracted Entities")
        for entity_type, entity_list in entities.items():
            st.write(f"**{entity_type}:** {', '.join(set(entity_list))}")


if __name__ == "__main__":
    main()
