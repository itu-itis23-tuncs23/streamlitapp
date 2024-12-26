import streamlit as st
from transformers import pipeline
import torch

# Streamlit run command: `streamlit run app.py`

# ------------------------------
# Load Whisper Model
# ------------------------------
@st.cache_resource
def load_whisper_model():
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny",
        device="cuda" if torch.cuda.is_available() else "cpu",
        return_timestamps=True
    )


# ------------------------------
# Load NER Model
# ------------------------------
@st.cache_resource
def load_ner_model():
    return pipeline(
        "ner",
        model="dslim/bert-base-NER",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )


# ------------------------------
# Audio Preprocessing Function
# ------------------------------
def preprocess_audio(uploaded_file, whisper_pipeline):
    """
    Preprocess a .wav file for Whisper transcription.
    Args:
        uploaded_file: Uploaded .wav file.
    Returns:
        str: Transcribed text.
    """
    try:
        # Using the whisper model pipeline to transcribe the WAV file
        output = whisper_pipeline(uploaded_file.read()) 
        transcription = output["text"]  # Extracting the transcription
        return transcription
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return ""


# ------------------------------
# Transcription Logic
# ------------------------------
def transcribe_audio(uploaded_file, model):
    """
    Transcribe audio into text using the Whisper model.
    Args:
        uploaded_file: Audio file uploaded by the user.
        model: Whisper model.
    Returns:
        str: Transcribed text from the audio file.
    """
    transcription = preprocess_audio(uploaded_file, model)
    return transcription


# ------------------------------
# Entity Extraction
# ------------------------------
def extract_entities(text, ner_pipeline):
    entities = ner_pipeline(text)
    grouped_entities = {
        "Organizations (ORGS)": set(),
        "Locations (LOCs)": set(),
        "Persons (PERs)": set()
    }
    
    current_entity = ""
    current_type = None
    
    for entity in entities:
        word = entity["word"].replace("##", "")  # Remove subword tokens
        
        if entity["entity"].startswith("B-"):
            if current_entity and current_type:
                # Add the cleaned entity to the correct category
                grouped_entities[current_type].add(current_entity.strip())
            current_entity = word  # Start a new entity
            if "PER" in entity["entity"]:
                current_type = "Persons (PERs)"
            elif "ORG" in entity["entity"]:
                current_type = "Organizations (ORGS)"
            elif "LOC" in entity["entity"]:
                current_type = "Locations (LOCs)"
        elif entity["entity"].startswith("I-") and current_type:
            # Continue the same entity, concatenate without extra spaces
            current_entity += word if word.startswith("##") else f" {word}"
    
    # Add the last entity if it exists
    if current_entity and current_type:
        grouped_entities[current_type].add(current_entity.strip())
    
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

    # Upload audio file
    uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

    # Load models
    whisper_model = load_whisper_model()
    ner_pipeline = load_ner_model()

    if uploaded_file is not None:
        # Transcribe audio
        with st.spinner("Transcribing audio..."):
            transcription = transcribe_audio(uploaded_file, whisper_model)
        st.success("Transcription completed!")
        st.header("Transcription:")
        st.markdown(f'<div class="transcription-text">{transcription}</div>', unsafe_allow_html=True)

        # Extract entities
        with st.spinner("Extracting entities..."):
            entities = extract_entities(transcription, ner_pipeline)
        st.success("Entity extraction completed!")

        st.header("Extracted Entities:")
        for entity_type, entity_set in entities.items():
            if entity_set:
                st.markdown(
                    f'<div style="font-size: 20px; font-weight: bold; margin-top: 10px;">{entity_type}</div>',
                    unsafe_allow_html=True
                )
                st.markdown('<div class="entity-items" style="margin-left: 10px;">', unsafe_allow_html=True)
                for entity in sorted(entity_set):
                    st.markdown(f'<div class="entity-item">• {entity}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
