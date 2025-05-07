# Import libraries
import streamlit as st
import whisper
import openai
from openai import OpenAI
import os

# --- Page Config ---
st.set_page_config(page_title="Better Mockingbird", layout="centered")

st.title(":red[🎙️ Better Mockingbird 📝]")
st.markdown("Upload or record audio → Transcribe with Whisper Automatic Speech Recognition → Fix errors using Large Language Models")

# --- API Key Mode ---
st.sidebar.header("🔐 API Key Settings")
key_mode = st.sidebar.radio("Access OpenAI API with:", ["Use demo key", "Paste my key"])

if key_mode == "Paste my key":
    user_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
    if user_key:
        openai.api_key = user_key
    else:
        st.warning("Please enter an API key.")
        st.stop()
else:
    openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("openai_api_key")
    if not openai.api_key:
        st.error("No API key set. Use secrets.toml or environment variable.")
        st.stop()
        
# --- LLM Mode ---
st.sidebar.header("🤖 Error Correction Agent")
llm_mode = st.sidebar.radio("Choose the LLM for error correction:", ["⚡️Turbo: GPT-4o", "🎯Economy: GPT-4o mini"])


# --- Step 1: Audio Input ---
st.header("Step 1: Upload or Record Audio")
input_method = st.radio("Choose input method:", ["Upload Audio", "Record Audio"])
audio_file = None

# Upload option
if input_method == "Upload Audio":
    audio_file = st.file_uploader("Upload a WAV or MP3 file", type=["wav", "mp3"])

# Record option
if input_method == "Record Audio":
    st.write("🎤 Click below to start/stop recording")
    audio_file = st.audio_input("Record a voice message")
    if audio_file:
        st.audio(audio_file)

# Transcription trigger
if audio_file and st.button("Transcribe Audio"):
    st.session_state.audio_path = audio_file

# --- Step 2: Whisper Transcription ---
if st.session_state.get("audio_path") and not st.session_state.get("transcript_complete"):
    st.header("Step 2: Whisper Transcription")
    status_placeholder = st.empty()
    status_placeholder.info("Transcribing with Whisper...")
    try:
        # Using whisper large-v2 via OpenAI API
        client = OpenAI(api_key=openai.api_key)
        with st.spinner("Sending audio to OpenAI ..."):
            transcript_response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

            # Save the transcription to session state
            hypotheses = [transcript_response.text]
            for temp in [0.5, 0.6, 0.7, 0.8]:
                hypotheses.append(client.audio.transcriptions.create(model="whisper-1", file=audio_file, temperature=temp).text)

            st.session_state.hypotheses = hypotheses
            status_placeholder.success("✅ Transcription complete!")      
            st.session_state.transcript_complete = True

    except Exception as e:
        status_placeholder.error("❌ Transcription failed.")
        st.error(f"{e}")
        st.stop()

if st.session_state.get("transcript_complete"):
    hypotheses = st.session_state.get("hypotheses", [])
    prompt = f"""You are given 5 ASR transcription hypotheses. Identify the words in the first hypothesis that are suspicious, meaning they differ from the majority of the other hypotheses or are likely incorrect considering the context. Return the first hypothesis with suspicious words wrapped in <span style="color:red;">word</span> tags for highlighting.
                Do not output any additional text that is not the highlighted version of the first hypothesis.\n
                Do not write any explanatory text that is not the highlighted version of the first hypothesis.\n
                Here are the hypotheses:
                1. {hypotheses[0]}
                2. {hypotheses[1]}
                3. {hypotheses[2]}
                4. {hypotheses[3]}
                5. {hypotheses[4]}
                """
    if llm_mode == "⚡️Turbo: GPT-4o":
        llm_model = "gpt-4o"
    else:
        llm_model = "gpt-4o-mini"
    client = OpenAI(api_key=openai.api_key)
    completion = client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                **{"max_tokens": 250, "temperature": 0.9})
    highlighted = completion.choices[0].message.content

    st.write("📝 Whisper Output:")
    st.markdown(highlighted, unsafe_allow_html=True)
    st.checkbox("Do you have in-context error examples?", value=False, key="enable_fewshot")
    error_examples = []
    if st.session_state.get("enable_fewshot"):
        example_text = ("<hypothesis1> the hamlet of whitewell likes to the west </hypothesis1>\n"
                        "<hypothesis2> the hamlet of white will lights to the west </hypothesis2>\n"
                        "<hypothesis3> the hamlet of whitewell lies to the west</hypothesis3>\n"
                        "<hypothesis4> the hamlet of whitewill lies to the west </hypothesis4>\n"
                        "<hypothesis5> the hamlet of whiteville likes to the west </hypothesis5>\n"
                        "Your output: the hamlet of whitewell lies to the west")
        error_examples = st.text_area("Example 1", value=example_text, height=200)
        
    st.session_state.error_examples = error_examples    
    if st.button("Apply LLM Correction"):
        st.session_state.correction_ready = True

# --- Step 3: Error Correction ---
if st.session_state.get("correction_ready"):
    st.header("Step 3: LLM Error Correction")

    # Create the prompt for the LLM
    prompt = ("You are a helpful assistant that corrects ASR errors. You will be presented with ASR transcription hypotheses and your task is to correct any errors in it and generate one output sentence.\n"
    "If you come across errors in ASR transcription, make corrections that closely match the original transcription acoustically or phonetically\n"
    "If you encounter grammatical errors, provide a corrected version adhering to proper grammar.\n"
    "Provide the most probable corrected transcription in string format.\n"
    "Do not output any additional text that is not the corrected transcription.\n"
    "Do not write any explanatory text that is not the corrected transcription.\n"
    "Here are the hypotheses:\n")
    
    for idx, hypothesis in enumerate(st.session_state.get("hypotheses", [])):
        prompt += "<hypothesis" + str(idx + 1) + ">" + hypothesis + "</hypothesis" + str(idx + 1) + ">\n"

    # Add error examples
    if st.session_state.get("enable_fewshot"):
        prompt += "\nHere are few in-context examples:\n\n" + st.session_state.get("error_examples", []) + '\n' + "\nFeel free to refer to these examples. Please start:\n"

    # Select the model
    if llm_mode == "⚡️Turbo: GPT-4o":
        llm_model = "gpt-4o"
    else:
        llm_model = "gpt-4o-mini"

    with st.spinner(f"Correcting transcript using {llm_model}..."):
        try:
            client = OpenAI(api_key=openai.api_key)
            completion = client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                **{"max_tokens": 250, "temperature": 0.9}
            )
            corrected = completion.choices[0].message.content
            st.text_area("✅ Corrected Output", corrected)

        except Exception as e:
            st.error(f"Error: {e}")

    if st.button("🔁 Start Over"):
        st.session_state.clear()
        st.rerun()