import whisper

def transcribe_audio(file_path):
    model = whisper.load_model("small")
    # Let Whisper auto-detect language by removing 'language="auto"'
    result = model.transcribe(file_path)
    return result["text"]

if __name__ == "__main__":
    text = transcribe_audio("audio_samples/test.wav")
    print("Transcribed text:", text)
