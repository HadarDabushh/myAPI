import io
import os
import openai
from fastapi import FastAPI, File, UploadFile
from gradio_client import Client
from pydub import AudioSegment

openai.api_key = os.getenv("OPENAI_API_KEY") or "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"


app = FastAPI()
# Initialize Gradio Client with your Hugging Face Space API endpoint
client = Client("https://jefercania-speech-to-text-whisper.hf.space/--replicas/934ee/")


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    # Read the uploaded file
    audio_data = await audio.read()

    # Create an in-memory file object
    audio_file = io.BytesIO(audio_data)
    # Load the audio file using pydub
    audio_segment = AudioSegment.from_file(audio_file, format=audio.filename.split('.')[-1])
    # Calculate the duration of the audio file in seconds
    duration_seconds = len(audio_segment) // 1000.0

    # so you may need to save the file temporarily (ensure you have permission and space).
    temp_file_path = "temp_" + audio.filename
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(audio_data)

    result = await generate_text(temp_file_path)
    os.remove(temp_file_path)

    story = generate_story(result, duration_seconds)

    return {"transcription": result, "duration": duration_seconds, "story": story}


async def generate_text(temp_file_path):
    # Use Gradio client to predict (transcribe) the audio
    return client.predict(
        temp_file_path,
        api_name="/predict"
    )


def generate_story(song_lyrics, duration_seconds):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"""Prompt:
        "Transform the following song lyrics into a short story. Capture the essence, emotions, and imagery conveyed in the lyrics to create a narrative that flows like a song clip. The story should weave together the themes of the song, bring characters to life, and visualize the setting in a way that complements the song's message. Incorporate elements of hope, struggle, love, or any other prominent themes from the lyrics to enrich the story."

        Song Lyrics:
        {song_lyrics}

        Additional Instructions:

        Ensure the story has a clear beginning, middle, and end.
        Include at least one main character who embodies the song's themes.
        Describe settings that reflect the mood and tone of the song.
        If the song mentions specific events or imagery, use them to drive the plot or deepen the narrative.
        Conclude the story in a way that echoes the song's core message or leaves the reader with a thematic takeaway."""}],
        temperature=0.7,
        max_tokens=int(50 * duration_seconds),
        top_p=1.0,
    )
    # Access the generated content from the choices array
    story = response['choices'][0]['message']['content']

    return story.strip()


def generate_Storyboard(story):
    pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

