import io
import os
import re  # Regular expressions module
import openai
from fastapi import FastAPI, File, UploadFile
from gradio_client import Client
from pydub import AudioSegment
import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO

# openai.api_key = os.getenv("OPENAI_API_KEY") or "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"
openai.api_key = os.getenv("OPENAI_API_KEY")


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

    # First, generate character descriptions
    character_descriptions = generate_character_descriptions(story)

    # Now, generate key frames and enhance them with character descriptions
    key_frames_with_characters = generate_key_frames_with_characters(story, character_descriptions, duration_seconds)

    # Generate images for each key frame
    for i, frame in enumerate(key_frames_with_characters):
        generate_story_image(frame, i)

    return {"transcription": result, "duration": duration_seconds, "story": story, "character description": character_descriptions, "frames": key_frames_with_characters}


async def generate_text(audio_file):
    """Using Gradio- Generate text from the given audio file."""
    return client.predict(
        audio_file,
        api_name="/predict"
    )


def generate_story(song_lyrics, n):
    """Generate a short story from the given song lyrics."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"""Prompt:
        "Transform the following song lyrics into a short story. Capture the essence, emotions, and imagery conveyed in 
        the lyrics to create a narrative that flows like a song clip. The story should weave together the themes of the 
        song, bring characters to life, and visualize the setting in a way that complements the song's message. 
        Incorporate elements of hope, struggle, love, or any other prominent themes from the lyrics to enrich the story."

        Song Lyrics:
        {song_lyrics}

        Additional Instructions:

        Ensure the story has a clear beginning, middle, and end.
        Include at least one main character who embodies the song's themes.
        Describe settings that reflect the mood and tone of the song.
        If the song mentions specific events or imagery, use them to drive the plot or deepen the narrative.
        Conclude the story in a way that echoes the song's core message or leaves the reader with a thematic takeaway."""}],
        temperature=0.7,
        max_tokens=int(50 * n),
        top_p=1.0,
        api_key=openai.api_key
    )
    # Access the generated content from the choices array
    story = response['choices'][0]['message']['content'].strip()

    return story.replace("\n", " ").strip()


def generate_character_descriptions(story):
    prompt = f"""Examine the short story provided and pinpoint the characters it features. For each character, generate a detailed 
        physical description that captures their defining physical features, specific types of clothing they might wear, 
        and any distinctive accessories that characterize them. If the narrative lacks explicit descriptions of a character's 
        appearance, use contextual clues and your creativity to deduce and elaborate a fitting appearance that aligns with 
        their personality and the story's setting. The descriptions should include the style, type, and color of the clothing, 
        reflecting the character's unique traits and the atmosphere of the story. Present this information in a dictionary format, 
        where each character's name is a key linked to a value containing their comprehensive physical description, including 
        clothing specifics.

    Short Story:
    {story}

    Example of expected output format:
    {{
        "John Doe": "A tall man with short brown hair, wearing glasses, a blue shirt, and dark pants.",
        "Jane Smith": "A woman with shoulder-length curly blonde hair, green eyes, wearing a red dress and a gold necklace."
    }}
    """

    # Call the OpenAI API with the prompt
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024,
        top_p=1.0,
        api_key=openai.api_key
    )

    # Process the response to parse into a dictionary
    # Assuming the model returns the dictionary in the format specified in the prompt
    character_descriptions = eval(response['choices'][0]['message']['content'])

    return character_descriptions


def generate_storyboard(story, n):
    """
    Generate n key frames from the given story.
    :return: A list of n key frames
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": f"""
                Prompt:
                Given the short story below, identify and describe {n} key frames that capture the most significant moments, 
                scenes, or elements of the story. These key frames should visually represent the narrative's progression, 
                highlight important characters, settings, and events. 
                For each key frame, provide a detailed description that includes the characters involved, their actions, 
                the setting, and any notable objects or symbolism. Ensure that each description is detailed enough to be visually represented in an image.
                Additionally, ensure that the key frames create a coherent flow that narratively connects one frame to the next, 
                providing a seamless visual narrative suitable for a video format. 

                Short Story(3D digital art CGI style):
                {story}

                Ensure each key frame is distinct and contributes to the narrative's overall flow. 
                Focus on moments that offer visual, emotional, or thematic depth to the setting. 
                Describe each frame in a way that provides a clear visual guide for image generation and ensures a matching flow for all the frames. 
                The descriptions should be formatted as a list where each item represents a key frame description, seperated by an at (@).
                """
            },
            {
                "role": "user",
                "content": f"""Please identify and describe {n} key frames from the story, ensuring a seamless narrative flow suitable for video. 
                           Format the descriptions as a list, with each item providing a detailed visual guide for a key frame, seperated by an at (@).
                           Example of expected output format: "
                           Lee sitting in his room surrounded by musical instruments and looking thoughtfully at a photo of Darawar, 
                           Lee picking up his guitar and starting to play a melody filled with emotion, 
                           A close-up of Lee's hand strumming the guitar strings, 
                           Lee recording a heartfelt video message for Darawar, 
                           Lee sending the video message and looking hopeful.
                           """
            }
        ],
        temperature=0.7,
        top_p=1.0,
        stop=None,
        api_key=openai.api_key
    )
    # Extracting the generated text
    generated_text = response['choices'][0]['message']['content'].strip().replace("\n", " ")

    # Splitting the generated text into a list of descriptions based on at (@) separator
    frame_descriptions = [desc.strip() for desc in generated_text.split("@") if desc.strip()]

    return frame_descriptions


def generate_key_frames_with_characters(story, character_descriptions, n):
    """
    Generate key frames from the story and add relevant character descriptions to each frame.
    :param story: The story text.
    :param character_descriptions: Dictionary of character names to descriptions.
    :param n: Number of key frames to generate.
    :return: List of key frames with character descriptions.
    """
    key_frames = generate_storyboard(story, n)  # This function is assumed to be defined already

    frames_with_characters = []
    for frame in key_frames:
        # Process each frame to find mentioned characters and add their descriptions
        frame_description = frame
        for character, description in character_descriptions.items():
            if character in frame:
                frame_description += f" {character} Description: {description}"

        frames_with_characters.append(frame_description)

    return frames_with_characters


def generate_story_image(description: str, index: int):
    """
    Generates an image based on a given description using DALL-E,
    and saves the generated image in the 'story_images' directory.

    Parameters:
    - description: A string containing the description of the frame to be visualized.
    """
    # Define the directory to save images
    story_images_dir = "./story_images"
    os.makedirs(story_images_dir, exist_ok=True)

    # Define the output path for the image
    image_filename = f"generated_image{index}.jpg"
    output_path = os.path.join(story_images_dir, image_filename)

    # Generate an image using DALL-E
    try:
        response = openai.Image.create(
            model="dall-e-3",
            prompt="use 3D digital art CGI style: " + description + "make sure to include the precise characters",
            n=1,
            size="1024x1024",
        )

        # Extract the URL of the generated image
        image_url = response['data'][0]['url']

        # Download and save the image
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        image.save(output_path)

        print(f"Image successfully saved to {output_path}")
    except Exception as e:
        print(f"Failed to generate or save image: {e}")




def images_to_video(image_paths, output_video_path, frame_size, fps=1):
    """
    Create a video from a list of images.

    :param image_paths: List of paths to the images.
    :param output_video_path: Path where the output video will be saved.
    :param frame_size: The size of the video frame (width, height).
    :param fps: Frames per second, defaults to 1.
    """
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID' depending on your needs
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    for image_path in image_paths:
        img = cv2.imread(image_path)
        img_resized = cv2.resize(img, frame_size)  # Resize image to match frame size
        out.write(img_resized)

    out.release()


def videos_to_video(video_paths, output_video_path, frame_size, fps=30):
    """
    Concatenate the first second of each video from a list of videos into a single output video.

    :param video_paths: List of paths to the input videos.
    :param output_video_path: Path where the output video will be saved.
    :param frame_size: The size of the video frame (width, height).
    :param fps: Frames per second for the output video, defaults to 30.
    """
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        frames_to_read = fps

        for _ in range(frames_to_read):
            ret, frame = cap.read()
            if not ret:
                print(f"Finished reading video {video_path} early, or error occurred.")
                break
            # Resize frame to ensure consistency
            frame_resized = cv2.resize(frame, frame_size)
            out.write(frame_resized)

        cap.release()

    out.release()


if __name__ == "__main__":
    image_paths = ["story_images/" + path for path in os.listdir("story_images") if re.match(r"generated_image\d+\.jpg", path)]
    print(image_paths)
    output_video_path = 'output_video.mp4'
    frame_size = (1280, 720)  # Width, Height - change according to your needs

    images_to_video(image_paths, output_video_path, frame_size)
    # Example usage:
    # video_paths = ['video1.mp4', 'video2.mp4', 'video3.mp4']  # Add your video paths here
    # output_video_path = 'output_video.mp4'
    # frame_size = (1280, 720)  # Width, Height - change according to your needs
    #
    # videos_to_video(video_paths, output_video_path, frame_size)

# if __name__ == "__main__":
#     import uvicorn
#
#     uvicorn.run(app, host="0.0.0.0", port=8000)

