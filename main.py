import io
import json
import logging
import os
import re  # Regular expressions module
import time

import numpy as np
import openai
from fastapi import FastAPI, File, UploadFile
from gradio_client import Client
from pydub import AudioSegment
import cv2
import requests
from PIL import Image
from io import BytesIO
from moviepy.editor import VideoFileClip, AudioFileClip


# $env:OPENAI_API_KEY="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"
openai.api_key = os.getenv("OPENAI_API_KEY")


app = FastAPI()
# Initialize Gradio Client with your Hugging Face Space API endpoint
voice_to_text_client = Client("https://jefercania-speech-to-text-whisper.hf.space/--replicas/934ee/")
audio_to_voice_client = Client("https://mohamedrashad-audio-separator.hf.space/")
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler()])

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    logging.info(f"New API call was made!")
    # Read the uploaded file
    audio_data = await audio.read()
    # Create an in-memory file object
    audio_file = io.BytesIO(audio_data)
    # Load the audio file using pydub
    audio_segment = AudioSegment.from_file(audio_file, format=audio.filename.split('.')[-1])
    # Calculate the duration of the audio file in seconds
    duration_seconds = len(audio_segment) // 1000.0

    # so you may need to save the file temporarily (ensure you have permission and space).
    temp_audio_path = "temp_" + audio.filename
    with open(temp_audio_path, "wb") as temp_audio_file:
        temp_audio_file.write(audio_data)

    # Separate voice from music using the external API
    # _, separated_audio_path = await separate_voice_music(temp_audio_path)

    # TO USE THE SEPARATED AUDIO CHANGE THE PATH TO THE SEPARATED_AUDIO_PATH
    # Transcribe the separated voice audio
    # result = await generate_text(temp_audio_path)

    with open(temp_audio_path, "rb") as audio_file:
        result = openai.Audio.transcribe(
            model="whisper-1",
            file=audio_file,
        )
    # result = "I'll place my heart on your hand made bruises Heal your scars without lame excuses I can be your happy place With me you won't know heartache I'll place my heart on your hand Made bruises Heal your scars without lame excuses I can be your happy place With me we won't know heartache"
    logging.info(f"Transcription: {result}")

    # Remove temporary files
    os.remove(temp_audio_path)
    # os.remove(separated_audio_path)

    story = generate_story(result, duration_seconds)
    # story="""In a small town nestled among rolling hills and lush greenery, there lived a young woman named Lily. She carried the weight of the world on her shoulders, her heart heavy with the scars of a painful past. Her once vibrant spirit had dimmed, overshadowed by the bruises life had left on her soul.
    #     One day, a stranger arrived in town, his presence a breath of fresh air in the stagnant atmosphere. His name was Jack, and he exuded a warmth that drew Lily to him like a moth to a flame. In him, she saw a glimmer of hope, a promise of healing that she had longed for but never dared to believe in.
    #     As they spent time together, Jack showed Lily a kindness she had never known. He listened to her stories without judgment, held her hand through her dar
    #     kest moments, and slowly but surely, he began to mend the wounds that had festered within her for so long. With each passing day, Lily felt herself coming alive again, her heart blooming like a flower after a long winter.
    #     Jack became her happy place, a sanctuary in a world that had often been cruel and unforgiving. He showed her that love could be gentle and patient, that it could heal even the deepest of scars. And in his arms, Lily found a peace she had thought was lost to her forever.
    #     Together, they walked through the town, hand in hand, their laughter ringing through the streets like a melody. They were a picture of hope and resilience, two souls brought together by fate to find solace in each other's presence.
    #     And as the sun set on the horizon, painting the sky in hues of pink and gold, Lily knew that she had found her heart's true home in Jack. He had placed
    #     his heart on her hand-made bruises, healed her scars without lame excuses, and in doing so, had shown her that with him, she would never know heartache again.
    #     In that moment, Lily realized that love had the power to transform even the darkest of nights into the brightest of days. And as she looked into Jack's
    #     eyes, she knew that she was finally free to embrace the happiness she had so longed for. Together, they stood strong, their hearts entwined in a bond that would weather any storm, their love a beacon of light in a world that often seemed shrouded in darkness.
    #     """
    character_descriptions = generate_character_descriptions(story)
    frames = generate_storyboard(story, duration_seconds)
    key_frames_with_characters = generate_key_frames_with_characters(frames, character_descriptions)

    # Generate images for each key frame
    for i, frame in enumerate(key_frames_with_characters[:15]):
        generate_story_image(frame, i)

    return {
        "transcription": result,
        "duration": duration_seconds,
        "story": story,
        "character_description": character_descriptions,
        "frames": frames
    }

async def generate_text(audio_file):
    """Using Gradio- Generate text from the given audio file."""
    return voice_to_text_client.predict(
        audio_file,
        api_name="/predict"
    )


async def separate_voice_music(audio_file):
    return audio_to_voice_client.predict(
            audio_file,
            api_name="/predict"
    )


def generate_story(song_lyrics, n):
    """Generate a short story from the given song lyrics."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"""Prompt:
            "Transform the following song lyrics into a short story. Capture the essence, emotions, and imagery conveyed in 
            the lyrics to create a narrative that flows like a song clip. The story should weave together the themes of the 
            song, bring characters to life, and visualize the setting in a way that complements the song's message. 
            Incorporate elements of hope, struggle, love, or any other prominent themes from the lyrics to enrich the story.
    
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
    except Exception as e:
        logging.error(f"Error generating story: {e}")
        story = "Failed to generate a story from the song lyrics."

    logging.info(f"Story generated: {story}")
    return story.replace("\n", " ").strip()


def generate_character_descriptions(story, max_retries=3):
    prompt = f"""
        Based on the story provided, create concise yet specific physical descriptions for each character mentioned.
        Focus on the most distinctive features that will ensure the characters are recognizable and consistent across various images.
        Keep the descriptions brief but clear, highlighting only:
        
        - Pinpoint each character's gender and approximate age.
        - The most defining aspect of their ethnicity or facial features.
        - Key elements of their hairstyle and color, and eye color.
        - Detail the characters' clothing with specificity, including colors and exact items they are wearing in this scene, to reflect their personality and the context of the moment.
        - One accessory or item that is particularly significant to their character or to the story at this moment.
        
        Use creative license to fill in gaps from the story as needed, but keep the focus narrow to maintain brevity.
        Present the descriptions in a dictionary format with each character's name as a key and their succinct description as the value.

        Story:
        {story}

        Example of expected output format:
        {{
            "Alex": "A mid-30s female with sharp features, embodying her Irish descent. Her fiery red hair is cut short, complementing her piercing blue eyes. She's dressed in a sleek, black motorcycle jacket over a white tank top, dark jeans, and combat boots, with a silver pendant necklace.",
            "Michael": "A 22-year-old male with a lean build, showcasing his East Asian heritage through his gentle facial structure. His hair is dyed platinum blonde, adding contrast to his dark brown eyes. Wears casual graphic tees, ripped jeans, and sneakers, always seen with his vintage camera."
        }}
        """

    for attempt in range(max_retries):
        try:
            # Call the OpenAI API with the prompt
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1024,
                top_p=1.0,
                api_key=openai.api_key
            )

            # Extract the response content
            response_content = response['choices'][0]['message']['content'].strip().replace("\n", " ")
            logging.info(response_content)
            # Parse the response content as eval
            character_descriptions = eval(response_content)
            logging.info(f"Character descriptions generated!")
            return character_descriptions  # Return the descriptions if parsing is successful

        except json.JSONDecodeError as e:
            # If a JSON decode error occurred, print an error message and retry
            print(f"Attempt {attempt + 1} failed with a JSON decode error: {e}")
            time.sleep(1)  # Wait for 1 second before retrying to avoid hitting the API too quickly

        except Exception as e:
            # If any other error occurred, print an error message and retry
            logging.error(f"Attempt {attempt + 1} failed with an error: {e}")
            time.sleep(1)  # Wait for 1 second before retrying

    # If all retries failed, return an empty dictionary or raise an exception
    logging.error("All attempts to generate character descriptions have failed.")
    return {}


def generate_storyboard(story, n):
    """
    Generate n key frames from the given story.
    :return: A list of n key frames
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"""
                You are creating a music video based on the following story, Provide {n} detailed key scenes (Separated by an at '@') 
                that will be visualized in images to show the story's flow. 
                Each scene should focus on just one interaction, setting, and the moment itself. 
                Position the characters as if they were actors within a movie scene, engaged with each other or the environment, 
                rather than facing the audience.
                
                Story:
                {story}

                Instructions for Key Frames:
                - Emphasize the characters' engagement with their surroundings or each other, and specify the direction of their gaze.
                - Describe the spatial arrangement, the frame's view, and environmental settings as a director would for a movie scene.
                - Highlight emotional tones, character dynamics, and environmental changes to enrich visual storytelling.
                - Make sure that you focus only on one instruction each frame.
                - Format the descriptions with detailed scenes, each separated by an at (@), focusing on narrative progression and character interaction within the scene.
                
                Detail is crucial. The more specific the instructions, the more focused the resulting images will be on the storytelling elements.
                """
                # As I'm creating a music video based on the following story, I need {n} detailed key scenes (separated by an at '@')
                # that will be visualized in images (using DALL-E) to show the story's flow.
                # Each scene should be rich in details to paint a complete picture in at least 5 sentences,
                # focusing on the interactions, setting, and the moment itself, position the characters as if they were actors.
                #
                # Story:
                # {story}
                #
                # Instructions for Key Frames:
                # - Focus on the spatial arrangement of characters, environmental settings, the frames view, and visible details to serve as
                # precise visual scripts for CGI image generation in DALL-E.
                # - Highlight the emotional tone, dynamics between characters, and significant environmental or situational
                # changes to enrich the visual storytelling and making the frame as a directors movie scene.
                # - Ensure each key frame visually builds upon the last, capturing critical situations or turning points in
                # the story that clearly demonstrate the narrative progression from scene to scene.
                # - Format your descriptions as a list of {n} items, with each entry providing a detailed description of a
                # key frame, separated by an at ('@').
                #
                # Get as much into detail as possible. The more detailed the better.
                # """
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
    if len(frame_descriptions) < n:
        frame_descriptions = [desc.strip() for desc in re.split(r'\d+\.', generated_text) if desc.strip()]

    logging.info(f"Storyboard generated: {frame_descriptions}")
    return frame_descriptions


def extand_frames(key_frames):
    """
    Generate intermediate key frames to create smooth transitions between existing key frames.
    :param key_frames: List of existing key frames.
    :return: List of original and intermediate key frames.
    """
    expanded_frames = []
    for i in range(len(key_frames)):
        expanded_frames.append(key_frames[i])
        if i < len(key_frames) - 1:  # If not the last frame
            # Generate a description for an intermediate frame
            transition_description = f"""
                Prompt:
                We are creating a transitional scene to connect two existing key frames and provide a smooth narrative
                progression for an animated video. Please craft a description for a transitional scene that bridges the
                gap between the following moments in the narrative. The description should reflect subtle changes in setting,
                character emotions, and actions that lead from one key frame to the next.

                End of Frame {i}:
                {key_frames[i]}

                Beginning of Frame {i + 1}:
                {key_frames[i + 1]}

                Please describe the transitional scene with sufficient detail for creating a 3D digital art CGI style
                image that captures the essence of a story's progression. Your description should be enclosed
                within '@' symbols for clarity.

                Format your response as:
                '@transitional scene description@'
                """
            try:
                # Call the OpenAI API with the transition_description
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": transition_description}],
                    temperature=0.7,
                    top_p=1.0,
                )
                response_content = response['choices'][0]['message']['content'].strip()

                # Use regex to extract the text enclosed within '@' symbols
                intermediate_frame = re.search(r'@(.+?)@', response_content).group(1).strip()
                logging.info(f"Intermediate frame generated")
            except Exception as e:
                logging.error(f"Error generating intermediate frame: {e} \n response_content: {response_content}")
                intermediate_frame = f"Failed to generate a transitional scene between frames {i} and {i + 1}."

            expanded_frames.append(intermediate_frame)

    return expanded_frames


def generate_key_frames_with_characters(key_frames, character_descriptions):
    """
    Generate key frames from the story and add relevant character descriptions to each frame.
    :param key_frames: List of key frames generated from the story.
    :param character_descriptions: Dictionary of character names to descriptions.
    :param n: Number of key frames to generate.
    :return: List of key frames with character descriptions.
    """
    frames_with_characters = []
    for frame in key_frames:
        # Process each frame to find mentioned characters and add their descriptions
        frame_description = frame
        for character, description in character_descriptions.items():
            if character.lower() in frame.lower():
                frame_description += f" {character} Description: {description}"

        frames_with_characters.append(frame_description)

    return frames_with_characters


def generate_story_image(description: str,index: int):
    """
    Generates an image based on a given description using DALL-E,
    and saves the generated image in the 'story_images' directory.

    Parameters:
    - description: A string containing the description of the frame to be visualized.
    """
    # Define the directory to save images
    story_images_dir = "story_images"
    os.makedirs(story_images_dir, exist_ok=True)

    # Define the output path for the image
    image_filename = f"generated_image{index}.jpg"
    output_path = os.path.join(story_images_dir, image_filename)
    prompt = f"""
        Create an image that captures both the essence of the scene and the characters within it, 
        focusing equally on the atmosphere and the character interactions. 
        Ensure that the characters are integrated into the scene in a way that supports the overall mood and theme, 
        without overshadowing the environmental and atmospheric details. 
        The image should be a balanced composition of character and scene, illustrating the story's emotions and settings vividly.
        Style: 3D digital art CGI
        Description:
        {description}
        """

    # Generate an image using DALL-E
    try:
        response = openai.Image.create(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024",
            api_key=openai.api_key
        )

        # Extract the URL of the generated image
        image_url = response['data'][0]['url']

        # Download and save the image
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        image.save(output_path)

        logging.info(f"Image successfully saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to generate or save image: {e}")


def images_to_video_with_transitions(image_paths, output_video_path, frame_size, overall_fps=40, transition_frames=15, display_duration=1):
    """
    Create a video from a list of images with smooth transitions, and each image shown for a longer duration.

    :param image_paths: List of paths to the images.
    :param output_video_path: Path where the output video will be saved.
    :param frame_size: The size of the video frame (width, height).
    :param overall_fps: Overall frames per second for the video, defaults to 30.
    :param transition_frames: Number of frames to use for each transition.
    :param display_duration: Duration (in seconds) to display each image before transitioning, defaults to 2 seconds.
    """
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, overall_fps, frame_size)

    previous_image = None
    # Calculate the number of frames to hold each image based on the display duration
    hold_frames = int(overall_fps * display_duration)

    for image_path in image_paths:
        img = cv2.imread(image_path)
        img_resized = cv2.resize(img, frame_size)

        if previous_image is not None:
            # Create smooth transition
            for t in range(transition_frames + 1):
                alpha = t / transition_frames
                transition_image = cv2.addWeighted(previous_image, 1 - alpha, img_resized, alpha, 0)
                out.write(transition_image)

        # Display the current image for the specified duration
        for _ in range(hold_frames):
            out.write(img_resized)

        previous_image = img_resized

    out.release()


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


def add_audio_to_video(video_path="output_video.mp4", audio_path="a-short-song-in-case-you-re-feeling-down-(mp3convert.org).mp3", output_video_path="final_output_video.mp4"):
    """
    Adds an audio track to a video file.

    :param video_path: Path to the original video file.
    :param audio_path: Path to the audio file to add to the video.
    :param output_video_path: Path where the output video file with the added audio will be saved.
    """
    # Load the video file
    video_clip = VideoFileClip(video_path)
    # Load the audio file
    audio_clip = AudioFileClip(audio_path)
    # Set the audio of the video clip as the audio clip
    video_clip_with_audio = video_clip.set_audio(audio_clip)
    # Write the result to the output file
    video_clip_with_audio.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
    # Close the clips to free up system resources
    video_clip.close()
    audio_clip.close()


if __name__ == "__main__":
    # Determine the number of image files in the directory
    num_files = len([name for name in os.listdir("story_images") if name.endswith(".jpg")])
    # Construct the paths with ordered numbers from 0 to n
    image_paths = [f"story_images\\generated_image{i}.jpg" for i in range(num_files)]
    print(image_paths)
    output_video_path = 'output_video.mp4'
    frame_size = (1024, 1024)  # Width, Height - change according to your needs

    images_to_video_with_transitions(image_paths, output_video_path, frame_size)
    add_audio_to_video()
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

