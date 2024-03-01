import asyncio
import io
import json
import logging
import os
import re
import tempfile
import time
import openai
from fastapi import FastAPI, File, UploadFile, Query
from gradio_client import Client
from pydub import AudioSegment
import cv2
import requests
from PIL import Image
from io import BytesIO
from moviepy.editor import VideoFileClip, AudioFileClip
import ffmpeg

# $env:OPENAI_API_KEY="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"
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()
# Initialize Gradio Client with your Hugging Face Space API endpoint
voice_to_text_client = Client("https://jefercania-speech-to-text-whisper.hf.space/--replicas/934ee/")
audio_to_voice_client = Client("https://broadfield-music-separation.hf.space/--replicas/0vhh5/")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", handlers=[logging.StreamHandler()])


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), song_name: str = Query(default=None), singer: str = Query(default=None)):
    logging.info(f"New API call was made for song '{song_name}' by {singer}!")
    # Generate the first image for the video based on the song name and singer
    generate_first_image(song_name, singer)

    # # Read the uploaded file
    # audio_data = await audio.read()
    # # Create an in-memory file object
    # audio_file = io.BytesIO(audio_data)
    # # Load the audio file using pydub
    # audio_segment = AudioSegment.from_file(audio_file, format=audio.filename.split('.')[-1])
    # # Calculate the duration of the audio file in seconds
    # duration_seconds = len(audio_segment) // 1000.0
    #
    # # so you may need to save the file temporarily (ensure you have permission and space).
    # temp_audio_path = "temp_" + audio.filename
    # with open(temp_audio_path, "wb") as temp_audio_file:
    #     temp_audio_file.write(audio_data)
    #
    # new_audio_path = extract_voice_from_music(temp_audio_path)
    # logging.info(f"vocals path: {new_audio_path}")
    #
    # # # Transcribe the separated voice audio
    # # Transcription = await generate_text(temp_audio_path)
    #
    # with open(new_audio_path, "rb") as audio_file:
    #     Transcription = openai.Audio.transcribe(
    #         model="whisper-1",
    #         file=audio_file,
    #     )
    Transcription = "But you're going, and you know that All you have to do is stay a minute Just take your time, the clock is ticking So stay, all you have to do is wait A second, your hands on mine The clock is ticking, don't stay Thanks for watching!"
    logging.info(f"Transcription: {Transcription}")
    #
    # story = generate_story(Transcription)
    # # story = """
    # # In the dim glow of the streetlamp, on a bench that had known too many farewells, sat Amelia. The city around her throbbed with the pulse of lives moving too quickly, each second slipping through fingers like grains of sand. She clutched her coat tighter against the evening chill, a physical attempt to hold onto something, anything, as everything else seemed to be slipping away.
    # #
    # # Across from her, Michael shuffled his feet, a dance of hesitation. His eyes, usually so full of stories and laughter, were now clouded with a weight Amelia could feel pressing down on them both. They were at a crossroads, a moment suspended in time where every tick of the clock thundered louder than the city's cacophony.
    # #
    # # "But you're going," Amelia whispered, her voice barely rising above the hum of life around them. It wasn't a question. Michael had received an offer, one that would take him thousands of miles away, to a place where their shared moments would be reduced to memories, flickering and fading like the streetlamp's light.
    # #
    # # "And you know that," Michael replied, his voice thick with unspoken emotions. He wanted to tell her that his heart was tethered to this bench, to the imprint her hand left in his. But dreams and duties called with a voice too loud to ignore, promising a future bright but uncertain.
    # #
    # # "All you have to do is stay a minute," Amelia said, her plea hanging in the air between them. She wasn't asking for promises of forever, just a pause, a breath shared in the space between leaving and left. "Just take your time, the clock is ticking."
    # #
    # # Michael sat beside her, the bench creaking under the weight of their shared sorrow. Time, that relentless thief, seemed to slow, granting them a reprieve. "So stay, all you have to do is wait," Amelia continued, her hand finding his in the darkness. For a heartbeat, or perhaps an eternity, they were suspended in a moment where the clock's ticking softened into a gentle lull.
    # #
    # # "A second, your hands on mine," she murmured, tracing the lines of his palm, memorizing the feel of him. In that touch, there was a promise, not of forever, but of now. The city's heartbeat synced with theirs, a reminder that endings were also beginnings.
    # #
    # # "The clock is ticking, don't stay," Michael finally said, his voice a whisper of resignation. It was a sacrifice, a release, because love, he realized, was letting go when every fiber of your being screamed to hold on tighter.
    # #
    # # As he stood, the distance between them grew, not just in steps but in the silent acknowledgment of what must be. "Thanks for watching," Amelia said, her voice steady despite the tears that threatened to fall. It was her gift to him, permission to chase the dreams that awaited, with the hope that one day, in another time, under a different streetlamp, their paths might cross again.
    # #
    # # And so, under the watchful eye of the city, they parted, their story a testament to the moments that define us, the love that shapes us, and the courage it takes to say goodbye. In the end, it wasn't about the staying; it was about the strength found in letting go, and the hope that love, like time, finds a way to endure.
    # # """
    #
    # character_descriptions = generate_character_descriptions(story)
    character_descriptions = {
        "Amelia": "A 25-year-old female with a delicate frame, embodying her mixed European and Latina heritage. Her long, wavy hair cascades down her back, a rich blend of dark brown with subtle caramel highlights, framing her olive-toned skin. Her eyes, a deep hazel, hold a world of emotions, reflecting the complexity of her character. She's wrapped in a thick, woolen coat, its color a soft grey that contrasts with the vibrancy of her red scarf, providing a splash of color against the city's monochrome backdrop. Underneath, she wears a simple, elegant black dress that reaches just above her knees, paired with black tights and ankle boots. A silver locket necklace, a family heirloom, rests against her chest, a constant in her ever-changing world.",
        "Michael": "A 27-year-old male with a sturdy build, his features a testament to his European heritage. His hair is kept short, a neat fade that accentuates his strong jawline. His eyes, a striking gold-brown, seem to capture the light, even in the dimness of the streetlamp. Michael's attire is a careful balance of comfort and style: a dark green button-down shirt, sleeves rolled up to the elbows, revealing a watch with a leather strap on his left wrist; black jeans that fit just right; and well-worn leather boots. A small, leather-bound notebook peeks out from his shirt pocket, a companion to his thoughts and dreams."
    }
    logging.info(f"Character descriptions: {character_descriptions}")
    #
    # if not character_descriptions:
    #     return {"error": "Failed to generate character descriptions."}
    #
    # frames = generate_storyboard(story)
    # frames_with_characters = generate_key_frames_with_characters(frames, character_descriptions)
    # logging.info(f"Frames with characters: {frames_with_characters}")
    #
    # await generate_final_video(frames_with_characters, temp_audio_path)
    # # await add_subtitles_to_video()
    #
    # # Remove temporary files
    # os.remove(temp_audio_path)
    #
    # return {
    #     "transcription": Transcription,
    #     "duration": duration_seconds,
    #     "story": story,
    #     "character_description": character_descriptions,
    #     "frames": frames_with_characters
    # }


async def generate_final_video(final_frames, audio_path):
    # Generate images for each key frame
    for i, frame in enumerate(final_frames):
        generate_story_image(frame, i+1)

    num_files = len([name for name in os.listdir("story_images") if name.endswith(".jpg")])
    image_paths = [f"story_images\\generated_image{i}.jpg" for i in range(num_files)]
    output_video_path = 'output_video.mp4'
    frame_size = (1024, 1024)  # Width, Height - change according to your needs

    images_to_video_with_transitions(image_paths, output_video_path, frame_size)
    add_audio_to_video(audio_path=audio_path)

async def generate_text_with_fallback(temp_audio_path):
    try:
        # Attempt to use the first method with a timeout of 20 seconds
        result = await asyncio.wait_for(generate_text(temp_audio_path), timeout=20)
    except asyncio.TimeoutError:
        # If the first method takes longer than 20 seconds, fallback to the second method
        with open(temp_audio_path, "rb") as audio_file:
            result = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file,
            )
            # Assuming you want to extract text from the response directly here
            result = result['data']['text']
    except Exception as e:
        # Handle other possible exceptions from generate_text or file operations
        logging.error(f"An error occurred: {e}")
        result = None

    return result


async def add_subtitles_to_video(video_path=os.path.abspath("final_output_video.mp4"), subtitles_path=os.path.abspath("subtitles.srt"), output_path=os.path.abspath("final_video_with_subtitles.mp4")):
    """
    Adds subtitles to a video file and saves the output to a new file.

    Args:
    - video_path: Path to the input video file.
    - subtitles_path: Path to the .srt or .vtt subtitles file.
    - output_path: Path where the output video file with subtitles should be saved.
    """
    try:
        # Use ffmpeg to add subtitles to the video
        (
            ffmpeg
            .input(video_path)
            .output(output_path, vf=f'subtitles={subtitles_path}')
            .run(overwrite_output=True)
        )
        logging.info(f"Video with subtitles saved to {output_path}")
    except ffmpeg.Error as e:
        logging.error(f"Failed to add subtitles to video: {e}")

async def generate_text(audio_file):
    """Using Gradio- Generate text from the given audio file."""
    return voice_to_text_client.predict(
        audio_file,
        api_name="/predict"
    )


def extract_voice_from_music(audio_file):
    """Using Gradio- Extract voice from the given audio file."""
    result = audio_to_voice_client.predict(
        audio_file,
        api_name="/predict"
    )
    return result[0]


def generate_first_image(song_name, singer):
    """Generate the first image for the video based on the song name and singer."""
    # Define the directory to save images
    story_images_dir = "story_images"
    os.makedirs(story_images_dir, exist_ok=True)

    # Define the output path for the image
    image_filename = f"generated_image0.jpg"
    output_path = os.path.join(story_images_dir, image_filename)
    response = openai.Image.create(
        model="dall-e-3",
        prompt=f"""Generate an opening image for a music video focusing on the song's name and author: "{song_name} by {singer}". 
               Please don't forget to include the song's name and the singer's name on the image and don't include any other text.""",
        n=1,
        size="1024x1024",
        api_key=openai.api_key
    )

    image_url = response['data'][0]['url']
    # Download and save the image
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    image.save(output_path)

    logging.info(f"First image successfully saved !")


def generate_story(song_lyrics):
    """Generate a short story from the given song lyrics."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo-preview",
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
            top_p=1.0,
            api_key=openai.api_key
        )
        # Access the generated content from the choices array
        story = response['choices'][0]['message']['content'].strip()
    except Exception as e:
        logging.error(f"Error generating story: {e}")
        return

    logging.info(f"Story generated: {story}")
    return story.replace("\n", " ").strip()


def generate_character_descriptions(story, max_retries=3):
    prompt = f"""
        Based on the story provided, create a detailed and specific physical description for each character mentioned. 
        The description should be precise enough to ensure the character's visual representation remains consistent across various images. 
        
        Highlight each character's ethnicity, exact hairstyle and color, eye color, and any unique facial or skin features.
        Pinpoint each character's gender and approximate age. Describe the exact outfit they are wearing, including garment types, colors, and any distinctive designs. 
        Include descriptions of any consistent accessories they carry or wear that are significant to their character or any current body condition relevant to the story. 
        If the story does not provide detailed descriptions, use creative license to develop a fitting appearance based on the character's role and actions.
        Present the descriptions in a dictionary format with each character's name as a key and their detailed description as the value. 
        
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
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=1024,
                top_p=1.0,
                api_key=openai.api_key
            )

            response_content = response['choices'][0]['message']['content'].strip().replace("\n", " ")
            logging.info(response_content)
            # Parse the response content as eval
            character_descriptions = eval(response_content)
            logging.info(f"Character descriptions generated!")
            return character_descriptions  # Return the descriptions if parsing is successful

        except json.JSONDecodeError as e:
            print(f"Attempt {attempt + 1} failed with a JSON decode error: {e}")
            time.sleep(1)  # Wait for 1 second before retrying to avoid hitting the API too quickly

        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed with an error: {e}")
            time.sleep(1)  # Wait for 1 second before retrying

    logging.error("All attempts to generate character descriptions have failed.")
    return {}


def generate_storyboard(story, n=19):
    """
    Generate n key frames from the given story.
    :return: A list of n key frames
    """
    # prompt = f"""
    #             Provide {n} key frames from the following story that illustrate the unfolding of events and the story's progression.
    #             Each key frame should be conceptualized as a continuation of the last,
    #             showcasing a pivotal change that advances the narrative. From a director's point of view,
    #             describe the situation in such a way that allows for the creation of an accurate image depicting the scene.
    #             In your descriptions, detail the positioning of characters, the setting,
    #             and any visible changes to the scene that highlight the story's development.
    #             Emphasize how each frame builds upon the previous one, marking a significant moment or transition in the storyline.
    #
    #             Story:
    #             {story}
    #
    #             Instructions for Key Frames:
    #             - Ensure each key frame visually builds upon the last, capturing a critical moment or turning point in the story that clearly demonstrates the narrative progression.
    #             - Your descriptions should serve as precise visual scripts for CGI image generation, focusing on the spatial arrangement of characters, environmental settings, and visible details, including the changes from the previous frame.
    #             - Highlight the emotional tone, dynamics between characters, and significant environmental or situational changes to enrich the visual storytelling.
    #             - Format your descriptions as a list of {n} items, with each entry providing a detailed description of a key frame, emphasizing the transition and change from the previous scene, separated by an at (@).
    #
    #             Get as much into detail as possible. The more detailed the better.
    #             """
    prompt = f"""As im generating a video clip for a song based on the following story, generate {n} key frames, separeted by an at '@' symbol, that illustrate the unfolding of events and the story's progression.
                Each key will be generated as a dall-e prompt and will be used to generate a visual representation of the story.
                
                Story:
                {story}
                
                Important points:
                - Remember that each frame will be sent as an independent dall-e prompt so make sure to include all the necessary setting details in each and every frame.
                - Also focus on the characters and their interaction and positions in the scene. (Include their specific names and dont use pronouns to refer to them)
                - The more detailed the better. What ever you thing is important to include, include it. Each frame will be sent to generate a visual representation of the frame and all together will show the story flow.
                - Provide the frames in a list format, separated by an at (@) symbol.
                """
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        top_p=1.0,
        stop=None,
        api_key=openai.api_key
    )
    # Extracting the generated text
    generated_text = response['choices'][0]['message']['content'].strip().replace("\n", " ")

    frame_descriptions = [desc.strip() for desc in generated_text.split("@") if desc.strip()]
    logging.info(f"Storyboard generated")
    return frame_descriptions


def improved_frames(key_frames, character_descriptions):
    """
    Generate improves key frames to using the characters description.
    :param key_frames: List of existing key frames.
    :return: List of more detailed frames.
    """
    improved_frames_list = []
    for i, frame in enumerate(key_frames):
        # Dynamically create character description part of the prompt
        characters_prompt_part = " and knowing the characters:\n"
        for name, description in character_descriptions.items():
            characters_prompt_part += f"{name}, described briefly as {description},\n"

        # Add context from adjacent frames if available
        if i > 0:  # Previous frame exists
            prev_frame_context = f"Previously, {key_frames[i - 1]}"
        else:
            prev_frame_context = "This is the beginning of the story."

        if i < len(key_frames) - 1:  # Next frame exists
            next_frame_context = f"Following this, {key_frames[i + 1]}"
        else:
            next_frame_context = "This is the end of the story."

        # Adjust the prompt for concise and visually focused output with contextual cues
        prompt = f"""
            Given the scene description:
            "{frame}"
            {characters_prompt_part}
            With the context: "{prev_frame_context}" and "{next_frame_context}",
            craft a brief, visually rich narrative that integrates these character traits and contextual cues into the scene.
            Focus on key elements crucial for an image aimed at visual storytelling within a story video.
            Ensure the narrative captures the essence of the moment, the characters, and their interaction in a concise manner.

            Enclose the improved scene description within '@' symbols for clarity.
            """
        try:
            # Call the OpenAI API with the transition_description
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.7,
                top_p=1.0,
            )

            response_content = response['choices'][0]['message']['content'].strip().replace("\n", " ")
            improved_frame = re.search(r'@(.+?)@', response_content).group(1).strip()
            improved_frames_list.append(improved_frame)
            logging.info(f"Improved frame generated")

        except Exception as e:
            logging.error(f"Error generating Improved frame: {e} \n response_content: {response_content}")

    return improved_frames_list


def generate_key_frames_with_characters(key_frames, character_descriptions):
    """
    Generate key frames from the story and add relevant character descriptions to each frame.
    :param key_frames: List of key frames generated from the story.
    :param character_descriptions: Dictionary of character names to descriptions.
    :return: List of key frames with character descriptions.
    """
    frames_with_characters = []
    for frame in key_frames:
        # Process each frame to find mentioned characters and add their descriptions
        frame_description = frame
        for character, description in character_descriptions.items():
            if character in frame:
                if character.lower() in frame.lower():
                    frame_description += f" {character} Description: {description}"

        frames_with_characters.append(frame_description)
        logging.info(f"Generated frame with characters!")
    return frames_with_characters


def generate_story_image(description: str, index: int):
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
    prompt = f"""Generate an image in a 3D digital art CGI style based on the following description: 
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

        response_url = response['data'][0]['url']
        print(f"original image url!!!!!!!: {response_url}")

        try:
            logging.info("First time trying to check image")
            image_url = check_image(response_url, description)
            if image_url != response_url:
                logging.info(f"Second time trying to check image")
                image_url = check_image(image_url, description)
        except Exception as e:
            logging.error(f"Image hasn't change: {e}")
            image_url = response_url
        # Download and save the image
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        image.save(output_path)

        logging.info(f"Image successfully saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to generate or save image: {e}")


def check_image(image_path, frame_with_character_description: str):
    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": """Check if this image is suitable for the following scene and note any elements that are not consistent with the characters description.
                    Description:"""},
                    {"type": "text", "text": frame_with_character_description},
                    {"type": "image_url", "image_url": image_path},
                    {"type": "text", "text": "If it does, response with a simple 'yes'. Else, provide a better description for dall-e to generate an image for the same scene. DO NOT provide any other explenation, just the new description as a dall-e prompt."},
                ],
            }
        ],
        max_tokens=4096,
    )

    vision_response = response['choices'][0]['message']['content'].strip().replace("\n", " ")
    logging.info(f"Image check response: {vision_response}")
    corrected_image = image_path
    if "yes" != vision_response.lower()[0:3]:
        logging.info(f"Image not suitable, generating a new image based on the corrected description.")
        image_response = openai.Image.create(
            model="dall-e-3",
            prompt=f"In a 3D digital art CGI style: {vision_response[4:]} + {frame_with_character_description}",
            n=1,
            size="1024x1024",
            api_key=openai.api_key,
        )

        corrected_image = image_response['data'][0]['url']

    return corrected_image


def images_to_video_with_transitions(image_paths, output_video_path, frame_size, overall_fps=40, transition_frames=15,
                                     display_duration=1):
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


def add_audio_to_video(video_path="output_video.mp4",
                       audio_path="a-short-song-in-case-you-re-feeling-down-(mp3convert.org).mp3",
                       output_video_path="final_output_video.mp4"):
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

#
# if __name__ == "__main__":
#     num_files = len([name for name in os.listdir("story_images") if name.endswith(".jpg")])
#     image_paths = [f"story_images\\generated_image{i}.jpg" for i in range(num_files)]
#     output_video_path = 'output_video.mp4'
#     frame_size = (1024, 1024)  # Width, Height - change according to your needs
#
#     images_to_video_with_transitions(image_paths, output_video_path, frame_size)
#     add_audio_to_video(audio_path="temp_WhatsApp Ptt 2024-02-21 at 23.06.56.ogg")
    # Example usage:
    # video_paths = ['video1.mp4', 'video2.mp4', 'video3.mp4']  # Add your video paths here
    # output_video_path = 'output_video.mp4'
    # frame_size = (1280, 720)  # Width, Height - change according to your needs
    #
    # videos_to_video(video_paths, output_video_path, frame_size)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
