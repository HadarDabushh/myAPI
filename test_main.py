import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from pydub import AudioSegment
import main

client = TestClient(main.app)

# Mock OpenAI response for generate_story
mock_story_response = {
    'choices': [{
        'message': {
            'content': "Transformed song lyrics into a short story."
        }
    }]
}

# Mock OpenAI response for generate_storyboard
mock_storyboard_response = {
    'choices': [{
        'message': {
            'content': "1. Key moment description. 2. Another key moment description."
        }
    }]
}

@pytest.mark.asyncio
async def test_generate_story():
    song_lyrics = "Mock song lyrics"
    n = 2  # Expected to generate a story with a detailed description

    with patch("openai.ChatCompletion.create", return_value=mock_story_response) as mock_openai:
        story = main.generate_story(song_lyrics, n)
        assert "Transformed song lyrics into a short story." in story
        mock_openai.assert_called_once()

@pytest.mark.asyncio
async def test_generate_storyboard():
    story = "This is a simulated story for testing."
    n = 2  # Expecting 2 key frames based on the story

    with patch("openai.ChatCompletion.create", return_value=mock_storyboard_response) as mock_openai:
        frames = main.generate_storyboard(story, n)
        print("frames: " + str(frames))
        assert len(frames) == n
        assert "Key moment description." == frames[0]
        assert "Another key moment description." == frames[1]
        mock_openai.assert_called_once()

@pytest.mark.asyncio
async def test_transcribe_endpoint_with_mocked_functions():
    with patch("main.generate_text", return_value="Simulated transcription from audio"), \
            patch("main.generate_story", return_value="Simulated story based on lyrics"), \
            patch("main.generate_storyboard", return_value=["Simulated key frame 1", "Simulated key frame 2"]), \
            patch("pydub.AudioSegment.from_file", return_value=AudioSegment.silent(duration=1000)):  # Mock AudioSegment.from_file to return a silent audio segment of 1 second

        # Perform the test with the mocked AudioSegment.from_file
        response = client.post("/transcribe", files={"audio": ("audio.mp3", b"fake audio data", "audio/mp3")})

        assert response.status_code == 200
        data = response.json()
        assert data["transcription"] == "Simulated transcription from audio"
        assert data["story"] == "Simulated story based on lyrics"
        assert len(data["frames"]) == 2
        assert data["frames"][0] == "Simulated key frame 1"
        assert data["frames"][1] == "Simulated key frame 2"
