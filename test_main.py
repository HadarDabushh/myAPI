import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import main

client = TestClient(main.app)

# Mock OpenAI responses based on your specified structure
mock_story_response = {
    'choices': [{'message': {'content': "In a quiet town, two friends embark on an adventure."}}]
}
mock_character_description_response = {
    'choices': [{'message': {'content': '{"Alex": "A curious adventurer", "Jordan": "A wise companion"}'}}]
}
mock_storyboard_response = {
    'choices': [{'message': {'content': "Alex and Jordan set out.@They face challenges.@Victory is theirs."}}]
}
mock_check_image_yes_response = {
    'choices': [{'message': {'content': "yes"}}]
}
mock_check_image_new_description_response = {
    'choices': [{'message': {'content': "A better description for the scene"}}]
}


@pytest.fixture
def mock_openai_response():
    with patch('openai.ChatCompletion.create') as mock_create:
        mock_create.return_value = mock_story_response
        yield mock_create


# Test generating a story from given inputs
def test_generate_story(mock_openai_response):
    story = main.generate_story("Dummy lyrics")
    assert "In a quiet town, two friends embark on an adventure." in story
    mock_openai_response.assert_called_once()


# Test generating character descriptions from a story
def test_generate_character_descriptions(mock_openai_response):
    mock_openai_response.return_value = mock_character_description_response
    descriptions = main.generate_character_descriptions("Dummy story")
    expected_descriptions = {"Alex": "A curious adventurer", "Jordan": "A wise companion"}
    assert descriptions == expected_descriptions
    mock_openai_response.assert_called_once()


# Test generating a storyboard from a story
def test_generate_storyboard(mock_openai_response):
    mock_openai_response.return_value = mock_storyboard_response
    frames = main.generate_storyboard("Dummy story", 3)
    expected_frames = ["Alex and Jordan set out.", "They face challenges.", "Victory is theirs."]
    assert frames == expected_frames
    mock_openai_response.assert_called_once()


# Test checking an image and receiving a positive response
def test_check_image_yes(mock_openai_response):
    mock_openai_response.return_value = mock_check_image_yes_response
    response = main.check_image("Dummy image path", "Dummy description")
    assert response == "Dummy image path"
    mock_openai_response.assert_called_once()


# Test the combining of key frames with character descriptions
def test_generate_key_frames_with_characters():
    key_frames = [
        "Alex and Jordan set out on their journey.",
        "Alex encounter a mysterious forest.",
        "Jordan find a hidden treasure."
    ]
    character_descriptions = {
        "Alex": "A curious adventurer with short brown hair.",
        "Jordan": "A wise companion with a keen eye for detail."
    }
    expected_frames_with_characters = [
        "Alex and Jordan set out on their journey. Alex Description: A curious adventurer with short brown hair. Jordan Description: A wise companion with a keen eye for detail.",
        "Alex encounter a mysterious forest. Alex Description: A curious adventurer with short brown hair.",
        "Jordan find a hidden treasure. Jordan Description: A wise companion with a keen eye for detail."
    ]
    frames_with_characters = main.generate_key_frames_with_characters(key_frames, character_descriptions)
    assert frames_with_characters == expected_frames_with_characters


# Test the generate_story_image function's behavior
def test_generate_story_image_calls_check_image_no_more_than_twice():
    with patch('main.openai.Image.create', return_value=MagicMock(data=[{'url': 'mock_url'}])) as mock_dalle:
        with patch('main.requests.get') as mock_requests_get:
            with patch('main.Image.open') as mock_image_open:
                mock_image_open.return_value.save = MagicMock()
                with patch('main.check_image') as mock_check_image:
                    # Simulate the first check_image call returning a new URL, and the second call confirming the image
                    mock_check_image.side_effect = ['new_image_url', 'new_image_url']
                    main.generate_story_image("A description of the scene", 1)

                    # Verify check_image was called no more than twice
                    assert mock_check_image.call_count <= 2, "check_image was called more than twice"


def test_video_generation():
    # Assuming you have a function that generates the video from images
    with patch('main.images_to_video') as mock_images_to_video:
        main.images_to_video(["path/to/image1", "path/to/image2"], "path/to/output/video")
        mock_images_to_video.assert_called_once()
