import streamlit as st
import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import sys
import requests
import json
import riva
from pathlib import Path
from typing import List, Optional, Tuple, Union
import grpc
from typing import Generator, Optional, Union
from grpc._channel import _MultiThreadedRendezvous
import riva.client.proto.riva_tts_pb2 as rtts
import riva.client.proto.riva_tts_pb2_grpc as rtts_srv
from riva.client import Auth
from riva.client.proto.riva_audio_pb2 import AudioEncoding
import wave

def configure():
    load_dotenv()

configure()

st.title("Vision Voice")
st.write("Put in image, that you want to be described in format png or jpeg: ")

image_data = st.file_uploader("Choose a file")

if image_data is not None:
    st.image(image_data)
    # Define endpoint and key in Google Colab
    os.environ["VISION_ENDPOINT"] = os.getenv('vision_endpoint')
    os.environ["VISION_KEY"] = os.getenv('vision_key')

    # Set the values of your computer vision endpoint and computer vision key
    # as environment variables:
    try:
        endpoint = os.environ["VISION_ENDPOINT"]
        key = os.environ["VISION_KEY"]
    except KeyError:
        print("Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
        print("Set them before running this sample.")
        exit()

    # Create an Image Analysis client
    client = ImageAnalysisClient(
        endpoint = endpoint,
        credential=AzureKeyCredential(key)
    )

    # Analysiere alle visuellen Merkmale des Bildes
    result = client.analyze(
        image_data=image_data,
        visual_features=[
            VisualFeatures.CAPTION,
            VisualFeatures.DENSE_CAPTIONS,
        ],
        smart_crops_aspect_ratios=[0.9, 1.33],  # Optional
        gender_neutral_caption=False,  # Optional
        language="en",  # Optional
    )

    def get_object_position(x, y, width, height, image_width, image_height):
        """
        Calculate the object's position within the image based on the bounding box parameters.

        Parameters:
        - x (float): The x-coordinate of the top-left corner of the bounding box.
        - y (float): The y-coordinate of the top-left corner of the bounding box.
        - width (float): The width of the bounding box.
        - height (float): The height of the bounding box.
        - image_width (float): The width of the image.
        - image_height (float): The height of the image.

        Returns:
        - str: A string describing the object's position (e.g., 'top left', 'center', 'bottom right').
        """

        # Calculate the center point of the bounding box
        x_center = x + (width / 2)
        y_center = y + (height / 2)

        # Normalize the center point coordinates
        x_norm = x_center / image_width
        y_norm = y_center / image_height

        # Determine horizontal position
        if x_norm < 0.33:
            horizontal_position = 'left'
        elif x_norm <= 0.66:
            horizontal_position = 'center'
        else:
            horizontal_position = 'right'

        # Determine vertical position
        if y_norm < 0.33:
            vertical_position = 'furthest'
        elif y_norm <= 0.66:
            vertical_position = 'middle'
        else:
            vertical_position = 'nearest'

        # Combine positions
        if vertical_position == 'middle' and horizontal_position == 'center':
            position = 'center'
        else:
            position = f"{horizontal_position}"

        return position

    # Print all analysis results to the console

    prompt = "Describe the content of this image shortly for a visually impaired person. Start with an overview sentence that serves as the main caption. The picture depicts one coherent situation without any close-ups. DATA:"

    if result.caption is not None:
        prompt += f"Image caption (Main caption): '{result.caption.text}' "# with confidence: {result.caption.confidence:.4f}"

    if result.dense_captions is not None:
        prompt += "Detailed descriptions (Dense Captions):"
        for caption in result.dense_captions.list:
            prompt += f"'{caption.text}' located at '{get_object_position(caption.bounding_box.x, caption.bounding_box.y, caption.bounding_box.width, caption.bounding_box.height, result.metadata.width, result.metadata.height)}' "# with confidence: {caption.confidence:.4f}"

    print(prompt)

    # SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    # SPDX-License-Identifier: MIT

    def create_channel(
        ssl_cert: Optional[Union[str, os.PathLike]] = None, use_ssl: bool = False, uri: str = "localhost:50051", metadata: Optional[List[Tuple[str, str]]] = None,
    ) -> grpc.Channel:

        def metadata_callback(context, callback):
            callback(metadata, None)

        if ssl_cert is not None or use_ssl:
            root_certificates = None
            if ssl_cert is not None:
                ssl_cert = Path(ssl_cert).expanduser()
                with open(ssl_cert, 'rb') as f:
                    root_certificates = f.read()
            creds = grpc.ssl_channel_credentials(root_certificates)
            if metadata:
                auth_creds = grpc.metadata_call_credentials(metadata_callback)
                creds = grpc.composite_channel_credentials(creds, auth_creds)
            channel = grpc.secure_channel(uri, creds)
        else:
            channel = grpc.insecure_channel(uri)
        return channel


    class Auth:
        def __init__(
            self,
            ssl_cert: Optional[Union[str, os.PathLike]] = None,
            use_ssl: bool = False,
            uri: str = "localhost:50051",
            metadata_args: List[List[str]] = None,
        ) -> None:
            """
            A class responsible for establishing connection with a server and providing security metadata.

            Args:
                ssl_cert (:obj:`Union[str, os.PathLike]`, `optional`): a path to SSL certificate file. If :param:`use_ssl`
                    is :obj:`False` and :param:`ssl_cert` is not :obj:`None`, then SSL is used.
                use_ssl (:obj:`bool`, defaults to :obj:`False`): whether to use SSL. If :param:`ssl_cert` is :obj:`None`,
                    then SSL is still used but with default credentials.
                uri (:obj:`str`, defaults to :obj:`"localhost:50051"`): a Riva URI.
            """
            self.ssl_cert: Optional[Path] = None if ssl_cert is None else Path(ssl_cert).expanduser()
            self.uri: str = uri
            self.use_ssl: bool = use_ssl
            self.metadata = []
            if metadata_args:
                for meta in metadata_args:
                    if len(meta) != 2:
                        raise ValueError(f"Metadata should have 2 parameters in \"key\" \"value\" pair. Receieved {len(meta)} parameters.")
                    self.metadata.append(tuple(meta))
            self.channel: grpc.Channel = create_channel(self.ssl_cert, self.use_ssl, self.uri, self.metadata)

        def get_auth_metadata(self) -> List[Tuple[str, str]]:
            """
            Will become useful when API key and OAUTH tokens will be enabled.

            Metadata for authorizing requests. Should be passed to stub methods.

            Returns:
                :obj:`List[Tuple[str, str]]`: an empty list.
            """
            metadata = []
            return metadata

    # SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    # SPDX-License-Identifier: MIT

    def add_custom_dictionary_to_config(req, custom_dictionary):
        result_list = [f"{key}  {value}" for key, value in custom_dictionary.items()]
        result_string = ','.join(result_list)
        req.custom_dictionary = result_string

    class SpeechSynthesisService:
        """
        A class for synthesizing speech from text. Provides :meth:`synthesize` which returns entire audio for a text
        and :meth:`synthesize_online` which returns audio in small chunks as it is becoming available.
        """
        def __init__(self, auth: Auth) -> None:
            """
            Initializes an instance of the class.

            Args:
                auth (:obj:`Auth`): an instance of :class:`riva.client.auth.Auth` which is used for authentication metadata
                    generation.
            """
            self.auth = auth
            self.stub = rtts_srv.RivaSpeechSynthesisStub(self.auth.channel)

        def synthesize(
            self,
            text: str,
            voice_name: Optional[str] = None,
            language_code: str = 'en-US',
            encoding: AudioEncoding = AudioEncoding.LINEAR_PCM,
            sample_rate_hz: int = 44100,
            audio_prompt_file: Optional[str] = None,
            audio_prompt_encoding: AudioEncoding = AudioEncoding.LINEAR_PCM,
            quality: int = 20,
            future: bool = False,
            custom_dictionary: Optional[dict] = None,
        ) -> Union[rtts.SynthesizeSpeechResponse, _MultiThreadedRendezvous]:
            """
            Synthesizes an entire audio for text :param:`text`.

            Args:
                text (:obj:`str`): An input text.
                voice_name (:obj:`str`, `optional`): A name of the voice, e.g. ``"English-US-Female-1"``. You may find
                    available voices in server logs or in server model directory. If this parameter is :obj:`None`, then
                    a server will select the first available model with correct :param:`language_code` value.
                language_code (:obj:`str`): a language to use.
                encoding (:obj:`AudioEncoding`): An output audio encoding, e.g. ``AudioEncoding.LINEAR_PCM``.
                sample_rate_hz (:obj:`int`): Number of frames per second in output audio.
                audio_prompt_file (:obj:`str`): An audio prompt file location for zero shot model.
                audio_prompt_encoding: (:obj:`AudioEncoding`): Encoding of audio prompt file, e.g. ``AudioEncoding.LINEAR_PCM``.
                quality: (:obj:`int`): This defines the number of times decoder is run. Higher number improves quality of generated
                                    audio but also takes longer to generate the audio. Ranges between 1-40.
                future (:obj:`bool`, defaults to :obj:`False`): Whether to return an async result instead of usual
                    response. You can get a response by calling ``result()`` method of the future object.
                custom_dictionary (:obj:`dict`, `optional`): Dictionary with key-value pair containing grapheme and corresponding phoneme

            Returns:
                :obj:`Union[riva.client.proto.riva_tts_pb2.SynthesizeSpeechResponse, grpc._channel._MultiThreadedRendezvous]`:
                a response with output. You may find :class:`riva.client.proto.riva_tts_pb2.SynthesizeSpeechResponse` fields
                description `here
                <https://docs.nvidia.com/deeplearning/riva/user-guide/docs/reference/protos/protos.html#riva-proto-riva-tts-proto>`_.
            """
            req = rtts.SynthesizeSpeechRequest(
                text=text,
                language_code=language_code,
                sample_rate_hz=sample_rate_hz,
                encoding=encoding,
            )
            if voice_name is not None:
                req.voice_name = voice_name
            if audio_prompt_file is not None:
                with wave.open(str(audio_prompt_file), 'rb') as wf:
                    rate = wf.getframerate()
                    req.zero_shot_data.sample_rate_hz = rate
                with audio_prompt_file.open('rb') as wav_f:
                    audio_data = wav_f.read()
                    req.zero_shot_data.audio_prompt = audio_data
                req.zero_shot_data.encoding = audio_prompt_encoding
                req.zero_shot_data.quality = quality

            add_custom_dictionary_to_config(req, custom_dictionary)

            func = self.stub.Synthesize.future if future else self.stub.Synthesize
            return func(req, metadata=self.auth.get_auth_metadata())

        def synthesize_online(
            self,
            text: str,
            voice_name: Optional[str] = None,
            language_code: str = 'en-US',
            encoding: AudioEncoding = AudioEncoding.LINEAR_PCM,
            sample_rate_hz: int = 44100,
            audio_prompt_file: Optional[str] = None,
            audio_prompt_encoding: AudioEncoding = AudioEncoding.LINEAR_PCM,
            quality: int = 20,
            custom_dictionary: Optional[dict] = None,
        ) -> Generator[rtts.SynthesizeSpeechResponse, None, None]:
            """
            Synthesizes and yields output audio chunks for text :param:`text` as the chunks
            becoming available.

            Args:
                text (:obj:`str`): An input text.
                voice_name (:obj:`str`, `optional`): A name of the voice, e.g. ``"English-US-Female-1"``. You may find
                    available voices in server logs or in server model directory. If this parameter is :obj:`None`, then
                    a server will select the first available model with correct :param:`language_code` value.
                language_code (:obj:`str`): A language to use.
                encoding (:obj:`AudioEncoding`): An output audio encoding, e.g. ``AudioEncoding.LINEAR_PCM``.
                sample_rate_hz (:obj:`int`): Number of frames per second in output audio.
                audio_prompt_file (:obj:`str`): An audio prompt file location for zero shot model.
                audio_prompt_encoding: (:obj:`AudioEncoding`): Encoding of audio prompt file, e.g. ``AudioEncoding.LINEAR_PCM``.
                quality: (:obj:`int`): This defines the number of times decoder is run. Higher number improves quality of generated
                                    audio but also takes longer to generate the audio. Ranges between 1-40.
                custom_dictionary (:obj:`dict`, `optional`): Dictionary with key-value pair containing grapheme and corresponding phoneme

            Yields:
                :obj:`riva.client.proto.riva_tts_pb2.SynthesizeSpeechResponse`: a response with output. You may find
                :class:`riva.client.proto.riva_tts_pb2.SynthesizeSpeechResponse` fields description `here
                <https://docs.nvidia.com/deeplearning/riva/user-guide/docs/reference/protos/protos.html#riva-proto-riva-tts-proto>`_.
                If :param:`future` is :obj:`True`, then a future object is returned. You may retrieve a response from a
                future object by calling ``result()`` method.
            """
            req = rtts.SynthesizeSpeechRequest(
                text=text,
                language_code=language_code,
                sample_rate_hz=sample_rate_hz,
                encoding=encoding,
            )
            if voice_name is not None:
                req.voice_name = voice_name

            if audio_prompt_file is not None:
                with wave.open(str(audio_prompt_file), 'rb') as wf:
                    rate = wf.getframerate()
                    req.zero_shot_data.sample_rate_hz = rate
                with audio_prompt_file.open('rb') as wav_f:
                    audio_data = wav_f.read()
                    req.zero_shot_data.audio_prompt = audio_data
                req.zero_shot_data.encoding = audio_prompt_encoding
                req.zero_shot_data.quality = quality

            add_custom_dictionary_to_config(req, custom_dictionary)

            return self.stub.SynthesizeOnline(req, metadata=self.auth.get_auth_metadata())

    def tts(server, text, voice, language_code, output):
        output = Path(output)
        if output.is_dir():
            print("Empty output file path not allowed")
            return

        auth = Auth(None, None, server, None)
        service = riva.client.SpeechSynthesisService(auth)
        nchannels = 1
        sampwidth = 2
        sound_stream, out_f = None, None

        if not text:
            print("No input text provided")
            return

        try:

            if output is not None:
                out_f = wave.open(str(output), 'wb')
                out_f.setnchannels(nchannels)
                out_f.setsampwidth(sampwidth)
                out_f.setframerate(44100)

            custom_dictionary_input = {}

            print("Generating audio for request...")


            resp = service.synthesize(
                text, voice, language_code, sample_rate_hz=44100,
                audio_prompt_file=None, quality=20 ,
                custom_dictionary=custom_dictionary_input
            )


            if sound_stream is not None:
                sound_stream(resp.audio)
            if out_f is not None:
                out_f.writeframesraw(resp.audio)
        except Exception as e:
            print(e.details())
        finally:
            if out_f is not None:
                out_f.close()
            if sound_stream is not None:
                sound_stream.close()

    url = 'http://3.77.217.49:8000/v1/chat/completions'#LLM Verbindung

    transcribed_text = prompt

    print(transcribed_text)

    myobj = {"model": "meta/llama-3.1-8b-instruct",
            "messages": [{"role":"user", "content":transcribed_text}],
            "max_tokens": 250}

    x = requests.post(url, json = myobj) #LLM request gesendet

    json_object = json.loads(x.text) #können wir kürzen

    text_response = json_object["choices"][0]["message"]["content"]

    print(text_response)

    url_tts = '18.199.232.221:50051'#Verbindung Text to Speech

    tts(url_tts, text_response, "English-US.Female-1", "en-US", "./output.mp3")

    # Output audio

    st.audio("output.mp3", format="audio/mpeg", loop=True)
