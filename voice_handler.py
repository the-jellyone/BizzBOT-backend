import io
import wave
import os
from groq import Groq
from agent import get_bizz_bot_response, llm
from google import genai
from google.genai import types
from langchain_core.messages import HumanMessage

groq_client   = Groq(api_key=os.getenv("GROQ_API_KEY"))
google_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def _summarise_for_voice(visual_text: str) -> str:
    """
    Uses the same Groq LLM from agent.py to compress the full response
    into 1-2 short spoken sentences — no extra API quota needed.
    """
    response = llm.invoke([
        HumanMessage(content=(
            "Summarise the following business response into 1-2 short natural spoken sentences. "
            "No bullet points, no markdown, no lists. "
            "Just a clean verbal takeaway a human would say out loud.\n\n"
            f"{visual_text}"
        ))
    ])
    summary = response.content.strip()
    if len(summary) > 400:
        summary = summary[:400].rsplit(".", 1)[0] + "."
    return summary


async def handle_voice_conversation(
    audio_bytes: bytes,
    user_id: str,
    chat_id: str,
) -> tuple[bytes, str, str]:
    """
    Full voice pipeline. Returns a 3-tuple:
        (wav_audio_bytes, visual_text, user_transcript)

    - wav_audio_bytes : TTS audio of the Gemini-summarised voice script
    - visual_text     : full structured response shown in the chat UI
    - user_transcript : what the user said (for the frontend user bubble)
    """

    #  transcribe the incoming audio
    transcription = groq_client.audio.transcriptions.create(
        file=("input.wav", audio_bytes),
        model="whisper-large-v3",
    )
    user_text = transcription.text.strip()

    # BRAIN: run the agent
    processed = await get_bizz_bot_response(user_id, chat_id, user_text)

    visual_report = processed.get("visual", "Your strategy is ready on screen.")

    # SUMMARISE: compress to 1-2 spoken sentences using Groq LLM(in an attempt to save tts tokens)
    voice_script = _summarise_for_voice(visual_report)

    #convert summarised script to spoken audio
    tts_response = google_client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=voice_script,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Charon"
                    )
                )
            ),
        ),
    )

    # WRAP: pack raw PCM into a proper WAV container
    raw_pcm = tts_response.candidates[0].content.parts[0].inline_data.data
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(raw_pcm)

    return wav_buffer.getvalue(), visual_report, user_text