import httpx
from fastapi import APIRouter, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from groq import GroqError
from pydantic import ValidationError

from src.app.api.routes.tasks import tasks
from src.app.schemas.voice import InstructionRequest, TranscribeFlowResponse
from src.app.services.llm import interpret_instruction, transcribe_audio
from src.app.utils.language import normalize_transcription_language

router = APIRouter(tags=["transcribe"])


@router.get("/")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


async def _read_transcription(request: Request) -> str:
    """Accept multipart audio (file + optional language) or JSON {"transcription": ...}."""
    if request.headers.get("content-type", "").startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Missing 'file' audio upload.")
        language = normalize_transcription_language(form.get("language"))
        content = await upload.read()
        text = await run_in_threadpool(
            transcribe_audio, upload.filename or "audio.webm", content, language
        )
        if not text:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Empty transcription.")
        return text

    try:
        return InstructionRequest.model_validate(await request.json()).transcription
    except (ValueError, ValidationError) as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Expected JSON body {'transcription': str} or multipart audio.",
        ) from exc


@router.post("/transcribe", response_model=TranscribeFlowResponse)
async def transcribe_and_run_flow(request: Request) -> TranscribeFlowResponse:
    try:
        transcription = await _read_transcription(request)
        instruction = await run_in_threadpool(interpret_instruction, transcription, tasks)
    except GroqError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=f"Groq API error: {exc}") from exc
    except ValidationError as exc:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, detail="The LLM returned an invalid instruction payload."
        ) from exc

    # Execute the LLM-chosen call against this same app, so routing stays 100% LLM-driven.
    transport = httpx.ASGITransport(app=request.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://internal") as client:
        response = await client.request(
            instruction.method, instruction.endpoint, json=instruction.params
        )

    result = response.json() if response.content else None
    if response.is_error:
        raise HTTPException(status_code=response.status_code, detail=result)

    return TranscribeFlowResponse(
        transcription=transcription, instruction=instruction, result=result
    )
