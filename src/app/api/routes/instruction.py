from fastapi import APIRouter, HTTPException, status
from groq import GroqError
from pydantic import ValidationError

from src.app.api.routes.tasks import tasks
from src.app.schemas.voice import InstructionPayload, InstructionRequest
from src.app.services.llm import interpret_instruction

router = APIRouter(tags=["instruction"])


@router.post("/instruction", response_model=InstructionPayload)
def route_instruction(
    payload: InstructionRequest,
) -> InstructionPayload:
    try:
        return interpret_instruction(payload.transcription, tasks)
    except GroqError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Groq API error: {exc}",
        ) from exc
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The LLM returned an invalid instruction payload.",
        ) from exc
