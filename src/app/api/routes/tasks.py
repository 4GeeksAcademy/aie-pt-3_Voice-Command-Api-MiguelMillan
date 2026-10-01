from itertools import count

from fastapi import APIRouter, HTTPException, status

from src.app.schemas.voice import Task, TaskCreate, TaskReplace, TaskUpdate

router = APIRouter(prefix="/tasks", tags=["tasks"])

tasks: list[dict] = []
_task_id_counter = count(start=1)


def _find_task(task_id: int) -> dict:
    task = next((t for t in tasks if t["id"] == task_id), None)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found",
        )
    return task


@router.get("", response_model=list[Task])
def get_tasks() -> list[dict]:
    return tasks


@router.post("", response_model=Task, status_code=status.HTTP_201_CREATED)
def create_task(payload: TaskCreate) -> dict:
    task = {"id": next(_task_id_counter), **payload.model_dump()}
    tasks.append(task)
    return task


@router.put("/{task_id}", response_model=Task)
def replace_task(task_id: int, payload: TaskReplace) -> dict:
    task = _find_task(task_id)
    task.update(payload.model_dump())
    return task


@router.patch("/{task_id}", response_model=Task)
def update_task(task_id: int, payload: TaskUpdate) -> dict:
    task = _find_task(task_id)
    task.update(payload.model_dump(exclude_unset=True, exclude_none=True))
    return task


@router.delete("/{task_id}")
def delete_task(task_id: int) -> dict[str, str]:
    tasks.remove(_find_task(task_id))
    return {"message": "Task deleted successfully"}
