from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from src.api.dependencies import get_pipeline
from src.api.file_validation import (
    MAX_FILES_PER_BATCH,
    compute_code_metrics,
    detect_language_from_filename,
    normalize_language,
    validate_code_text,
    validate_file_content,
    validate_file_name,
)
from src.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeResponse,
    FileAnalyzeResponse,
)
from src.pipeline.orchestrator import AnalysisPipeline

router = APIRouter(prefix="/analyze", tags=["analysis"])

@router.post("/", response_model=AnalyzeResponse)
def analyze_code(
    request: AnalyzeRequest,
    pipeline: AnalysisPipeline = Depends(get_pipeline),
):

    language, language_error = normalize_language(request.language)
    if language_error:
        raise HTTPException(status_code=400, detail=language_error)

    metrics, validation_error = validate_code_text(request.code)
    if validation_error:
        raise HTTPException(status_code=400, detail=validation_error)

    result = pipeline.run(
        code=request.code,
        language=language,
        input_metrics=metrics,
    )

    return AnalyzeResponse(**result)


@router.post("/file", response_model=FileAnalyzeResponse)
async def analyze_file(
    file: UploadFile = File(...),
    pipeline: AnalysisPipeline = Depends(get_pipeline),
):
    filename = file.filename or "uploaded_file"

    file_error = validate_file_name(filename)
    if file_error:
        raise HTTPException(status_code=400, detail=file_error)

    content = await file.read()
    decoded, content_error = validate_file_content(content)
    if content_error:
        raise HTTPException(status_code=400, detail=content_error)

    language = detect_language_from_filename(filename)
    language, language_error = normalize_language(language)
    if language_error:
        raise HTTPException(status_code=400, detail=language_error)

    metrics, validation_error = validate_code_text(decoded)
    if validation_error:
        raise HTTPException(status_code=400, detail=validation_error)

    try:
        result = pipeline.run(code=decoded, language=language, input_metrics=metrics)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to analyze file: {exc}") from exc

    return FileAnalyzeResponse(
        filename=filename,
        language=language,
        plagiarism_percentage=result["plagiarism_percentage"],
        ai_probability=result["ai_probability"],
        confidence=result["confidence"],
        explanation=result["explanation"],
    )


@router.post("/files", response_model=BatchAnalyzeResponse)
async def analyze_files(
    files: list[UploadFile] = File(...),
    pipeline: AnalysisPipeline = Depends(get_pipeline),
):
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="At least one file is required")

    if len(files) > MAX_FILES_PER_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Max allowed per request is {MAX_FILES_PER_BATCH}",
        )

    results: list[FileAnalyzeResponse] = []
    errors: dict[str, str] = {}

    for item in files:
        filename = item.filename or "uploaded_file"

        file_error = validate_file_name(filename)
        if file_error:
            errors[filename] = file_error
            continue

        content = await item.read()
        decoded, content_error = validate_file_content(content)
        if content_error:
            errors[filename] = content_error
            continue

        language = detect_language_from_filename(filename)
        language, language_error = normalize_language(language)
        if language_error:
            errors[filename] = language_error
            continue

        metrics, validation_error = validate_code_text(decoded)
        if validation_error:
            errors[filename] = validation_error
            continue

        try:
            result = pipeline.run(code=decoded, language=language, input_metrics=metrics)
        except Exception as exc:
            errors[filename] = f"Failed to analyze file: {exc}"
            continue

        results.append(
            FileAnalyzeResponse(
                filename=filename,
                language=language,
                plagiarism_percentage=result["plagiarism_percentage"],
                ai_probability=result["ai_probability"],
                confidence=result["confidence"],
                explanation=result["explanation"],
            )
        )

    return BatchAnalyzeResponse(
        total_files=len(files),
        succeeded=len(results),
        failed=len(errors),
        results=results,
        errors=errors,
    )
