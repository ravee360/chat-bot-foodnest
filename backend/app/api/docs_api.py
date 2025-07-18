# backend/app/api/docs_api.py

import os
import shutil
from pathlib import Path
from typing import List, Union, Optional 

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks # Removed Form

# Import schemas
from backend.app.models.schemas import DocumentProcessResponse, ProcessingMode 
# REMOVED: ProcessingMode from schemas import as it's no longer a choice from frontend
from pydantic import BaseModel, HttpUrl

# Import services and settings
from backend.app.core.config import settings
# REMOVED: from backend.app.services.doc_parser import DocParserService 
from backend.app.services.doc_parser_fast import DocParserFastService, process_url_background # ONLY Fast rule-based parser
from backend.app.services.vstore_svc import VectorStoreService 
from backend.app.core.utils import get_next_document_id


# --- Dependency to get service instances ---
_vector_store_service_instance = None
# REMOVED: _doc_parser_llm_instance = None
_doc_parser_fast_instance = None # This will be the only parser used








def get_vector_store_service():
    global _vector_store_service_instance
    if _vector_store_service_instance is None:
        _vector_store_service_instance = VectorStoreService()
    return _vector_store_service_instance

# REMOVED: get_doc_parser_llm_service

def get_doc_parser_fast_service(vstore_svc: VectorStoreService = Depends(get_vector_store_service)):
    global _doc_parser_fast_instance
    if _doc_parser_fast_instance is None:
        _doc_parser_fast_instance = DocParserFastService(vector_store_service=vstore_svc)
    return _doc_parser_fast_instance

router = APIRouter(prefix="/documents", tags=["Documents"])
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

# REMOVED: AnyDocParserService type alias as only one parser is used

def process_document_background(
    file_path: str, 
    source_doc_id: str, 
    doc_parser_svc_instance: DocParserFastService, # MODIFIED: Specific to DocParserFastService
    serial_no: Optional[int] = None, 
    total_count: Optional[int] = None,
    collection_name: Optional[str] = None
):
    parser_name = doc_parser_svc_instance.__class__.__name__ # Will be DocParserFastService
    progress_log = f"Document {serial_no}/{total_count}" if serial_no and total_count else "Single document"
    
    print(f"Background task started for: {Path(file_path).name}, Source ID: {source_doc_id}, Parser: {parser_name}, ({progress_log})")
    try:
        success = doc_parser_svc_instance.process_document(
            file_path=file_path, 
            source_doc_id=source_doc_id,
            collection_name=collection_name
        )
        if success:
            print(f"Background processing completed successfully for {source_doc_id} ({Path(file_path).name}) using {parser_name}")
        else:
            print(f"Background processing (using {parser_name}) had issues or no chunks generated for {source_doc_id} ({Path(file_path).name})")
    except Exception as e:
        print(f"Error during background document processing for {source_doc_id} ({Path(file_path).name}) (using {parser_name}): {e}")

class URLProcessRequest(BaseModel):
    url: HttpUrl
    collection: Optional[str] = None

@router.post("/process-url", response_model=DocumentProcessResponse, status_code=202)
async def process_url(
    request: URLProcessRequest,
    background_tasks: BackgroundTasks,
    fast_parser: DocParserFastService = Depends(get_doc_parser_fast_service),
    vstore_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """
    Processes a document from a URL.
    """
    source_doc_id = get_next_document_id()
    
    # Validate collection if provided
    if request.collection:
        try:
            vstore_svc.get_collection(request.collection)
        except Exception as e:
            if "does not exist" in str(e).lower():
                raise HTTPException(status_code=404, detail=f"Collection '{request.collection}' not found")
            raise HTTPException(status_code=500, detail=f"Error validating collection: {str(e)}")

    background_tasks.add_task(
        process_url_background,
        url=str(request.url),
        source_doc_id=source_doc_id,
        doc_parser_svc_instance=fast_parser,
        collection_name=request.collection
    )
    
    return DocumentProcessResponse(
        message="URL received and queued for background processing.",
        file_name=str(request.url),
        source_doc_id=source_doc_id,
        status="queued_for_processing",
        processing_mode_used=ProcessingMode.FAST_RULE_BASED
    )

@router.post("/upload", response_model=DocumentProcessResponse, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="The document file to upload (PDF or image)."),
    collection: Optional[str] = None,  # Add collection parameter
    fast_parser: DocParserFastService = Depends(get_doc_parser_fast_service),
    vstore_svc: VectorStoreService = Depends(get_vector_store_service)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided.")

    # Validate collection if provided
    if collection:
        try:
            vstore_svc.get_collection(collection)
        except Exception as e:
            if "does not exists" in str(e).lower():
                raise HTTPException(status_code=404, detail=f"Collection '{collection}' not found")
            raise HTTPException(status_code=500, detail=f"Error validating collection: {str(e)}")

    source_doc_id = get_next_document_id()
    safe_original_filename = "".join(c if c.isalnum() or c in ['.', '_'] else '_' for c in file.filename)
    temp_file_path = Path(settings.UPLOAD_DIR) / f"{source_doc_id}_{safe_original_filename}"

    # No need to choose parser, always use fast_parser
    chosen_parser = fast_parser

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"File '{file.filename}' saved temporarily to '{temp_file_path}' for fast rule-based processing. Assigned Source ID: {source_doc_id}")
    except Exception as e:
        if temp_file_path.exists():
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Could not save uploaded file: {e}")
    finally:
        file.file.close()

    background_tasks.add_task(
        process_document_background, 
        str(temp_file_path), 
        source_doc_id, 
        chosen_parser,
        serial_no=1, 
        total_count=1,
        collection_name=collection
    )

    return DocumentProcessResponse(
        message="File uploaded successfully and queued for fast rule-based background processing.",
        file_name=file.filename,
        source_doc_id=source_doc_id,
        status="queued_for_processing",
        processing_mode_used=ProcessingMode.FAST_RULE_BASED
    )

@router.post("/upload-multiple", response_model=List[DocumentProcessResponse], status_code=202)
async def upload_multiple_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="A list of document files to upload."),
    collection: Optional[str] = None,  # Add collection parameter
    fast_parser: DocParserFastService = Depends(get_doc_parser_fast_service),
    vstore_svc: VectorStoreService = Depends(get_vector_store_service)
):
    responses = []
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    # Validate collection if provided
    if collection:
        try:
            vstore_svc.get_collection(collection)
        except Exception as e:
            if "does not exists" in str(e).lower():
                raise HTTPException(status_code=404, detail=f"Collection '{collection}' not found")
            raise HTTPException(status_code=500, detail=f"Error validating collection: {str(e)}")

    # No need to choose parser, always use fast_parser
    chosen_parser = fast_parser

    total_files_in_batch = len(files)
    for idx, file_upload_item in enumerate(files, start=1):
        if not file_upload_item.filename:
            responses.append(DocumentProcessResponse(
                message="File skipped: No filename.",
                file_name="Unknown",
                status="failed_upload",
                processing_mode_used=ProcessingMode.FAST_RULE_BASED
            ))
            continue

        source_doc_id = get_next_document_id()
        safe_original_filename = "".join(c if c.isalnum() or c in ['.', '_'] else '_' for c in file_upload_item.filename)
        temp_file_path = Path(settings.UPLOAD_DIR) / f"{source_doc_id}_{safe_original_filename}"

        try:
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file_upload_item.file, buffer)
            print(f"File '{file_upload_item.filename}' saved to '{temp_file_path}' for fast rule-based processing. Assigned Source ID: {source_doc_id} ({idx}/{total_files_in_batch})")
            
            # Add collection to the background task
            background_tasks.add_task(
                process_document_background,
                str(temp_file_path),
                source_doc_id,
                chosen_parser,
                serial_no=idx,
                total_count=total_files_in_batch,
                collection_name=collection  # Pass collection name to background task
            )
            responses.append(DocumentProcessResponse(
                message="File uploaded successfully and queued for fast rule-based background processing.",
                file_name=file_upload_item.filename,
                source_doc_id=source_doc_id,
                status="queued_for_processing",
                processing_mode_used=ProcessingMode.FAST_RULE_BASED
            ))
        except Exception as e:
            if temp_file_path.exists():
                os.remove(temp_file_path)
            responses.append(DocumentProcessResponse(
                message=f"Failed to save or queue file '{file_upload_item.filename}': {e}",
                file_name=file_upload_item.filename,
                source_doc_id=source_doc_id,
                status="failed_to_queue",
                processing_mode_used=ProcessingMode.FAST_RULE_BASED
            ))
        finally:
            if file_upload_item and hasattr(file_upload_item, 'file') and file_upload_item.file and not file_upload_item.file.closed:
                 try:
                    file_upload_item.file.close()
                 except Exception as e_close:
                    print(f"Error closing file {file_upload_item.filename if file_upload_item else 'N/A'}: {e_close}")
            
    return responses

@router.get("/extract-text")
async def extract_text_from_all_uploaded_documents(
    fast_parser: DocParserFastService = Depends(get_doc_parser_fast_service)
):
    """
    Returns the extracted text for all files currently in the uploads directory.
    No input required.
    """
    uploads_dir = Path(settings.UPLOAD_DIR)
    results = []
    for file_path in uploads_dir.iterdir():
        if file_path.is_file():
            try:
                extracted_text = fast_parser.extract_text_only(str(file_path))
                results.append({
                    "file_name": file_path.name,
                    "extracted_text": extracted_text
                })
            except Exception as e:
                results.append({
                    "file_name": file_path.name,
                    "extracted_text": f"Error extracting text: {e}"
                })
    return results


if __name__ == "__main__":
    print("docs_api.py can be tested by running the main FastAPI application.")
