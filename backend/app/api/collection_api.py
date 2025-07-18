from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Optional
from pydantic import BaseModel
import logging

from backend.app.services.vstore_svc import VectorStoreService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/collections", tags=["Collections"])

class CollectionResponse(BaseModel):
    name: str
    document_count: int

class CollectionCreateRequest(BaseModel):
    name: str

def get_vector_store_service():
    return VectorStoreService()

@router.post("", response_model=CollectionResponse)
async def create_collection(
    request: CollectionCreateRequest,
    vstore_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """Create a new collection."""
    logger.info(f"Received request to create collection: {request.name}")
    try:
        logger.info(f"Calling VectorStoreService.create_collection with name: {request.name}")
        vstore_svc.create_collection(request.name)
        logger.info(f"Successfully created collection: {request.name}")
        return CollectionResponse(
            name=request.name,
            document_count=0
        )
    except ValueError as e:
        logger.error(f"ValueError while creating collection: {str(e)}")
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error while creating collection: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to create collection: {str(e)}")
    
@router.get("", response_model=List[CollectionResponse])
async def list_collections(
    vstore_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """List all collections."""
    try:
        collections = vstore_svc.list_collections()
        return [
            CollectionResponse(
                name=name,
                document_count=count
            ) for name, count in collections
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")

@router.get("/get-list", response_model=List[CollectionResponse])
async def list_collections(
    collection_name: Optional[str] = None,
    vstore_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """List all collections."""
    try:
        collections = vstore_svc.list_collections_with_control(collection_name)
        return [
            CollectionResponse(
                name=name,
                document_count=count
            ) for name, count in collections
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")

@router.delete("/{name}")
async def delete_collection(
    user_name: str,
    collection_name: str,
    vstore_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """Delete a collection."""
    try:
        success = vstore_svc._delete_with_access_control(user_collection=user_name, collection_name=collection_name)
        if not success:
            raise HTTPException(status_code=403, detail=f"User '{user_name}' does not have access or collection not found.")
        return {"message": f"Collection '{collection_name}' deleted successfully by User: {user_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete collection: {str(e)}")

@router.post("/add_access")
async def add_control_acess(
    user_collection: Optional[str],
    client_collection: Optional[str],
    collection_name: Optional[str],
    vstore_svc: VectorStoreService = Depends(get_vector_store_service)
):
    try:
        success = vstore_svc._add_control_to_user(user_collection=user_collection,
                                                 client_collection=client_collection,
                                                 collection_name=collection_name)
        if not success:
            raise HTTPException(status_code=403, detail=f"Unable to add the access by user {user_collection} to client_collection {client_collection} for collection {collection_name}")
        return {
            "message": f"Access control given successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Getting error {e}")