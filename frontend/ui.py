# frontend/ui.py

import streamlit as st
import requests
import os
from typing import List, Dict, Any, Optional
import time 
import json

# --- Configuration ---
BACKEND_URL = os.environ.get("BACKEND_API_URL", "http://localhost:8000")
# The backend will now always use the fast parser.
# The `processing_mode` parameter will be removed from the API call.
DOCUMENTS_UPLOAD_URL = f"{BACKEND_URL}/api/v1/documents/upload-multiple"
URL_PROCESS_URL = f"{BACKEND_URL}/api/v1/documents/process-url"
CHAT_QUERY_URL = f"{BACKEND_URL}/api/v1/chat/query"
COLLECTIONS_URL = f"{BACKEND_URL}/api/v1/collections"

# --- Helper Functions ---
def get_collections(user_id: Optional[str] = None, only_accessible: bool = False):
    """Get list of collections from the API. If only_accessible is True, get only those the user has access to."""
    try:
        if only_accessible and user_id:
            response = requests.get(f"{COLLECTIONS_URL}/get-list", params={"collection_name": user_id})
        else:
            response = requests.get(COLLECTIONS_URL)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching collections: {str(e)}")
        return []

def create_collection(name: str) -> bool:
    """Create a new collection."""
    try:
        print(f"Attempting to create collection with name: {name}")
        response = requests.post(
            COLLECTIONS_URL,
            json={"name": name},
            headers={"Content-Type": "application/json"}
        )
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.text}")
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Request error details: {str(e)}")
        if hasattr(e.response, 'text'):
            print(f"Error response content: {e.response.text}")
        st.error(f"Error creating collection: {str(e)}")
        return False

def delete_collection(name: str) -> bool:
    """Delete a collection."""
    try:
        response = requests.delete(f"{COLLECTIONS_URL}/{name}")
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Error deleting collection: {str(e)}")
        return False

def reset_chat_history():
    """Resets the chat history in the session state."""
    st.session_state.messages = []
    st.session_state.processed_doc_ids = set() 
    st.session_state.user_has_been_warned_about_processing = False
    if "processing_active_message_placeholder" in st.session_state and st.session_state.processing_active_message_placeholder is not None:
        st.session_state.processing_active_message_placeholder.empty()
        st.session_state.processing_active_message_placeholder = None

def query_chat(query: str, collection: Optional[str] = None) -> Dict[str, Any]:
    """Send a query to the chat API."""
    try:
        response = requests.post(
            CHAT_QUERY_URL,
            json={
                "query": query,
                "collection": collection,
                "n_results": 5
            }
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error communicating with the backend: {str(e)}")
        raise

def display_chat_message(role: str, content: Any):
    """Helper to display a chat message with a consistent avatar."""
    avatar_map = {"user": "👤", "assistant": "🤖"}
    avatar = avatar_map.get(role)
    with st.chat_message(role, avatar=avatar):
        if isinstance(content, str): 
            st.markdown(content)
        elif isinstance(content, dict): 
            # 1. Display the main answer
            st.markdown(f"**Answer:**\n{content.get('answer', 'No answer provided.')}")
            
            # 2. Display themes with evidence
            themes = content.get('themes', [])
            if themes:
                st.markdown("\n**Identified Themes:**")
                for i, theme_data in enumerate(themes):
                    theme_summary = theme_data.get('theme_summary', 'N/A')
                    supporting_refs = theme_data.get('supporting_reference_numbers', []) 
                    evidence_snippets = theme_data.get('evidence_snippets', [])
                    
                    # Display theme summary with references
                    if supporting_refs and all(isinstance(ref, int) for ref in supporting_refs):
                        refs_str = ", ".join([f"[{ref_num}]" for ref_num in supporting_refs])
                        st.markdown(f"  - **{i+1}. {theme_summary}** (Supported by Refs: {refs_str})")
                    else:
                        st.markdown(f"  - **{i+1}. {theme_summary}**")
                    
                    # Display evidence in collapsible section
                    if evidence_snippets:
                        with st.expander(f"Show Evidence for Theme {i+1}", expanded=False):
                            for evidence in evidence_snippets:
                                st.markdown(f"**Text:**\n_{evidence.get('text', 'No text available')}_")
                                st.markdown(f"**Location:** Doc ID: `{evidence.get('source_doc_id', 'N/A')}`, Page {evidence.get('page', 'N/A')}, Paragraph {evidence.get('paragraph', 'N/A')}")
                                st.markdown("---")
            else:
                st.markdown("\n*No specific themes were identified for this query.*")

            # 3. Display references
            references = content.get('references', [])
            if references:
                st.markdown("\n**References:**")
                sorted_references = sorted(references, key=lambda x: x.get('reference_number', 0))
                for ref_data in sorted_references:
                    ref_num = ref_data.get('reference_number', 'N/A')
                    file_name = ref_data.get('file_name', 'Unknown File')
                    source_id = ref_data.get('source_doc_id', 'Unknown Source ID')
                    st.markdown(f"  - **[{ref_num}]** {file_name} (Source ID: `{source_id}`)")

            # 4. Display LLM thought process
            synthesized_expert_answer = content.get('synthesized_expert_answer', None)
            if synthesized_expert_answer:
                with st.expander("Show LLM Thought Process", expanded=False):
                    st.markdown("### LLM Analysis and Synthesis")
                    st.markdown(synthesized_expert_answer)

            # 5. Display document details in a tabular format
            document_details = content.get('document_details', [])
            if document_details:
                with st.expander("Show Document Details", expanded=False):
                    st.markdown("### Document Details")
                    for doc in document_details:
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col1:
                            st.markdown(f"**Doc ID:**\n`{doc.get('source_doc_id', 'N/A')}`")
                        with col2:
                            st.markdown(f"**Content:**\n_{doc.get('extracted_answer', 'No content available')}_")
                        with col3:
                            st.markdown(f"**Citation:**\nPage: {doc.get('page_number', 'N/A')}\nPara: {doc.get('paragraph_number', 'N/A')}")
                        st.markdown("---")

            # Debug info
            retrieved_ids = content.get('retrieved_context_document_ids', [])
            if retrieved_ids:
                st.markdown(f"\n_(Debug: Context drawn from document IDs: {', '.join(retrieved_ids)})_")

            # 6. Display SVG visualization if present
            svg_code = content.get('svg')
            if svg_code and svg_code.strip() and svg_code.lower() != "none":
                st.markdown("**Visualization (SVG):**")
                st.components.v1.html(svg_code, height=400, scrolling=True)
        else: 
            st.markdown(str(content))

def collection_exists(collection_name: str) -> bool:
    """Check if a collection exists in the system (all collections, not just accessible)."""
    try:
        all_collections = get_collections()
        return any(col["name"] == collection_name for col in all_collections)
    except Exception:
        return False

def main():
    # --- Streamlit App ---
    st.set_page_config(page_title="DocBot - Document Research & Theme ID", layout="wide")

    st.title("📄 DocBot: Document Research & Theme Identifier")
    st.caption("Upload your documents (PDFs, images). Documents will be processed using fast, rule-based chunking.")

    # --- Initialize session state ---
    default_session_state = {
        "messages": [],
        "processed_doc_ids": set(),
        "uploaded_file_details": [],
        "user_has_been_warned_about_processing": False,
        "processing_active_message_placeholder": None,
        "selected_collection": None,
        "user_id": ""
    }
    for key, default_value in default_session_state.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # --- Sidebar for Collection and Document Management ---
    with st.sidebar:
        st.header("📚 Collection & Document Management")
        
        # User and Collection Management
        st.subheader("User Login")
        user_id_input = st.text_input(
            "Enter Unique User ID to Load/Create Collection",
            value=st.session_state.user_id,
            key="user_id_input_key"
        )

        if st.button("Set User and Collection", use_container_width=True):
            if user_id_input:
                st.session_state.user_id = user_id_input
                with st.spinner(f"Loading collection for '{user_id_input}'..."):
                    collections = get_collections()
                    collection_names = [col["name"] for col in collections]
                    if user_id_input not in collection_names:
                        st.info(f"User collection '{user_id_input}' not found. Creating it...")
                        if create_collection(user_id_input):
                            st.success(f"Collection '{user_id_input}' created and selected!")
                            st.session_state.selected_collection = user_id_input
                            st.rerun()
                        else:
                            st.error(f"Failed to create collection for user '{user_id_input}'.")
                    else:
                        st.session_state.selected_collection = user_id_input
                        st.success(f"Switched to collection for user '{user_id_input}'.")
                        st.rerun()
            else:
                st.warning("Please enter a User ID.")

        if st.session_state.selected_collection:
            st.success(f"Active Collection: `{st.session_state.selected_collection}`")
            # Add a button to switch user
            if st.button("Switch User/Collection", use_container_width=True):
                # Reset relevant session state
                reset_chat_history()
                st.session_state.uploaded_file_details = []
                st.session_state.processing_info_message = None 
                st.session_state.selected_collection = None
                st.session_state.user_id = ""
                st.rerun()

            # --- New: Delete Collection Option ---
            with st.expander("Danger Zone: Delete Collection", expanded=False):
                st.warning("Deleting a collection is irreversible. All documents and access will be lost.")
                # Fetch all collections the user has access to
                collections = get_collections(st.session_state.user_id, only_accessible=True)
                user_collections = [col["name"] for col in collections]
                if not user_collections:
                    st.info("You do not have any collections you can delete.")
                else:
                    # Initialize session state for deletion confirmation if not already set
                    if "delete_confirmation" not in st.session_state:
                        st.session_state.delete_confirmation = False

                    # Select collection to delete
                    collection_to_delete = st.selectbox("Select a collection to delete:", user_collections, key="delete_collection_select")

                    # Delete button
                    if st.button(f"Delete Collection '{collection_to_delete}'", type="primary", use_container_width=True):
                        if not collection_exists(collection_to_delete):
                            st.error(f"Collection '{collection_to_delete}' does not exist.")
                        else:
                            # Set confirmation state to True to trigger confirmation prompt
                            st.session_state.delete_confirmation = True

                    # Conditional rendering based on confirmation state
                    if st.session_state.delete_confirmation:
                        st.warning(f"Are you sure you want to delete '{collection_to_delete}'?")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("Yes, delete", type="primary", use_container_width=True):
                                try:
                                    response = requests.delete(
                                        f"{COLLECTIONS_URL}/{collection_to_delete}",
                                        params={
                                            "user_name": st.session_state.user_id,
                                            "collection_name": collection_to_delete
                                        },
                                        timeout=30
                                    )
                                    if response.status_code == 200:
                                        st.success(f"Collection '{collection_to_delete}' deleted successfully.")
                                        
                                        # Handle cleanup if needed
                                        if collection_to_delete == st.session_state.selected_collection:
                                            reset_chat_history()
                                            st.session_state.uploaded_file_details = []
                                            st.session_state.processing_info_message = None 
                                            st.session_state.selected_collection = None
                                            st.session_state.user_id = ""
                                        
                                        # Reset confirmation and rerun
                                        st.session_state.delete_confirmation = False
                                        st.rerun()
                                    else:
                                        st.error(f"Failed to delete collection: {response.text}")
                                        st.session_state.delete_confirmation = False
                                except Exception as e:
                                    st.error(f"Error deleting collection: {e}")
                                    st.session_state.delete_confirmation = False
                        
                        with col2:
                            if st.button("Cancel", use_container_width=True):
                                st.session_state.delete_confirmation = False
                                st.rerun()

            # --- New: Grant Access Option ---
            with st.expander("Grant Access to Another User", expanded=False):
                st.info("You can grant another user access to one of your collections.")
                # Fetch all collections the user has access to
                collections = get_collections(st.session_state.user_id, only_accessible=True)
                user_collections = [col["name"] for col in collections]
                if not user_collections:
                    st.info("You do not have any collections to grant access for.")
                else:
                    target_user = st.text_input("Enter the User ID to grant access:", key="grant_access_user_id")
                    collection_to_grant = st.selectbox("Select a collection to grant access to:", user_collections, key="grant_access_collection_select")
                    if st.button("Grant Access", use_container_width=True, key="grant_access_btn"):
                        if not collection_exists(collection_to_grant):
                            st.error(f"Collection '{collection_to_grant}' does not exist.")
                        elif not target_user or target_user == st.session_state.user_id:
                            st.warning("Please enter a valid User ID different from your own.")
                        else:
                            try:
                                response = requests.post(
                                    f"{COLLECTIONS_URL}/add_access",
                                    params={
                                        "user_collection": st.session_state.user_id,
                                        "client_collection": target_user,
                                        "collection_name": collection_to_grant
                                    },
                                    timeout=30
                                )
                                if response.status_code == 200:
                                    st.success(f"Access granted to user '{target_user}' for collection '{collection_to_grant}'.")
                                else:
                                    st.error(f"Failed to grant access: {response.text}")
                            except Exception as e:
                                st.error(f"Error granting access: {e}")

        else:
            st.info("Enter a User ID and click 'Set User' to begin.")

        st.markdown("---")
        
        # Conditionally show upload options only when a collection is selected
        if st.session_state.selected_collection:
            # Document Upload Section
            st.subheader("Document Upload")
            st.info("Using fast, rule-based document chunking.")

            # URL Processing Section
            st.subheader("Process from URL")
            url_input = st.text_input("Enter a URL to process")
            if st.button("Process URL", use_container_width=True):
                if not url_input:
                    st.warning("Please enter a URL")
                else:
                    try:
                        with st.spinner(f"Sending URL for processing..."):
                            response = requests.post(
                                URL_PROCESS_URL,
                                json={"url": url_input, "collection": st.session_state.selected_collection},
                                timeout=60
                            )
                            response.raise_for_status()
                            result = response.json()
                            st.success(f"URL '{result.get('file_name')}' (ID: {result.get('source_doc_id')}) queued for processing.")
                            # Add to uploaded documents log
                            existing_names = {f['name'] for f in st.session_state.uploaded_file_details}
                            if result['file_name'] not in existing_names:
                                st.session_state.uploaded_file_details.append({
                                    "name": result['file_name'],
                                    "status": "Queued",
                                    "doc_id": result.get('source_doc_id')
                                })
                                st.rerun()

                    except Exception as e:
                        st.error(f"Error processing URL: {e}")

            st.markdown("---")

            uploaded_files = st.file_uploader(
                "Upload Documents (PDF, PNG, JPG, TIFF)",
                type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "gif", "docx", "pptx" , "csv", "json", "xlsx", "txt"],
                accept_multiple_files=True,
                key="file_uploader" 
            )

            if uploaded_files:
                if st.button("Process Uploaded Documents", type="primary", use_container_width=True):
                    files_to_send_for_api = []
                    for uploaded_file_widget_instance in uploaded_files: 
                        files_to_send_for_api.append(
                            ("files", (uploaded_file_widget_instance.name, uploaded_file_widget_instance.getvalue(), uploaded_file_widget_instance.type))
                        )

                    if files_to_send_for_api:
                        if st.session_state.processing_active_message_placeholder is None:
                                st.session_state.processing_active_message_placeholder = st.empty()
                        
                        with st.session_state.processing_active_message_placeholder.container():
                            st.info(
                                (
                                    "🚀 **Processing initiated using fast rule-based chunking!** "
                                    "Your documents are being queued. This involves text extraction and OCR (for images). "
                                    "Please monitor backend logs. You can query once processing seems complete."
                                )
                            )
                        st.session_state.user_has_been_warned_about_processing = True

                        try:
                            with st.spinner(f"Uploading & queueing {len(files_to_send_for_api)} document(s)..."):
                                response = requests.post(
                                    DOCUMENTS_UPLOAD_URL, 
                                    files=files_to_send_for_api,
                                        params={"collection": st.session_state.selected_collection},
                                        timeout=60
                                ) 
                                response.raise_for_status() 
                                
                                results = response.json()
                                current_uploads = []
                                successful_queues = 0
                                for res_item in results:
                                    file_name = res_item.get("file_name", "Unknown file")
                                    status = res_item.get("status", "failed")
                                    doc_id = res_item.get("source_doc_id")
                                    mode_used = res_item.get("processing_mode_used", "fast_rule_based") 
                                    
                                    if status == "queued_for_processing":
                                        st.success(f"'{file_name}' (ID: {doc_id}) queued for {mode_used} processing.")
                                        current_uploads.append({"name": file_name, "status": f"Queued ({mode_used})", "doc_id": doc_id})
                                        successful_queues +=1
                                    else:
                                        st.error(f"Failed to queue '{file_name}': {res_item.get('message', 'Unknown error')}")
                                        current_uploads.append({"name": file_name, "status": f"Failed: {res_item.get('message', '')[:50]}...", "doc_id": doc_id})
                                
                                existing_names = {f['name'] for f in st.session_state.uploaded_file_details}
                                for up_detail in current_uploads:
                                    if up_detail['name'] not in existing_names:
                                        st.session_state.uploaded_file_details.append(up_detail)
                                        existing_names.add(up_detail['name'])
                                    else: 
                                        for i, existing_f in enumerate(st.session_state.uploaded_file_details):
                                            if existing_f['name'] == up_detail['name']:
                                                st.session_state.uploaded_file_details[i] = up_detail
                                                break
                                if successful_queues == 0 and files_to_send_for_api:
                                     if st.session_state.processing_active_message_placeholder:
                                        st.session_state.processing_active_message_placeholder.error("⚠️ No files were successfully queued.")
                        except Exception as e:
                            st.error(f"Error during upload: {e}")
                            if st.session_state.processing_active_message_placeholder:
                                st.session_state.processing_active_message_placeholder.empty() 
                    else:
                        st.warning("No files selected to upload.")
        
        st.markdown("---")
        st.subheader("Uploaded Documents Log")
        if st.session_state.uploaded_file_details:
            for detail in st.session_state.uploaded_file_details:
                status_emoji = "⏳" if "Queued" in detail['status'] else "❌" 
                st.markdown(f"- {status_emoji} **{detail['name']}** (ID: `{detail.get('doc_id', 'N/A')}` | Status: {detail['status']})")
        else:
            st.info("No documents uploaded in this session yet.")
        
        if st.button("Clear Chat, Logs & Processing Message", use_container_width=True):
            reset_chat_history()
            st.session_state.uploaded_file_details = []
            st.session_state.processing_info_message = None 
            st.rerun()

    # --- Main Chat Interface ---
    st.header("💬 Chat with Your Documents")

    if "processing_info_message" in st.session_state and st.session_state.processing_info_message:
        if st.session_state.processing_active_message_placeholder is None:
            st.session_state.processing_active_message_placeholder = st.empty()
        st.session_state.processing_active_message_placeholder.info(st.session_state.processing_info_message)

    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        if st.session_state.processing_active_message_placeholder:
            st.session_state.processing_active_message_placeholder.empty()
            st.session_state.processing_active_message_placeholder = None 
            st.session_state.processing_info_message = None 
            st.session_state.user_has_been_warned_about_processing = False 

        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message("user", prompt)

        with st.spinner("Thinking... (This may take a moment for complex queries)"):
            try:
                payload = {
                    "query": prompt,
                    "collection": st.session_state.selected_collection
                }
                response = requests.post(CHAT_QUERY_URL, json=payload, timeout=240) 
                response.raise_for_status()
                assistant_response_data = response.json()
                
                st.session_state.messages.append({"role": "assistant", "content": assistant_response_data})
                display_chat_message("assistant", assistant_response_data)

                if isinstance(assistant_response_data, dict):
                    for doc_id in assistant_response_data.get('retrieved_context_document_ids', []):
                        st.session_state.processed_doc_ids.add(doc_id)
            except requests.exceptions.Timeout:
                st.error("The request to the backend timed out. The server might be busy or the query is too complex.")
                st.session_state.messages.append({"role": "assistant", "content": "Sorry, I took too long to respond. Please try again or simplify your query."})
                display_chat_message("assistant", "Sorry, I took too long to respond. Please try again or simplify your query.")
            except requests.exceptions.RequestException as e: 
                st.error(f"Error communicating with the backend: {e}")
                error_message_ui = f"Sorry, I encountered an error trying to process your request. Please check the backend server. (Error: {str(e)[:100]})"
                st.session_state.messages.append({"role": "assistant", "content": error_message_ui})
                display_chat_message("assistant", error_message_ui)
            except json.JSONDecodeError:
                st.error("Received an invalid response from the backend (not valid JSON). Please check server logs.")
                error_message_ui = "Sorry, the backend response was not in the expected format."
                st.session_state.messages.append({"role": "assistant", "content": error_message_ui})
                display_chat_message("assistant", error_message_ui)
            except Exception as e: 
                st.error(f"An unexpected error occurred in the frontend: {e}")
                error_message_ui = f"An unexpected error occurred: {str(e)[:150]}..."
                st.session_state.messages.append({"role": "assistant", "content": error_message_ui})
                display_chat_message("assistant", error_message_ui)

if __name__ == "__main__":
    main()
