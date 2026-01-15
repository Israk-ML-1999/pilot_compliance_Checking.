import base64
import json
from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from config import settings

# Setup Models
embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=settings.COHERE_API_KEY
)

# Using Gemini 1.5 Pro (Best for reasoning across multiple images)
llm_reasoning = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-pro", 
    temperature=0,
    google_api_key=settings.GOOGLE_API_KEY
)

def get_vector_store():
    return Chroma(
        persist_directory=settings.CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name="pilot_rules"
    )

async def analyze_compliance(query_text: str = None, uploaded_files: List[Dict[str, Any]] = None):
    vectorstore = get_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) 
    
    extracted_schedule_data = ""

    # --- PHASE 1: VISION EXTRACTION (If files exist) ---
    if uploaded_files:
        content_parts = []
        
        # --- THE "BEST ANALYSIS" PROMPT ---
        # This prompt enforces ZULU time and Continuity across multiple images
        vision_prompt = """
        Analyze the attached files (Images or PDFs) as a SINGLE continuous Pilot Schedule.
        
        **CRITICAL INSTRUCTIONS FOR HIGH ACCURACY:**
        1. **TIME ZONE:** The schedule may show multiple times (e.g., "12:53 Z" and "03:53 AKST"). When extracting DATE and TIME must carefully consider the time zone KEEP THE SAME DATE.
           You MUST ONLY use the 'Z' (Zulu/UTC) time for ALL calculations. IGNORE AKST, KST, EST, etc.
        2. **CONTINUITY:** If a flight appears at the bottom of Image 1 and continues on Image 2, treat it as one timeline.
        3. **CALCULATIONS:**
           - Calculate "Flight Duration" (ETA Z - ETD Z).
           - Calculate "Rest Period" (Time gap between Previous Flight's Arrival Z and Current Flight's Departure Z).
        4. Duty Time and Flight Time:
           - Duty Time is the total time from ETD to ETA.
           - Flight Time is the total time from ETD to ETA.
           - Carefully calculate the duty time and flight time than apply the ruls to check compliance.
        **OUTPUT FORMAT:**
        Return a text summary listing:
        - Date
        - duty period
        - Flight Number
        - Dep Z / Arr Z
        - Flight Duration
        - Previous Rest Period
        """
        content_parts.append({"type": "text", "text": vision_prompt})

        # Loop through all files and add them to the SAME message
        for file_info in uploaded_files:
            file_path = file_info["path"]
            mime_type = file_info["mime_type"] or "application/octet-stream"
            
            with open(file_path, "rb") as f:
                file_bytes = f.read()
                b64_data = base64.b64encode(file_bytes).decode("utf-8")

            if "pdf" in mime_type:
                # Handle PDF
                content_parts.append({
                    "type": "media", 
                    "mime_type": "application/pdf", 
                    "data": b64_data
                })
            else:
                # Handle Images (jpeg, png, etc.)
                # Ensure mime type is valid image type, default to jpeg if unsure
                img_mime = mime_type if "image" in mime_type else "image/jpeg"
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{img_mime};base64,{b64_data}"}
                })

        # Call Gemini (Sends Text + Image 1 + Image 2 + ... in one request)
        msg = HumanMessage(content=content_parts)
        extraction_res = await llm_reasoning.ainvoke([msg])
        extracted_schedule_data = extraction_res.content

    # --- PHASE 2: RETRIEVAL (RAG) ---
    # Construct search query: User Question + Key terms from extracted schedule
    search_context = query_text if query_text else ""
    if extracted_schedule_data:
        search_context += f" {extracted_schedule_data[:500]}" 
    
    relevant_docs = retriever.invoke(search_context)
    rules_context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # --- PHASE 3: COMPLIANCE REASONING ---
    
    # Scenario A: We have Schedule Data -> Return JSON Report
    if extracted_schedule_data:
        final_prompt = f"""
        You are a Senior Aviation Compliance Officer.carefully check the schedule and rules to identify any violations also Alcohal and Drug test.        
        **OFFICIAL RULES (Context):**.Always check (Context) to identify any violations with reference to rules where break the rules. provide every ans you must the check *(Context) OFFICIAL RULES*
        {rules_context}
        
        **EXTRACTED SCHEDULE (Analyzed Data):**
        {extracted_schedule_data}
        
        **USER QUESTION/NOTE:**
        {query_text if query_text else "Perform full compliance check."}
        
        **TASK:**
        1. Summarize the schedule (Focus on Flight Duty Periods & Rest).
        2. Check the schedule against the Rules. Use the calculated Z-time values.
        3. Identify violations.
        
        **OUTPUT (Strict JSON):**
        {{
            "schedule_summary": {{ "key_points": ["Summary point 1", "Summary point 2"] }},
            "violations": [
                {{ "rule_reference": "Section X.X (Where is rules violations)", "description": "Violation details" }}
            ],
            "email_report": "Draft email to pilot (or null if compliant). The email should be in HTML format and should include the schedule summary and violations. Mention the rule reference and the description of the violation. Wright the email small and clear"
        }}
        """
        
        final_res = await llm_reasoning.ainvoke([HumanMessage(content=final_prompt)])
        
        # Clean JSON
        clean_json = final_res.content.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(clean_json)
        except:
            return {
                "schedule_summary": {"key_points": ["Error parsing JSON", query_text]},
                "violations": [],
                "email_report": final_res.content
            }

    # Scenario B: Text Query Only (No Files) -> Return Text Answer
    else:
        prompt = f"""
        Answer the user's question based strictly on the Pilot Rules.
        
        **Rules:**
        {rules_context}
        
        **Question:** {query_text}
        """
        res = await llm_reasoning.ainvoke([HumanMessage(content=prompt)])
        
        return {
            "schedule_summary": "N/A - Text Query Only",
            "violations": [],
            "email_report": None,
            "answer": res.content
        }