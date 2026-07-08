from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Prompt for document analysis
document_analysis_prompt = ChatPromptTemplate.from_template("""
You are a highly capable legal document analyst. Analyze the provided document and return ONLY valid JSON matching the exact schema below.

{format_instructions}

Analyze this document:
{document_text}
""")

# Prompt for document comparison
document_comparison_prompt = ChatPromptTemplate.from_template("""
You are a legal document comparison expert. Compare the two documents provided and identify ALL differences.

{format_instruction}

Reference Document:
{combined_docs}

Provide a page-by-page comparison. In the JSON response, format the 'Changes' value as a clean, bullet-pointed list (e.g. using '•' for each point) listing all the differences on that page. If a page has no changes, set the 'Changes' string to 'NO CHANGE'.
""")


# Prompt for contextual question rewriting
contextualize_question_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Given a conversation history and the most recent user query about a legal contract, "
        "rewrite the query as a standalone question that makes sense without the previous context. "
        "Do not answer — only reformulate if necessary; otherwise return unchanged."
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Prompt for answering based on context
context_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a precise legal document assistant analyzing contracts. "
        "Answer questions using ONLY the retrieved contract context below. "
        "When referencing obligations, deadlines, or parties — be specific and cite page numbers when available. "
        "If the answer is not in the context, say 'This information is not found in the document.' "
        "Never speculate or add information not present in the contract.\n\n"
        "Retrieved Context:\n{context}"
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Prompt for key-term extraction (Phase 1C)
key_term_extraction_prompt = ChatPromptTemplate.from_template("""
You are an expert legal contract analyst. Extract ALL key terms from the following contract.

For each field, provide a confidence score (0.0 to 1.0):
- 1.0 = clearly stated in the contract
- 0.5 = inferred from context
- 0.0 = not found in the document

Return ONLY valid JSON with these fields:
- parties: [{{"name": "...", "role": "buyer/seller/etc."}}]
- effective_date, expiration_date, auto_renewal
- payment_terms: [{{"amount": "...", "currency": "...", "frequency": "...", "due_date": "..."}}]
- obligations: [{{"party": "...", "description": "...", "deadline": "...", "consequence_of_breach": "..."}}]
- termination_clauses: ["clause summary"]
- liability_cap, confidentiality_period, governing_law
- non_compete: {{"exists": true/false, "duration": "...", "geographic_scope": "..."}}
- risk_flags: [{{"clause": "...", "risk_level": "low/medium/high/critical", "plain_english": "...", "recommendation": "..."}}]
- confidence_scores: {{"field_name": 0.0-1.0}}

Do NOT hallucinate. If a term is not in the document, set it to "Not found" with confidence 0.0.

Contract text:
{document_text}
""")

# Prompt for extraction validation (self-correction loop)
validation_prompt = ChatPromptTemplate.from_template("""
You are a quality assurance analyst reviewing AI-extracted contract terms.

Review the following extraction result and identify:
1. Any fields that appear incorrect or hallucinated
2. Any critical fields that are missing
3. Any confidence scores that seem too high or too low

Previous extraction:
{extracted_terms}

Original document text (first 5000 chars):
{document_text_preview}

Return JSON with:
- "issues": ["list of problems found"]
- "suggested_fixes": {{"field_name": "corrected_value"}}
- "overall_quality": 0.0-1.0
- "should_retry": true/false
""")

# Central dictionary to register prompts
PROMPT_REGISTRY = {
    "document_analysis": document_analysis_prompt,
    "document_comparison": document_comparison_prompt,
    "contextualize_question": contextualize_question_prompt,
    "context_qa": context_qa_prompt,
    "key_term_extraction": key_term_extraction_prompt,
    "validation": validation_prompt,
}
