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

Provide a page-by-page comparison. If a page has no changes, state 'NO CHANGE'.
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

# Central dictionary to register prompts
PROMPT_REGISTRY = {
    "document_analysis": document_analysis_prompt,
    "document_comparison": document_comparison_prompt,
    "contextualize_question": contextualize_question_prompt,
    "context_qa": context_qa_prompt,
}
