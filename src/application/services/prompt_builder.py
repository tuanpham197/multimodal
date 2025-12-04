from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from src.domain.entities import ChatContext

SYSTEM_PROMPT = """
Expert Document Analysis and Information Retrieval Agent
You are an expert Document Analysis and Information Retrieval Agent. Your primary function is to provide the most accurate, comprehensive, and contextually relevant answers to user queries based only on the provided document context.

1. Role and Goal
 - Role: Expert Document Analyst, Information Synthesizer, and Multimodal Data Interpreter.
 - Goal: To extract, synthesize, and present information from diverse document formats to answer the user's question with high fidelity and precision.

2. Core Instructions
 - Accuracy First: Base your answer strictly on the provided context.
 - Synthesis: Synthesize information into a coherent, well-structured response.
 - Handle Ambiguity: If context is insufficient, state clearly that the answer cannot be determined from provided documents.

3. Multimodal Instructions
 - Integrate Visual Data: Treat information from images/charts with same importance as text.
 - Prioritize Data: Use numerical data directly from charts/tables.
 - Describe Visual Evidence: Mention type of visual evidence when relevant.

4. Output Format
 - Start with a direct, concise answer followed by detailed explanation.
 - Maintain professional and objective tone.
"""


class PromptBuilder:
    def build(self, context: ChatContext, question: str) -> ChatPromptTemplate:
        has_context = context.texts or context.images

        if has_context:
            context_text = "\n".join(context.texts) if context.texts else ""
            prompt_text = f"""
{SYSTEM_PROMPT}

Context: {context_text}

Question: {question}

Answer the question based only on the provided context.
"""
        else:
            prompt_text = f"""
Answer the question. If you don't have enough information, just have a normal conversation.

Question: {question}
"""

        prompt_content = [{"type": "text", "text": prompt_text}]

        for image in context.images:
            prompt_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            })

        return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])


class ImageSummaryPromptBuilder:
    TEMPLATE = """Summarize this image concisely.
Focus on the key information: what it shows, main components, and purpose.
Special case: If the photo contains a chart or graph, analyze it carefully and list all numbers, values, and data points shown in detail. Include every label, unit, axis name, category, and corresponding value. Describe the chart’s structure (for example, which values belong to which categories or time periods). For instance, if the image shows a rainfall chart for each month, list every month and the exact rainfall amount for each one. Present the results clearly and completely in a structured format such as a bullet list or table.
Include every label, unit, axis name, category, and corresponding value.
Present the results clearly and completely in a structured format."""

    def build(self) -> ChatPromptTemplate:
        messages = [
            (
                "user",
                [
                    {"type": "text", "text": self.TEMPLATE},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}},
                ],
            )
        ]
        return ChatPromptTemplate.from_messages(messages)

