from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from backend.app import GROQ_MODEL, GROQ_API_KEY


_llm = ChatOllama(
    model = GROQ_MODEL,
    base_url= GROQ_API_KEY
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Acting like a World Best Data Analytics Engineer 
            Think of the best way to represent below data and Generate SVG code to represent the data in form of chart precisely with proper labeling.
            Make sure to give a proper color to each components(like background , labels and more) so that color labeling don't collide with each other 
            make sure that font size of labels looks good.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Acting like a World Best Data Analytics Engineer"
            "Always provide detailed recommendations, including requests for labels , colouring , designing, overall representation etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_chain = generation_prompt | _llm
reflection_chain = reflection_prompt | _llm