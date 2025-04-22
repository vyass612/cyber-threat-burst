import os

from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables from .env
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Create a retriever for querying the vector store
# `search_type` specifies the type of search (e.g., similarity)
# `search_kwargs` contains additional arguments for the search (e.g., number of results to return)
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# Create a ChatOpenAI model
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Contextualize question prompt
# This system prompt helps the AI understand that it should reformulate the question
# based on the chat history to make it a standalone question
contextualize_q_system_prompt = (

    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."

)

def check_scope_of_question(question):
    prompt = pre_question_prompt.format_messages(input=question)
    response = llm_pre_question.invoke(prompt)
    return response.content.strip()



#Create a second LLM for pre-processing the question
llm_pre_question= ChatOpenAI(model= "gpt-4o")
pre_question_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a guardrail for a retrieval-augmented generation (RAG) assistant. "
     "Your job is to check if the user's question is likely to be answerable from a known internal knowledge base. "
     "If the question is general, personal, speculative, or otherwise not likely to have a factual answer based on a document database, "
     "respond with 'OUT_OF_SCOPE'. Otherwise, respond with 'IN_SCOPE'. Do not explain."),
    ("human", "{input}")
])

llm_pre_answer= ChatOpenAI(model= "gpt-4o")


pre_answer_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a fact-checker assistant. Given a context (retrieved documents) and an answer, "
     "verify whether the answer is fully supported by the context. "
     "If the answer is directly grounded in the context, respond with 'VALID'. "
     "If it is not supported or includes made-up information, respond with 'HALLUCINATED'. "
     "Do not provide explanations."),
    ("human", "Context:\n{context}\n\nAnswer:\n{answer}")
])


def check_for_hallucination(context_docs, answer):
    context_text= "\n\n".join(doc.page_content for doc in context_docs) 
    prompt= pre_answer_prompt.format_messages(context= context_text, answer= answer)
    response= llm_pre_answer.invoke(prompt)
    return response.content.strip()


    


# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever
# This uses the LLM to help reformulate the question based on chat history
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer question prompt
# This system prompt helps the AI understand that it should provide concise answers
# based on the retrieved context and indicates what to do if the answer is unknown
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question."
    "\n\n"
    "{context}"
)

# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for question answering
# `create_stuff_documents_chain` feeds all retrieved context into the LLM
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a retrieval chain that combines the history-aware retriever and the question answering chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# Function to simulate a continual chat
def continual_chat():

    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []  # Collect chat history here (a sequence of messages)
    
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
    
        # Step 1: Scope check using llm_pre_question
        # scope = check_scope_of_question(query)
        # if scope == "OUT_OF_SCOPE":
        #     print("AI: Sorry, that question seems to be out of the database's scope.")
        #     continue

        # Step 2: Use RAG chain to get answer + retrieve context
            # Process the user's query through the retrieval chain
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
            # Display the AI's response
        answer= result["answer"]
        print(f"AI: {result['answer']}")
        retrieved_docs= result['context']

        # # Step 3: Hallucination check using llm_pre_answer
        validation = check_for_hallucination(retrieved_docs, answer)
        if validation == "HALLUCINATED":
            print("AI: The answer seems to be hallucinated. I will not use it.")
        else:
            print("AI: {answer} ")

        # Update the chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))


# Main function to start the continual chat
if __name__ == "__main__":
    continual_chat()
