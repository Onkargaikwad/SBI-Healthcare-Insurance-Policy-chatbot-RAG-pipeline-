import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from dotenv import load_dotenv

from langchain.chains import RetrievalQA

from langchain_groq.chat_models import ChatGroq

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough



from rgbot.ingest import ingest_data  # must return a LangChain vectorstore

load_dotenv()

# config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(SCRIPT_DIR, "data", "SBI_General_Health_Insurance.pdf")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
TOP_K = int(os.environ.get("TOP_K", 3))

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ingest vectorstore (ensure ingest_data returns a LangChain vectorstore)
vstore = ingest_data(PDF_PATH)
if vstore is None:
    raise RuntimeError("ingest_data returned None — it must return a LangChain vectorstore.")
retriever = vstore.as_retriever(search_kwargs={"k": TOP_K})

# Groq LLM + prompt + chain
llm = ChatGroq(model=GROQ_MODEL)
TEMPLATE = """You are a helpful and intelligent chatbot.

Respond based on the nature of the question:

- If the question is related to insurance (e.g., health insurance, policies, premiums, claims, waiting periods, coverages, etc.), act as an **Insurance Assistant**.
  - Use only the information provided in the given context to answer.
  - Be clear, concise, and factual.
  - If the answer is not found in the context, respond with: 
    "I could not find this information in the provided policy document. Please check with the insurer directly for confirmation."

- If the question is general (not related to insurance), answer normally like a general-purpose chatbot without referring to the insurance context, keeping responses concise and appropriate unless extra detail is requested.

Context (for insurance-related questions only):
{context}

Question: {question}

Examples:
- Question: Hi
  Answer: Hi, how can I help you today?

- Question: What is the premium amount for SBI Super Health Insurance?
  Answer: The premium amount for SBI Super Health Insurance depends on several factors, including the age of the insured, sum insured, and policy term. According to the prospectus, premiums are listed under different plan options and sum insured amounts.

- Question: What is the maximum sum insured in SBI Super Health Insurance?
  Answer: According to the policy prospectus, the maximum sum insured is ₹2 Crores.

- Question: What is the waiting period for pre-existing diseases?
  Answer: As per the policy, pre-existing diseases are covered after 24 months of continuous coverage, provided they are declared and accepted by the insurer.

- Question: Who is the Prime Minister of India?
  Answer: As of 2025, the Prime Minister of India is Narendra Modi.

- Question: Can you explain black holes in simple terms?
  Answer: Sure! A black hole is a region in space where gravity is so strong that not even light can escape.

- Question: What are the tax benefits of buying health insurance?
  Answer: Health insurance premiums paid in India qualify for tax deductions under Section 80D of the Income Tax Act.
"""
prompt = ChatPromptTemplate.from_template(TEMPLATE)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query")
async def query(request: Request):
    data = await request.json()
    question = data.get("question")
    if not question:
        return {"error": "No question provided"}
    response = chain.invoke(question)
    return {"answer": response}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=9003, reload=True)
