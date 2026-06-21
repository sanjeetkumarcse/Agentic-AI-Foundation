from flask import Flask, render_template, request, jsonify, session
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

app = Flask(__name__)
app.secret_key = "my-secret-key"


@app.route("/")
def home():
    if "chat_history" not in session:
        session["chat_history"] = []

    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json

    endpoint = data["endpoint"]
    api_key = data["api_key"]
    deployment = data["deployment"]
    api_version = data.get("api_version", "2024-10-21")
    question = data["message"]

    llm = AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        azure_deployment=deployment,
        api_version=api_version,
        temperature=0.7
    )

    chat_history = session.get("chat_history", [])

    messages = []

    for item in chat_history:
        messages.append(HumanMessage(content=item["human"]))
        messages.append(AIMessage(content=item["ai"]))

    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages)

    answer = response.content

    chat_history.append({
        "human": question,
        "ai": answer
    })

    session["chat_history"] = chat_history

    return jsonify({"response": answer})