from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
import os

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

app = Flask(__name__)
app.secret_key = "my-secret-key"

# Azure OpenAI Model
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.7
)


@app.route("/")
def home():

    if "chat_history" not in session:
        session["chat_history"] = []

    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():

    data = request.json
    question = data["message"]

    chat_history = session.get("chat_history", [])

    messages = []

    # Previous history
    for item in chat_history:
        messages.append(HumanMessage(content=item["human"]))
        messages.append(AIMessage(content=item["ai"]))

    # Current question
    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages)

    answer = response.content

    chat_history.append({
        "human": question,
        "ai": answer
    })

    session["chat_history"] = chat_history

    return jsonify({
        "response": answer
    })


@app.route("/clear")
def clear():

    session.clear()

    return jsonify({
        "message": "Chat history cleared"
    })


if __name__ == "__main__":
    app.run(debug=True)