from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_groq import ChatGroq

app = Flask(__name__)
CORS(app)

# Initialize Groq LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key="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",
    model_name="llama-3.3-70b-versatile"
)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    result = llm.invoke(user_input)
    return jsonify({"response": result.content})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
