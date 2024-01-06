from flask import Flask, request, jsonify
from summarize import summarize

app = Flask(__name__)

@app.route('/summarize-text', methods=['POST'])
def summarize_text():
    data = request.get_json()
    input = data["sentences"] 
    result = summarize(input)
    res = {"summarized-text": result}
    return jsonify(res), 201

if __name__ == '__main__':
    app.run(debug=True)