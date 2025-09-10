from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import joblib
import numpy as np
import os
import logging

app = FastAPI(title="FraudGuard AI", description="Real-time credit card fraud detection")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and scaler
model_path = "fraud_model.pkl"
scaler_path = "scaler.pkl"
for path in [model_path, scaler_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File '{path}' not found.")
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

class TransactionInput(BaseModel):
    TransactionTime: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    TransactionAmount: float

security = HTTPBasic()

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != "admin" or credentials.password != "fraudguard123":
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return credentials

@app.post("/predict")
async def predict_fraud(input_data: TransactionInput, credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    try:
        raw_features = [
            input_data.TransactionTime,
            input_data.V1, input_data.V2, input_data.V3, input_data.V4,
            input_data.V5, input_data.V6, input_data.V7, input_data.V8,
            input_data.V9, input_data.V10, input_data.V11, input_data.V12,
            input_data.V13, input_data.V14, input_data.V15, input_data.V16,
            input_data.V17, input_data.V18, input_data.V19, input_data.V20,
            input_data.V21, input_data.V22, input_data.V23, input_data.V24,
            input_data.V25, input_data.V26, input_data.V27, input_data.V28,
            input_data.TransactionAmount
        ]
        scaled_features = scaler.transform([raw_features])[0]
        features = np.array(scaled_features).reshape(1, -1)

        if not (0 <= input_data.TransactionTime <= 172792):
            raise HTTPException(status_code=400, detail="TransactionTime must be between 0 and 172792.")
        if input_data.TransactionAmount < 0:
            raise HTTPException(status_code=400, detail="TransactionAmount cannot be negative.")

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        result = {
            "prediction": int(prediction),
            "fraud_probability": float(probability),
            "interpretation": "Fraud" if prediction == 1 else "No Fraud"
        }
        logger.info(f"Prediction made: {result}")
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FraudGuard AI - Real-Time Fraud Detection</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .fade-in { animation: fadeIn 0.5s ease-in-out; }
            .chatbot-container { transition: all 0.3s ease; }
            .chatbot-hidden { transform: translateY(100%); }
            .chatbot-visible { transform: translateY(0); }
        </style>
    </head>
    <body class="bg-gray-100 min-h-screen flex items-center justify-center">
        <div class="container mx-auto p-6 max-w-4xl">
            <h1 class="text-3xl font-bold text-center text-gray-800 mb-6">FraudGuard AI</h1>
            <p class="text-center text-gray-600 mb-8">Enter transaction details to detect potential fraud (for testing, use dataset values).</p>

            <div class="bg-white shadow-lg rounded-lg p-6 mb-8">
                <form id="fraudForm" class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <label for="TransactionTime" class="block text-sm font-medium text-gray-700">Transaction Time (seconds)</label>
                        <input type="number" id="TransactionTime" step="0.01" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., 0.0">
                    </div>
                    <div>
                        <label for="V1" class="block text-sm font-medium text-gray-700">Feature V1</label>
                        <input type="number" id="V1" step="0.000001" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., -1.359807">
                    </div>
                    <div>
                        <label for="V2" class="block text-sm font-medium text-gray-700">Feature V2</label>
                        <input type="number" id="V2" step="0.000001" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., -0.072781">
                    </div>
                    <div>
                        <label for="V3" class="block text-sm font-medium text-gray-700">Feature V3</label>
                        <input type="number" id="V3" step="0.000001" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., 2.536347">
                    </div>
                    <div>
                        <label for="V4" class="block text-sm font-medium text-gray-700">Feature V4</label>
                        <input type="number" id="V4" step="0.000001" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., 1.378155">
                    </div>
                    <div>
                        <label for="V5" class="block text-sm font-medium text-gray-700">Feature V5</label>
                        <input type="number" id="V5" step="0.000001" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., -0.338321">
                    </div>
                    <div>
                        <label for="V6" class="block text-sm font-medium text-gray-700">Feature V6</label>
                        <input type="number" id="V6" step="0.000001" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., 0.462388">
                    </div>
                    <div>
                        <label for="V7" class="block text-sm font-medium text-gray-700">Feature V7</label>
                        <input type="number" id="V7" step="0.000001" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., 0.239599">
                    </div>
                    <div>
                        <label for="V8" class="block text-sm font-medium text-gray-700">Feature V8</label>
                        <input type="number" id="V8" step="0.000001" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., 0.098698">
                    </div>
                    <div>
                        <label for="V9" class="block text-sm font-medium text-gray-700">Feature V9</label>
                        <input type="number" id="V9" step="0.000001" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., 0.363787">
                    </div>
                    <div>
                        <label for="V10" class="block text-sm font-medium text-gray-700">Feature V10</label>
                        <input type="number" id="V10" step="0.000001" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., 0.090794">
                    </div>
                    <div>
                        <label for="V11" class="block text-sm font-medium text-gray-700">Feature V11</label>
                        <input type="number" id="V11" step="0.000001" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., -0.551600">
                    </div>
                    <div>
                        <label for="V12" class="block text-sm font-medium text-gray-700">Feature V12</label>
                        <input type="number" id="V12" step="0.000001" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., -0.617801">
                    </div>
                    <div>
                        <label for="V13" class="block text-sm font-medium text-gray-700">Feature V13</label>
                        <input type="number" id="V13" step="0.000001" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., -0.991390">
                    </div>
                    <div>
                        <label for="V14" class="block text-sm font-medium text-gray-700">Feature V14</label>
                        <input type="number" id="V14" step="0.000001" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., -0.311169">
                    </div>
                    <div>
                        <label for="V15" class="block text-sm font-medium text-gray-700">Feature V15</label>
                        <input type="number" id="V15" step="0.000001" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., 1.468177">
                    </div>
                    <div>
                        <label for="V16" class="block text-sm font-medium text-gray-700">Feature V16</label>
                        <input type="number" id="V16" step="0.000001" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., -0.470401">
                    </div>
                    <div>
                        <label for="V17" class="block text-sm font-medium text-gray-700">Feature V17</label>
                        <input type="number" id="V17" step="0.000001" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., 0.207971">
                    </div>
                    <div>
                        <label for="V18" class="block text-sm font-medium text-gray-700">Feature V18</label>
                        <input type="number" id="V18" step="0.000001" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., 0.025791">
                    </div>
                    <div>
                        <label for="V19" class="block text-sm font-medium text-gray-700">Feature V19</label>
                        <input type="number" id="V19" step="0.000001" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., 0.403993">
                    </div>
                    <div>
                        <label for="V20" class="block text-sm font-medium text-gray-700">Feature V20</label>
                        <input type="number" id="V20" step="0.000001" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., 0.251412">
                    </div>
                    <div>
                        <label for="V21" class="block text-sm font-medium text-gray-700">Feature V21</label>
                        <input type="number" id="V21" step="0.000001" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., -0.018307">
                    </div>
                    <div>
                        <label for="V22" class="block text-sm font-medium text-gray-700">Feature V22</label>
                        <input type="number" id="V22" step="0.000001" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., 0.277838">
                    </div>
                    <div>
                        <label for="V23" class="block text-sm font-medium text-gray-700">Feature V23</label>
                        <input type="number" id="V23" step="0.000001" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., -0.110474">
                    </div>
                    <div>
                        <label for="V24" class="block text-sm font-medium text-gray-700">Feature V24</label>
                        <input type="number" id="V24" step="0.000001" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., 0.066928">
                    </div>
                    <div>
                        <label for="V25" class="block text-sm font-medium text-gray-700">Feature V25</label>
                        <input type="number" id="V25" step="0.000001" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., 0.128539">
                    </div>
                    <div>
                        <label for="V26" class="block text-sm font-medium text-gray-700">Feature V26</label>
                        <input type="number" id="V26" step="0.000001" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., -0.189115">
                    </div>
                    <div>
                        <label for="V27" class="block text-sm font-medium text-gray-700">Feature V27</label>
                        <input type="number" id="V27" step="0.000001" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., 0.133558">
                    </div>
                    <div>
                        <label for="V28" class="block text-sm font-medium text-gray-700">Feature V28</label>
                        <input type="number" id="V28" step="0.000001" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., -0.021053">
                    </div>
                    <div>
                        <label for="TransactionAmount" class="block text-sm font-medium text-gray-700">Transaction Amount</label>
                        <input type="number" id="TransactionAmount" step="0.01" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., 149.62">
                    </div>
                    <div class="md:col-span-2">
                        <button type="submit" class="w-full py-2 px-4 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 transition duration-300">Predict Fraud</button>
                    </div>
                </form>
                <div id="result" class="mt-6 p-4 bg-gray-50 rounded-md hidden"></div>
            </div>

            <div id="chatbot" class="fixed bottom-4 right-4 w-80 bg-white shadow-lg rounded-lg overflow-hidden chatbot-container chatbot-hidden">
                <div class="bg-indigo-600 text-white p-3 flex justify-between items-center">
                    <span>FraudGuard AI Assistant</span>
                    <button id="toggleChatbot" class="text-white hover:text-gray-200">−</button>
                </div>
                <div id="chatMessages" class="p-4 h-64 overflow-y-auto bg-gray-50">
                    <p class="text-gray-600">Welcome to FraudGuard AI! Ask me about fraud detection or how to use the form.</p>
                    <ul class="text-sm text-indigo-600 mt-2">
                        <li><a href="#" class="chat-option" data-question="How do I use this form?">How do I use this form?</a></li>
                        <li><a href="#" class="chat-option" data-question="What does the model do?">What does the model do?</a></li>
                        <li><a href="#" class="chat-option" data-question="How accurate is the model?">How accurate is the model?</a></li>
                        <li><a href="#" class="chat-option" data-question="What if I get a fraud alert?">What if I get a fraud alert?</a></li>
                    </ul>
                </div>
                <div class="p-2 border-t">
                    <input id="chatInput" type="text" placeholder="Type your question..." class="w-full p-2 border rounded-md focus:ring-indigo-500 focus:border-indigo-500">
                </div>
            </div>
            <button id="openChatbot" class="fixed bottom-4 right-4 bg-indigo-600 text-white p-3 rounded-full shadow-lg hover:bg-indigo-700 transition duration-300 chatbot-hidden">💬</button>
        </div>

        <script>
            document.getElementById('fraudForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const formData = new FormData(e.target);
                const data = Object.fromEntries(formData);
                const inputData = {
                    TransactionTime: parseFloat(data.TransactionTime),
                    V1: parseFloat(data.V1), V2: parseFloat(data.V2), V3: parseFloat(data.V3), V4: parseFloat(data.V4),
                    V5: parseFloat(data.V5), V6: parseFloat(data.V6), V7: parseFloat(data.V7), V8: parseFloat(data.V8),
                    V9: parseFloat(data.V9), V10: parseFloat(data.V10), V11: parseFloat(data.V11), V12: parseFloat(data.V12),
                    V13: parseFloat(data.V13), V14: parseFloat(data.V14), V15: parseFloat(data.V15), V16: parseFloat(data.V16),
                    V17: parseFloat(data.V17), V18: parseFloat(data.V18), V19: parseFloat(data.V19), V20: parseFloat(data.V20),
                    V21: parseFloat(data.V21), V22: parseFloat(data.V22), V23: parseFloat(data.V23), V24: parseFloat(data.V24),
                    V25: parseFloat(data.V25), V26: parseFloat(data.V26), V27: parseFloat(data.V27), V28: parseFloat(data.V28),
                    TransactionAmount: parseFloat(data.TransactionAmount)
                };

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': 'Basic ' + btoa('admin:fraudguard123')
                        },
                        body: JSON.stringify(inputData)
                    });
                    if (response.status === 401) throw new Error('Invalid credentials');
                    const result = await response.json();
                    const resultDiv = document.getElementById('result');
                    resultDiv.classList.remove('hidden');
                    resultDiv.classList.add('fade-in');
                    resultDiv.innerHTML = `
                        <h3 class="text-lg font-semibold text-gray-800">Prediction Result</h3>
                        <p><strong>Prediction:</strong> ${result.interpretation}</p>
                        <p><strong>Fraud Probability:</strong> ${(result.fraud_probability * 100).toFixed(2)}%</p>
                    `;
                } catch (error) {
                    const resultDiv = document.getElementById('result');
                    resultDiv.classList.remove('hidden');
                    resultDiv.classList.add('fade-in');
                    resultDiv.innerHTML = `<p class="text-red-600">Error: ${error.message}</p>`;
                }
            });

            const chatMessages = document.getElementById('chatMessages');
            const chatInput = document.getElementById('chatInput');
            const chatbot = document.getElementById('chatbot');
            const openChatbot = document.getElementById('openChatbot');
            const toggleChatbot = document.getElementById('toggleChatbot');

            const responses = {
                "How do I use this form?": "Enter transaction details from the dataset, including Transaction Time, V1-V28, and Amount. Use values like those in creditcard.csv (e.g., Time=0.0, V1=-1.359807, Amount=149.62). Click 'Predict Fraud' to check for fraud.",
                "What does the model do?": "FraudGuard AI uses a Random Forest model to predict if a transaction is fraudulent based on patterns in the creditcard.csv dataset.",
                "How accurate is the model?": "The model achieves ~99.99% accuracy, with high precision and recall for fraud detection, based on test data.",
                "What if I get a fraud alert?": "A fraud alert indicates a high likelihood of fraud. In a real-world setting, flag the transaction for review or contact the cardholder."
            };

            function addMessage(message, isUser = false) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `mt-2 ${isUser ? 'text-right' : 'text-left'}`;
                messageDiv.innerHTML = `<p class="${isUser ? 'bg-indigo-100' : 'bg-gray-200'} inline-block p-2 rounded-md">${message}</p>`;
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }

            document.querySelectorAll('.chat-option').forEach(option => {
                option.addEventListener('click', (e) => {
                    e.preventDefault();
                    const question = e.target.dataset.question;
                    addMessage(question, true);
                    const response = responses[question] || "Sorry, I don't understand. Try asking about the form, model, or fraud alerts!";
                    setTimeout(() => addMessage(response), 500);
                });
            });

            chatInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && chatInput.value.trim()) {
                    const userInput = chatInput.value.trim();
                    addMessage(userInput, true);
                    const response = responses[userInput] || "Sorry, I don't understand. Try asking about the form, model, or fraud alerts!";
                    setTimeout(() => addMessage(response), 500);
                    chatInput.value = '';
                }
            });

            toggleChatbot.addEventListener('click', () => {
                chatbot.classList.toggle('chatbot-hidden');
                chatbot.classList.toggle('chatbot-visible');
                openChatbot.classList.toggle('chatbot-hidden');
            });

            openChatbot.addEventListener('click', () => {
                chatbot.classList.toggle('chatbot-hidden');
                chatbot.classList.toggle('chatbot-visible');
                openChatbot.classList.toggle('chatbot-hidden');
            });
        </script>
    </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)