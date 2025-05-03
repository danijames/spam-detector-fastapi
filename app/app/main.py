from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

# Load model and vectorizer
with open('models/spam_classifier_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Define input data model
class Message(BaseModel):
    text: str

# Create FastAPI app
app = FastAPI()

# Define prediction route
@app.post("/predict")
def predict(message: Message):
    text = message.text.lower()
    vectorized_text = vectorizer.transform([text])
    
    # Get probability estimates
    probabilities = model.predict_proba(vectorized_text)[0]
    spam_probability = probabilities[1]  # index 1 → spam class probability

    # Prediction based on highest probability
    prediction = "spam" if spam_probability >= 0.5 else "ham"
    
    return {
        "prediction": prediction,
        "spam_probability": round(float(spam_probability), 3)  # rounded to 3 decimal points
    }

# For HTML templates
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)


def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})
