from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import logging
import time

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="WhatsApp Text Classifier", version="1.0.0")

# Globale Variable für Modell
classifier = None

# Kandidaten-Labels (genau deine Kategorien)
CANDIDATE_LABELS = ["short_greeting", "problem_statement", "problem_request", "other"]

# Label-Beschreibungen für bessere Klassifikation
LABEL_DESCRIPTIONS = {
    "short_greeting": "Eine einfache Begrüßung wie Hallo, Hi oder Guten Tag ohne weitere Informationen",
    "problem_statement": "Eine konkrete Beschreibung eines technischen Problems oder eine spezifische Anfrage wie Laptop startet nicht, VPN Zugang benötigt oder Outlook Lizenz freischalten",
    "problem_request": "Eine allgemeine Bitte um Hilfe ohne konkrete Details wie Ich habe ein Problem oder Können Sie helfen",
    "other": "Dankesnachrichten, Smalltalk, Off-Topic Nachrichten oder unverständlicher Text",
}


class TextInput(BaseModel):
    text: str

    class Config:
        json_schema_extra = {"example": {"text": "Mein Laptop startet nicht mehr"}}


class ClassificationResponse(BaseModel):
    category: str
    confidence: float
    inference_time_ms: float
    all_scores: dict


@app.on_event("startup")
async def load_model():
    """Lade Modell beim Start - dauert ca. 30-60 Sekunden"""
    global classifier
    try:
        logger.info("Loading zero-shot classification model...")
        start_time = time.time()

        # Nutze multilingual model für deutsche Texte
        classifier = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
            device=-1,  # CPU
        )

        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f}s")

        # Test-Klassifikation
        test_result = classifier(
            "Hallo", list(LABEL_DESCRIPTIONS.values()), multi_label=False
        )
        logger.info(f"Test classification successful: {test_result['labels'][0]}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if classifier else "model_not_loaded",
        "model_loaded": classifier is not None,
        "categories": CANDIDATE_LABELS,
    }


@app.post("/classify", response_model=ClassificationResponse)
def classify_text(input: TextInput):
    """
    Klassifiziere WhatsApp-Support-Nachricht

    Args:
        input: TextInput mit 'text' field

    Returns:
        ClassificationResponse mit category, confidence und all_scores
    """
    if not classifier:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Validierung
    if not input.text or len(input.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        start_time = time.time()

        # Kürze Text auf max 512 Zeichen (für Performance)
        text = input.text.strip()[:512]

        # Zero-shot Klassifikation mit Beschreibungen
        result = classifier(
            text,
            list(LABEL_DESCRIPTIONS.values()),
            multi_label=False,
            hypothesis_template="Dieser Text ist {}.",  # Deutscher Template
        )

        # Mappe zurück zu Original-Labels
        label_scores = {}
        for label, desc, score in zip(
            CANDIDATE_LABELS,
            LABEL_DESCRIPTIONS.values(),
            [
                result["scores"][result["labels"].index(desc)]
                for desc in LABEL_DESCRIPTIONS.values()
            ],
        ):
            label_scores[label] = round(score, 4)

        # Finde beste Kategorie
        best_label = max(label_scores, key=label_scores.get)
        best_score = label_scores[best_label]

        inference_time = (time.time() - start_time) * 1000  # in ms

        logger.info(
            f"Classified '{text[:50]}...' as '{best_label}' ({best_score:.2f}) in {inference_time:.0f}ms"
        )

        return ClassificationResponse(
            category=best_label,
            confidence=best_score,
            inference_time_ms=round(inference_time, 2),
            all_scores=label_scores,
        )

    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.get("/")
def root():
    """API Info"""
    return {
        "service": "WhatsApp Text Classifier",
        "version": "1.0.0",
        "model": "mDeBERTa-v3-base-xnli",
        "categories": CANDIDATE_LABELS,
        "endpoints": {
            "classify": "/classify (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET)",
        },
    }
