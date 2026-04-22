import os
import torch
import speechbrain as sb
from speechbrain.inference.speaker import EncoderClassifier
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import shutil
import pickle
import librosa
import numpy as np

# --- OTTIMIZZAZIONE PER RENDER FREE (512MB RAM) ---
torch.set_num_threads(1) # Risparmia CPU e RAM
device = "cpu"

app = FastAPI()

# Caricamento del modello all'avvio (più lento il boot, più veloce il riconoscimento)
print("Caricamento modello AI...")
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)

# Percorso database
DB_FILE = "voice_db.pkl"

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_db(db):
    with open(DB_FILE, "wb") as f:
        pickle.dump(db, f)

def get_embedding(file_path):
    """Estrae l'impronta vocale dal file audio"""
    signal, fs = librosa.load(file_path, sr=16000)
    tensor = torch.tensor(signal).unsqueeze(0)
    with torch.no_grad():
        embedding = classifier.encode_batch(tensor)
    return embedding.squeeze(0).cpu().numpy()

@app.get("/")
def health_check():
    return {"status": "online", "message": "VoiceID Server is running"}

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        current_embedding = get_embedding(temp_path)
        db = load_db()
        
        best_match = "Sconosciuto"
        max_score = 0
        
        for name, saved_embedding in db.items():
            # Calcolo somiglianza tra i vettori vocali
            score = np.dot(current_embedding, saved_embedding.T) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(saved_embedding)
            )
            if score > max_score:
                max_score = float(score)
                best_match = name
        
        os.remove(temp_path)
        return {"person": best_match, "confidence": max_score}
    
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/learn")
async def learn(name: String = Form(...), file: UploadFile = File(...)):
    temp_path = f"learn_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        embedding = get_embedding(temp_path)
        db = load_db()
        db[name] = embedding
        save_db(db)
        
        os.remove(temp_path)
        return {"message": f"Identità di {name} convalidata e salvata."}
    
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
