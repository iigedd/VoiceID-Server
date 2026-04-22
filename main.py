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

# --- OTTIMIZZAZIONE EXTREME ---
torch.set_num_threads(1)
os.environ["LRU_CACHE_CAPACITY"] = "1" # Riduce l'uso della cache per risparmiare RAM

app = FastAPI()

# Carichiamo il modello all'avvio
print("Avvio caricamento modello...")
try:
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"}
    )
    print("Modello caricato con successo!")
except Exception as e:
    print(f"Errore caricamento modello: {e}")

DB_FILE = "voice_db.pkl"

def load_db():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "rb") as f:
                return pickle.load(f)
        except:
            return {}
    return {}

def save_db(db):
    with open(DB_FILE, "wb") as f:
        pickle.dump(db, f)

def get_embedding(file_path):
    signal, fs = librosa.load(file_path, sr=16000)
    # Prendiamo solo i primi 3 secondi per non mandare il server in crash
    signal = signal[:16000*3] 
    tensor = torch.tensor(signal).unsqueeze(0)
    with torch.no_grad():
        embedding = classifier.encode_batch(tensor)
    return embedding.squeeze(0).cpu().numpy()

@app.get("/")
def health():
    return {"status": "online"}

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        emb = get_embedding(temp_path)
        db = load_db()
        
        best_match = "Sconosciuto"
        max_score = 0
        
        for name, saved_emb in db.items():
            score = np.dot(emb, saved_emb.T) / (np.linalg.norm(emb) * np.linalg.norm(saved_emb))
            if score > max_score:
                max_score = float(score)
                best_match = name
        
        os.remove(temp_path)
        return {"person": best_match, "confidence": max_score}
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/learn")
async def learn(name: str = Form(...), file: UploadFile = File(...)):
    temp_path = f"l_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        emb = get_embedding(temp_path)
        db = load_db()
        db[name] = emb
        save_db(db)
        
        os.remove(temp_path)
        return {"message": f"Salvato: {name}"}
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        return JSONResponse(status_code=500, content={"error": str(e)})
