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

# --- OTTIMIZZAZIONE EXTREME PER RENDER FREE ---
torch.set_num_threads(1)
os.environ["LRU_CACHE_CAPACITY"] = "1" 

app = FastAPI()

# Carichiamo il modello ResNet (più leggero di ECAPA)
print("Avvio caricamento modello LEGGERO...")
try:
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-resnet-voxceleb",
        run_opts={"device": "cpu"}
    )
    print("Modello leggero caricato con successo!")
except Exception as e:
    print(f"Errore critico caricamento modello: {e}")

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
    # Carichiamo l'audio
    signal, fs = librosa.load(file_path, sr=16000)
    # Tagliamo a 3 secondi per evitare picchi di RAM
    signal = signal[:16000*3] 
    tensor = torch.tensor(signal).unsqueeze(0)
    with torch.no_grad():
        embedding = classifier.encode_batch(tensor)
    return embedding.squeeze(0).cpu().numpy()

@app.get("/")
def health():
    return {"status": "online", "model": "resnet-light"}

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
            # Calcolo similarità coseno
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
        return {"message": f"Identità di {name} salvata correttamente."}
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
