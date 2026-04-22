import io, os, pickle, torch, librosa, uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from speechbrain.inference.speaker import EncoderClassifier

app = FastAPI()
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")

DB_FILE = "voice_db.pkl"
VOTES_FILE = "votes_db.pkl"
SOGLIA_VALIDAZIONE = 3 # Numero di conferme necessarie

def load_db(file):
    if os.path.exists(file):
        with open(file, "rb") as f: return pickle.load(f)
    return {}

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    signal, fs = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    with torch.no_grad():
        emb = classifier.encode_batch(torch.tensor(signal).unsqueeze(0))[0, 0].numpy()
    
    db = load_db(DB_FILE)
    best_match, highest_score = "Sconosciuto", 0.0
    
    for name, saved_emb in db.items():
        score = np.dot(emb, saved_emb) / (np.linalg.norm(emb) * np.linalg.norm(saved_emb))
        if score > highest_score:
            highest_score, best_match = score, name
    
    return {"person": best_match, "confidence": float(highest_score)}

@app.post("/learn")
async def learn(file: UploadFile = File(...), name: str = Form(...)):
    audio_bytes = await file.read()
    signal, fs = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    with torch.no_grad():
        new_emb = classifier.encode_batch(torch.tensor(signal).unsqueeze(0))[0, 0].numpy()
    
    votes = load_db(VOTES_FILE)
    
    # Se il nome non è mai stato suggerito, iniziamo il conteggio
    if name not in votes:
        votes[name] = {"count": 1, "embeddings": [new_emb]}
    else:
        votes[name]["count"] += 1
        votes[name]["embeddings"].append(new_emb)
    
    # Se raggiungiamo la soglia, diventa UFFICIALE
    if votes[name]["count"] >= SOGLIA_VALIDAZIONE:
        db = load_db(DB_FILE)
        # Facciamo la media di tutte le registrazioni ricevute per massima precisione
        db[name] = np.mean(votes[name]["embeddings"], axis=0)
        with open(DB_FILE, "wb") as f: pickle.dump(db, f)
        del votes[name] # Rimuoviamo dai voti perché ora è ufficiale
        msg = f"EVOLUZIONE: {name} è ora un personaggio ufficiale!"
    else:
        msg = f"Voto ricevuto per {name}. Manicano {SOGLIA_VALIDAZIONE - votes[name]['count']} conferme."

    with open(VOTES_FILE, "wb") as f: pickle.dump(votes, f)
    print(msg)
    return {"status": "success", "message": msg}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)