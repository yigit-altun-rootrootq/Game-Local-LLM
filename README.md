# Game-Local-LLM
Make sure to run a local llm model in your game
```python
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
import urllib.parse
import re
import multiprocessing
from huggingface_hub import hf_hub_download
import os

# CPU çekirdek sayısını al
CPU_CORES = multiprocessing.cpu_count()

# Hugging Face model bilgileri
HF_REPO_ID = "CultriX/Lama-DPOlphin-8B-Q3_K_S-GGUF"  # Kendi seçtiğin repo ID'yi gir
HF_MODEL_FILE = "lama-dpolphin-8b-q3_k_s.gguf"  # Model dosya adını kontrol et

# Modeli Hugging Face'den indir
model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_MODEL_FILE)

# GPU hızlandırmalı model yükleme
llm = Llama(
    model_path=model_path,
    n_gpu_layers=30,  # GPU'da çalıştırılacak katman sayısı
    n_threads=CPU_CORES,  # CPU çekirdeklerini kullan
    n_batch=512,  # Daha büyük işlem blokları
    context_size=2048,  # Bağlam uzunluğu artırıldı
    cache=True,  # Önceki istekleri önbelleğe al
    verbose=False  # Gereksiz logları kapat
)

app = FastAPI()

# Token limitleri ve güvenli aralık
MAX_TOKENS = 500  
SAFE_MARGIN = 20   
STOP_TOKENS = [".", "!", "?"]  

class Query(BaseModel):
    question: str

@app.get("/chat/{query}")
async def ask_ai(query: str):
    try:
        query = urllib.parse.unquote(query)
        print(f"Received query: {query}")  
        
        # Matematiksel işlem kontrolü
        math_pattern = r"^(-?\d+(\.\d+)?\s*[\+\-\*/\^]\s*-?\d+(\.\d+)?)$"  
        if re.match(math_pattern, query):  
            try:
                result = eval(query)  
                return str(result)  
            except Exception as e:
                return f"Hata: {str(e)}"

        else:  
            # Yapay zekaya soruyu ilet
            response = llm(
                query, 
                max_tokens=MAX_TOKENS,
                stop=STOP_TOKENS  
            )
            
            # Yanıtı kontrol et
            if isinstance(response, dict) and "choices" in response and len(response["choices"]) > 0:
                answer = response["choices"][0]["text"].strip()
                answer = answer.replace("\n", " ")  

                # Token sayısını göster
                token_count = len(answer.split())  
                print(f"Token count: {token_count}")  

                # Geçersiz yanıtları düzelt
                if len(answer) == 0 or answer in ["=", "None"]:
                    print("Model boş veya geçersiz bir yanıt verdi. Tekrar denenecek...")
                    return await ask_ai(query)  

                # Yanıtı temizle
                answer = re.sub(r'^[\'"]|[\'"]$', '', answer)  
                answer = re.sub(r'print\(.*?\)', '', answer, flags=re.DOTALL)  
                answer = re.sub(r'import .*?\n', '', answer)  
                answer = re.sub(r'def .*?\(.*?\):', '', answer)  

                # Yanıtın düzgün tamamlanmasını sağla
                if token_count >= MAX_TOKENS - SAFE_MARGIN:
                    last_stop = max(answer.rfind("."), answer.rfind("!"), answer.rfind("?"))  
                    if last_stop != -1:  
                        answer = answer[:last_stop+1]  
                    else:  
                        answer = " ".join(answer.split()[:-1]) + "..."  

                return answer  
            else:
                print("Yanıt beklenen formatta değil. Tekrar denenecek...")
                return await ask_ai(query)  
    
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8643)
