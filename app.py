import os
import re
from typing import List, Tuple, Optional
from contextlib import contextmanager, asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# ✅ peft는 설치 안 된 환경에서도 "base only" 모드로 실행 가능하게 지연 import 처리
PEFT_AVAILABLE = True
try:
    from peft import PeftModel
except Exception:
    PEFT_AVAILABLE = False

# =====================================================
# 0) 환경변수 (Windows/HF 캐시 안정화)
# =====================================================
os.environ["HF_HUB_DISABLE_XET"] = os.environ.get("HF_HUB_DISABLE_XET", "1")
os.environ["HF_HUB_DISABLE_SYMLINKS"] = os.environ.get("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ["HF_HUB_ENABLE_TQDM_MULTIPROCESSING"] = os.environ.get("HF_HUB_ENABLE_TQDM_MULTIPROCESSING", "0")

# HF 캐시 경로 (Windows 기본값)
os.environ["HF_HOME"] = os.environ.get("HF_HOME", r"C:\hf_cache_clean")

# =====================================================
# 1) 기본 설정 (KANANA)
# =====================================================
BASE_MODEL = os.getenv("BASE_MODEL", "kakaocorp/kanana-1.5-2.1b-instruct-2505")

# ✅ 네 실제 어댑터 루트 폴더/체크포인트 경로 기준으로 기본값 수정
# - 환경변수 ADAPTER_DIR이 있으면 그 값을 우선 사용
# - 없으면 아래 기본값을 사용
ADAPTER_DIR = os.getenv(
    "ADAPTER_DIR",
    r".\korsmishing-qlora_kanana"  # ✅ 실제 폴더명으로 수정 (루트)
)

CACHE_DIR = os.getenv("HF_HOME", r"C:\hf_cache_clean")
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN", "").strip() or None

# ✅ 절대경로로 강제 변환 (Windows에서 상대경로/한글경로 이슈 방지)
ADAPTER_DIR = os.path.abspath(os.path.normpath(ADAPTER_DIR))

# 추론 설정
GEN_MAX_NEW_TOKENS = int(os.getenv("GEN_MAX_NEW_TOKENS", "200"))
GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.1"))
GEN_REP_PENALTY = float(os.getenv("GEN_REP_PENALTY", "1.1"))
NO_REPEAT_NGRAM = int(os.getenv("NO_REPEAT_NGRAM", "4"))

# Windows에서 device_map="auto" 문제나면 False 권장
USE_DEVICE_MAP_AUTO = os.getenv("USE_DEVICE_MAP_AUTO", "0") == "1"

# LoRA 사용 여부 (환경변수로 제어)
# - NO_ADAPTER=1 이면 base 모델만 사용
USE_BASE_ONLY = os.getenv("NO_ADAPTER", "0") == "1"

SEED = int(os.getenv("SEED", "42"))
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =====================================================
# 1.5) 어댑터 경로 유틸 (최신 checkpoint 자동 선택)
# =====================================================
def pick_latest_checkpoint(adapter_root: str) -> str:
    """
    adapter_root가
    - 이미 adapter_config.json을 포함하면 그대로 사용
    - 아니면 하위 폴더 중 adapter_config.json 있는 폴더를 찾아 최신(숫자 or mtime) 선택
    """
    adapter_root = os.path.abspath(os.path.normpath(adapter_root))

    # 1) root 자체가 어댑터면 바로 사용
    if os.path.isfile(os.path.join(adapter_root, "adapter_config.json")):
        return adapter_root

    # 2) 하위 폴더 중 adapter_config.json 있는 후보 찾기
    candidates = []
    if os.path.isdir(adapter_root):
        for name in os.listdir(adapter_root):
            cand = os.path.join(adapter_root, name)
            if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "adapter_config.json")):
                candidates.append(cand)

    if not candidates:
        raise RuntimeError(
            f"[ADAPTER] adapter_config.json을 찾을 수 없습니다.\n"
            f"- 주어진 경로: {adapter_root}\n"
            f"- 해결: ADAPTER_DIR을 checkpoint-xxxxx 폴더로 지정하거나, 루트 바로 아래에 adapter_config.json이 있는 폴더가 있는지 확인하세요.\n"
            f"- 예) ADAPTER_DIR=C:\\Users\\m\\스미싱대회\\korsmishing-qlora_kanana\\checkpoint-28600"
        )

    # 3) 숫자(예: checkpoint-28600)가 있으면 숫자 기준으로 최신 선택, 없으면 수정시간(mtime)
    def score(path: str):
        base = os.path.basename(path)
        m = re.search(r"(\d+)", base)
        if m:
            return (1, int(m.group(1)))
        return (0, int(os.path.getmtime(path)))

    best = max(candidates, key=score)
    return best

# =====================================================
# 2) 유틸
# =====================================================
def supports_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 8

@contextmanager
def left_pad(tok):
    old = tok.padding_side
    tok.padding_side = "left"
    try:
        yield
    finally:
        tok.padding_side = old

def build_prompt(text: str, tokenizer) -> str:
    system_msg = "너는 문자를 분석하여 스미싱 여부를 판단하고, 그 이유를 한국어로 설명하는 보안 전문가야."
    user_msg = (
        "다음 $$문자$$를 보고 먼저 $$스미싱 여부$$를 판단한 뒤, 그 이유를 한두 문장으로 제시하세요.\n"
        "형식:\n"
        "$$스미싱 여부$$: (스미싱/정상)\n"
        "$$설명$$: (간단한 설명)\n\n"
        f"$$문자$$\n{text}\n"
        "<답변>"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # ✅ Kanana/Llama 계열은 chat_template 우선
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # ✅ fallback: chat_template 없을 때
    bos = tokenizer.bos_token or ""
    return f"{bos}\n[SYSTEM]\n{system_msg}\n[USER]\n{user_msg}\n[ASSISTANT]\n"

def parse_label_and_expl(text: str) -> Tuple[str, str]:
    """
    출력에서 라벨/설명 파싱
    - $$스미싱 여부$$: (스미싱/정상)
    - $$설명$$: ...
    """
    text = text.replace("[|endofturn|]", "").replace("</s>", "").strip()
    label = None

    smishing_pattern = r"\$\$스미.{0,10}여부\$\$[:：]?\s*(스미싱|정상)"
    m = re.search(smishing_pattern, text, flags=re.IGNORECASE)
    if m:
        label = m.group(1)

    if label is None:
        clean = text.replace(" ", "")
        m2 = re.search(r"스미[싱신링닝핑a-z]*여부[:：]?\s*(스미싱|정상)", clean, flags=re.IGNORECASE)
        if m2:
            label = m2.group(1)

    if label is None:
        head = text[:80]
        has_s = "스미싱" in head
        has_n = "정상" in head
        if has_s and not has_n:
            label = "스미싱"
        elif has_n and not has_s:
            label = "정상"
        elif has_s and has_n:
            label = "정상" if head.rfind("정상") > head.rfind("스미싱") else "스미싱"
        else:
            label = "판단불가"

    exp = text
    m_exp = re.search(r"\$\$설명.{0,5}[:：]?\s*(.+)", text, flags=re.S)
    if m_exp:
        exp = m_exp.group(1).strip()
        exp = re.sub(r"^[```:：\s]+", "", exp)

    return label, exp

# =====================================================
# 3) FastAPI Schema (JSON In/Out)
# =====================================================
class PredictRequest(BaseModel):
    request_id: Optional[str] = None
    text: str

class PredictResult(BaseModel):
    label: str
    explanation: str

class PredictResponse(BaseModel):
    request_id: Optional[str]
    success: bool
    result: PredictResult
    raw_output: str

class BatchItem(BaseModel):
    id: str
    text: str

class BatchRequest(BaseModel):
    requests: List[BatchItem]

class BatchResult(BaseModel):
    id: str
    label: str
    explanation: str

class BatchResponse(BaseModel):
    success: bool
    results: List[BatchResult]

# =====================================================
# 4) 모델 로딩 (lifespan)
# =====================================================
tokenizer = None
model = None
device = None
dtype = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model, device, dtype

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = supports_bf16()

    dtype = (
        torch.bfloat16 if (device == "cuda" and use_bf16)
        else (torch.float16 if device == "cuda" else torch.float32)
    )

    print(f"[INFO] device={device}, dtype={dtype}, base_only={USE_BASE_ONLY}, peft={PEFT_AVAILABLE}")
    print(f"[INFO] BASE_MODEL={BASE_MODEL}")
    print(f"[INFO] ADAPTER_DIR(root->abs)={ADAPTER_DIR}")
    print(f"[INFO] HF_HOME(CACHE_DIR)={CACHE_DIR}")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        cache_dir=CACHE_DIR,
        token=HF_TOKEN,
        use_fast=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # base model
    load_kwargs = dict(
        cache_dir=CACHE_DIR,
        token=HF_TOKEN,
        torch_dtype=dtype,
        trust_remote_code=False,
    )
    if device == "cuda" and USE_DEVICE_MAP_AUTO:
        load_kwargs["device_map"] = "auto"

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **load_kwargs)

    # device_map 안 쓰면 직접 이동
    if device == "cuda" and not USE_DEVICE_MAP_AUTO:
        base_model = base_model.to("cuda")

    # adapter
    if USE_BASE_ONLY:
        print("▶▶▶ [Mode] BASE MODEL ONLY (No Adapter) ◀◀◀")
        model = base_model
    else:
        if not PEFT_AVAILABLE:
            raise RuntimeError(
                "peft가 설치되어 있지 않습니다. "
                "1) python -m pip install peft accelerate\n"
                "또는 2) NO_ADAPTER=1로 base-only 모드로 실행하세요."
            )

        adapter_path = pick_latest_checkpoint(ADAPTER_DIR)
        cfg = os.path.join(adapter_path, "adapter_config.json")
        print(f"[INFO] Using adapter checkpoint: {adapter_path}")
        print(f"[INFO] adapter_config.json exists? {os.path.exists(cfg)} -> {cfg}")

        print(f"▶▶▶ [Mode] Loading LoRA Adapter: {adapter_path} ◀◀◀")
        model = PeftModel.from_pretrained(base_model, adapter_path)

    model.eval()
    print("[INFO] Model ready.")

    yield  # ✅ 앱 실행 구간

# =====================================================
# 5) FastAPI App
# =====================================================
app = FastAPI(
    title="Smishing Detector (Kanana + LoRA)",
    lifespan=lifespan
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "dtype": str(dtype),
        "base_model": BASE_MODEL,
        "adapter_root": None if USE_BASE_ONLY else ADAPTER_DIR,
        "peft_available": PEFT_AVAILABLE
    }

@torch.no_grad()
def generate_one(text: str) -> Tuple[str, str, str]:
    prompt = build_prompt(text, tokenizer)

    with left_pad(tokenizer):
        enc = tokenizer(
            [prompt],
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True,
        )

    enc.pop("token_type_ids", None)
    enc = {k: v.to(model.device) for k, v in enc.items()}

    out = model.generate(
        **enc,
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        do_sample=False,
        temperature=GEN_TEMPERATURE,
        repetition_penalty=GEN_REP_PENALTY,
        no_repeat_ngram_size=NO_REPEAT_NGRAM,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    in_len = enc["input_ids"].shape[1]
    raw = tokenizer.decode(out[0][in_len:], skip_special_tokens=True).strip()
    label, exp = parse_label_and_expl(raw)

    return label, exp, raw

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    label, exp, raw = generate_one(req.text)

    return PredictResponse(
        request_id=req.request_id,
        success=True,
        result=PredictResult(label=label, explanation=exp),
        raw_output=raw
    )

@app.post("/batch_predict", response_model=BatchResponse)
def batch_predict(req: BatchRequest):
    results: List[BatchResult] = []
    for item in req.requests:
        label, exp, _ = generate_one(item.text)
        results.append(BatchResult(id=item.id, label=label, explanation=exp))
    return BatchResponse(success=True, results=results)

# =====================================================
# 실행:
# py -m uvicorn app:app --host 0.0.0.0 --port 8000
# Swagger: http://127.0.0.1:8000/docs
#
# 특정 체크포인트 직접 지정(권장):
# $env:ADAPTER_DIR="C:\Users\m\스미싱대회\korsmishing-qlora_kanana\checkpoint-28600"
# py -m uvicorn app:app --host 0.0.0.0 --port 8000
#
# base-only 실행:
# $env:NO_ADAPTER="1"
# py -m uvicorn app:app --host 0.0.0.0 --port 8000
# =====================================================
