# GPU 지원 PyTorch 베이스 이미지
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY app.py .

# LoRA 어댑터 복사 (필요시)
# 로컬 어댑터를 사용하려면 아래 주석 해제
COPY korsmishing-qlora_kanana ./korsmishing-qlora_kanana

# 환경변수 설정
ENV HF_HOME=/app/hf_cache
ENV HF_HUB_DISABLE_XET=1
ENV HF_HUB_DISABLE_SYMLINKS=1
ENV HF_HUB_ENABLE_TQDM_MULTIPROCESSING=0

# 어댑터 경로 설정 (컨테이너 내 경로)
ENV ADAPTER_DIR=/app/korsmishing-qlora_kanana

# 포트 노출
EXPOSE 8000

# 헬스체크
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# 서버 실행
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
