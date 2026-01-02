FROM python:3.12-slim
LABEL authors="jaret, hinablue<hinablue@gmail.com>"

# 設置環境變數
ENV DEBIAN_FRONTEND=noninteractive \
    # 防止 Python 生成 .pyc 文件
    PYTHONDONTWRITEBYTECODE=1 \
    # 讓 log 立即輸出
    PYTHONUNBUFFERED=1 \
    # 禁用 pip 版本檢查與緩存以減小體積
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # 設定 Node.js 版本
    NODE_MAJOR=23

# 1. 安裝系統依賴 & Node.js
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    wget \
    # build-essential 用於編譯純 Python C 擴展 (非 CUDA)
    build-essential \
    # OpenCV 依賴
    libgl1 \
    libglib2.0-0 \
    # 多媒體處理
    ffmpeg \
    # 監控工具
    htop \
    procps \
    # 其他工具
    openssh-client \
    openssl \
    rsync \
    unzip \
    && \
    # 安裝 Node.js (Debian 方式)
    curl -fsSL https://deb.nodesource.com/setup_${NODE_MAJOR}.x | bash - && \
    apt-get install -y nodejs && \
    # 清理 apt 緩存
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. 預先安裝 PyTorch (利用 Layer Caching)
# 注意：在 slim 鏡像中，PyTorch 必須下載包含 CUDA Runtime 的 wheel (這通常很大)
# 確保 index-url 是正確的，這裡沿用你指定的 cu128 (若無則需降級)
RUN pip install --pre \
    xformers==0.0.33.post2 \
    torch==2.9.1 \
    torchvision==0.24.1 \
    torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128

# 3. 下載程式碼 (Cache Busting)
ARG CACHEBUST=1234
ARG GIT_COMMIT=main

RUN echo "Cache bust: ${CACHEBUST}" && \
    git clone https://github.com/ostris/ai-toolkit.git && \
    cd ai-toolkit && \
    git checkout ${GIT_COMMIT}

WORKDIR /app/ai-toolkit

# 4. 安裝 Python 依賴
RUN pip install -r requirements.txt

# 5. Build UI
WORKDIR /app/ai-toolkit/ui
RUN npm install && \
    npm run build && \
    npm run update_db

# 6. 啟動設定
EXPOSE 8675
WORKDIR /app/ai-toolkit

COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]%
