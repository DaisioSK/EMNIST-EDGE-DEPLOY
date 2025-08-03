# 使用官方轻量Python镜像
FROM python:3.10-slim

# 避免交互式配置等问题
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖（torch 有时需要 libgl）
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 创建工作目录
WORKDIR /app

# 复制代码和模型文件
COPY . /app

# 安装 Python 依赖
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 声明端口（Gradio 默认8000）
EXPOSE 8000

# 启动命令（gradio 默认运行）
CMD ["python", "app.py"]
