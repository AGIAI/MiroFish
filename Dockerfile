# ---- Stage 1: Build frontend static assets ----
FROM node:18-slim AS frontend-build

WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --ignore-scripts
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Production image ----
FROM python:3.11-slim

# Install system deps required by some Python packages
RUN apt-get update \
  && apt-get install -y --no-install-recommends tini \
  && rm -rf /var/lib/apt/lists/*

# Copy uv from official image
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

# Create non-root user
RUN useradd --create-home --shell /bin/bash mirofish
WORKDIR /app

# Install Python dependencies (layer cached)
COPY backend/pyproject.toml backend/uv.lock ./backend/
RUN cd backend && uv sync --frozen

# Copy backend source
COPY backend/ ./backend/

# Copy pre-built frontend
COPY --from=frontend-build /build/frontend/dist ./frontend/dist

# Create upload directories
RUN mkdir -p backend/uploads/projects backend/uploads/simulations backend/uploads/reports \
  && chown -R mirofish:mirofish /app

USER mirofish

EXPOSE 5001

# Use tini as init process (proper signal forwarding to child processes)
ENTRYPOINT ["tini", "--"]

# Production: serve with gunicorn (2 workers, 120s timeout matching LLM call timeout)
CMD ["sh", "-c", "cd backend && uv run gunicorn --bind 0.0.0.0:5001 --workers 2 --timeout 120 --access-logfile - 'app:create_app()'"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5001/health')" || exit 1
