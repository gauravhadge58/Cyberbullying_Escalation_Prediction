# ── Stage 1: Build static assets ─────────────────────────
FROM node:18-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

# VITE_BACKEND_URL and VITE_WS_URL are injected at build time as Docker build-args
ARG VITE_BACKEND_URL=/api
ENV VITE_BACKEND_URL=$VITE_BACKEND_URL

ARG VITE_WS_URL=/ws
ENV VITE_WS_URL=$VITE_WS_URL

RUN npm run build

# ── Stage 2: Serve via nginx ──────────────────────────────
FROM nginx:alpine

# Copy built assets to nginx web root
COPY --from=builder /app/dist /usr/share/nginx/html

# Use the existing project nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
