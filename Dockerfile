# ---------- etapa de compilación ----------
FROM golang:1.24-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/loader ./cmd/loader \
 && CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/train  ./cmd/train

# ---------- imagen final mínima ----------
FROM alpine:3.20
WORKDIR /app
COPY --from=build /out/ /app/bin/
ENV GOMAXPROCS=0
# por defecto: pipeline completo (Parte A) seguido del entrenamiento (Parte B)
CMD ["/bin/sh", "-c", "/app/bin/loader -in /app/data/raw -out /app/data/processed && /app/bin/train -data /app/data/processed/dataset.bin"]
