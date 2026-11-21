# ---------- etapa de compilación ----------
FROM golang:1.24-alpine AS build
WORKDIR /src
COPY go.mod go.sum* ./
COPY . .
# go mod tidy completa go.sum (drivers de MongoDB y Redis) si aún no se confirmó en el repositorio
RUN go mod tidy \
 && for c in loader train node clusterctl api apibench seed; do \
      CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/$c ./cmd/$c || exit 1; \
    done

# ---------- imagen final mínima ----------
FROM alpine:3.20
WORKDIR /app
COPY --from=build /out/ /app/bin/
ENV GOMAXPROCS=0
# por defecto: pipeline de la PC3 (loader → train). En la PC4 el compose define el comando de cada servicio.
CMD ["/bin/sh", "-c", "/app/bin/loader -in /app/data/raw -out /app/data/processed && /app/bin/train -data /app/data/processed/dataset.bin"]
