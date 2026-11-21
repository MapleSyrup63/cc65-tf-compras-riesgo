// api — coordinador REST de RiesgoGo (PC4).
//
// Configuración por variables de entorno (ver docker-compose.yml):
//
//	API_ADDR     :8080                   dirección HTTP
//	JWT_SECRET   (obligatoria, ≥16 car.)  clave de firma de los tokens
//	NODES        node1:9000,node2:9000    nodos ML (TCP); vacío = sin clúster
//	MONGO_URI    mongodb://mongodb:27017  (STORE=memory usa memoria en lugar de MongoDB)
//	MONGO_DB     riesgogo
//	REDIS_ADDR   redis:6379               vacío = sin caché
//	MODELS_DIR   models   REPORTS_DIR  reports
//	CORS_ORIGINS http://localhost:5173    orígenes permitidos (coma); "*" = cualquiera
//	ADMIN_USER / ADMIN_PASS / ANALISTA_USER / ANALISTA_PASS   usuarios iniciales
package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/api"
	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/cluster"
	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/mongostore"
	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/redisstore"
	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/store"
)

func env(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func main() {
	secret := os.Getenv("JWT_SECRET")
	if len(secret) < 16 {
		log.Fatal("JWT_SECRET es obligatoria (mínimo 16 caracteres)")
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// ---- persistencia (MongoDB) con reintentos: en Docker el contenedor puede tardar en estar listo
	var repo store.Repo
	if env("STORE", "mongo") == "memory" {
		log.Printf("[api] STORE=memory: usando almacenamiento en memoria (sin MongoDB)")
		repo = store.NewMemRepo()
	} else {
		uri, db := env("MONGO_URI", "mongodb://mongodb:27017"), env("MONGO_DB", "riesgogo")
		var err error
		for i := 1; i <= 30; i++ {
			cctx, cancel := context.WithTimeout(ctx, 6*time.Second)
			repo, err = mongostore.New(cctx, uri, db)
			cancel()
			if err == nil {
				break
			}
			log.Printf("[api] esperando a MongoDB (%d/30): %v", i, err)
			time.Sleep(2 * time.Second)
		}
		if err != nil {
			log.Fatalf("no se pudo conectar a MongoDB: %v", err)
		}
		log.Printf("[api] MongoDB conectado: %s/%s", uri, db)
	}

	// ---- caché (Redis)
	var cache store.Cache
	if addr := env("REDIS_ADDR", "redis:6379"); addr != "" {
		cache = redisstore.New(addr, os.Getenv("REDIS_PASSWORD"))
		pctx, cancel := context.WithTimeout(ctx, 3*time.Second)
		if err := cache.Ping(pctx); err != nil {
			log.Printf("[api] Redis no responde todavía (%v): la API funciona sin caché hasta que esté disponible", err)
		} else {
			log.Printf("[api] Redis conectado: %s", addr)
		}
		cancel()
	}

	// ---- clúster de nodos ML (los nodos tardan en cargar el dataset: se espera en segundo plano)
	var cl *cluster.Cluster
	if nodes := strings.TrimSpace(os.Getenv("NODES")); nodes != "" {
		cl = cluster.NewCluster(strings.Split(nodes, ","))
		go func() {
			if err := cl.Connect(ctx, 5*time.Minute); err != nil {
				log.Printf("[api] clúster: %v", err)
				return
			}
			log.Printf("[api] clúster listo: %d/%d nodos", len(cl.Healthy()), len(cl.Nodes))
		}()
	}

	cfg := api.Config{
		JWTSecret: []byte(secret), ModelsDir: env("MODELS_DIR", "models"), ReportsDir: env("REPORTS_DIR", "reports"),
		AllowedOrigins: strings.Split(env("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"), ","),
		Seed: []api.SeedUser{
			{Usuario: env("ADMIN_USER", "admin"), Password: env("ADMIN_PASS", "admin123"), Rol: "admin"},
			{Usuario: env("ANALISTA_USER", "analista"), Password: env("ANALISTA_PASS", "analista123"), Rol: "analista"},
		},
	}
	srv, err := api.New(cfg, repo, cache, cl)
	if err != nil {
		log.Fatal(err)
	}
	srv.Start(ctx)

	hs := &http.Server{Addr: env("API_ADDR", ":8080"), Handler: srv.Handler(), ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout: 15 * time.Second, WriteTimeout: 30 * time.Second, IdleTimeout: 60 * time.Second}
	go func() {
		<-ctx.Done()
		sctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		hs.Shutdown(sctx)
	}()
	log.Printf("[api] escuchando en %s  |  Swagger: /docs", hs.Addr)
	if err := hs.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
	log.Printf("[api] detenida")
}
