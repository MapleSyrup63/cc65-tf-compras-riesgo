// Package api implementa el coordinador REST de la PC4: autenticación JWT, predicciones,
// control del clúster de nodos ML (entrenamiento distribuido), persistencia en MongoDB y
// caché en Redis. Usa solo net/http de la biblioteca estándar.
package api

import (
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"runtime/debug"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/cluster"
	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/store"
)

//go:embed openapi.yaml
var openapiYAML []byte

// SeedUser usuario que se crea al arrancar si no existe.
type SeedUser struct{ Usuario, Password, Rol string }

// Config parámetros del servidor.
type Config struct {
	JWTSecret       []byte
	TokenTTL        time.Duration
	ModelsDir       string
	ReportsDir      string
	AllowedOrigins  []string // CORS; "*" permite cualquiera (solo desarrollo)
	PredictCacheTTL time.Duration
	Seed            []SeedUser
}

// Server es el coordinador. Sus campos son de solo lectura tras New, salvo los
// protegidos por atomic o mutex indicados.
type Server struct {
	cfg     Config
	repo    store.Repo
	cache   store.Cache
	cl      *cluster.Cluster // puede ser nil (sin clúster)
	models  atomic.Pointer[Models]
	started time.Time

	routes   map[string]*latRec // se llena en New (solo lectura después)
	loginRL  *rateLimiter
	predQ    chan store.Prediccion
	dropped  atomic.Int64
	cacheHit atomic.Int64
	cacheMis atomic.Int64
	jobs     *jobManager
}

// New crea el servidor, siembra usuarios y carga los modelos si existen.
func New(cfg Config, repo store.Repo, cache store.Cache, cl *cluster.Cluster) (*Server, error) {
	if len(cfg.JWTSecret) < 16 {
		return nil, errors.New("JWT_SECRET debe tener al menos 16 caracteres")
	}
	if cfg.TokenTTL == 0 {
		cfg.TokenTTL = 2 * time.Hour
	}
	if cfg.PredictCacheTTL == 0 {
		cfg.PredictCacheTTL = 10 * time.Minute
	}
	s := &Server{cfg: cfg, repo: repo, cache: cache, cl: cl, started: time.Now(),
		routes: map[string]*latRec{}, loginRL: newRateLimiter(10, time.Minute),
		predQ: make(chan store.Prediccion, 4096)}
	s.jobs = newJobManager(s)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	for _, u := range cfg.Seed {
		if _, err := repo.GetUsuario(ctx, u.Usuario); errors.Is(err, store.ErrNotFound) {
			if err := repo.UpsertUsuario(ctx, NewUsuario(u.Usuario, u.Password, u.Rol)); err != nil {
				return nil, fmt.Errorf("sembrar usuario %s: %w", u.Usuario, err)
			}
			log.Printf("[api] usuario %q (%s) creado", u.Usuario, u.Rol)
		}
	}
	if m, err := LoadModels(cfg.ModelsDir); err != nil {
		log.Printf("[api] sin modelos cargados (%v): entrene desde POST /api/v1/train", err)
	} else {
		s.models.Store(m)
		log.Printf("[api] modelos cargados: %s (%d variables)", m.Version, len(m.Features))
	}
	return s, nil
}

// Start lanza las goroutines de fondo: escritor de predicciones y muestreo de nodos.
func (s *Server) Start(ctx context.Context) {
	go s.predictionWriter(ctx)
	if s.cl != nil {
		go s.nodeSampler(ctx, 15*time.Second)
	}
}

// route registra un manejador medido (latencia por ruta + encabezado X-Latency-Ms).
func (s *Server) route(mux *http.ServeMux, pattern string, h http.HandlerFunc) {
	rec := newLatRec(10000)
	s.routes[pattern] = rec
	mux.HandleFunc(pattern, func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		sw := &statusWriter{ResponseWriter: w, start: start}
		h(sw, r)
		rec.add(float64(time.Since(start).Microseconds()) / 1000)
	})
}

// Handler devuelve el http.Handler completo (rutas + CORS + cabeceras + recuperación de pánicos).
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	const v = "/api/v1"
	s.route(mux, "GET "+v+"/health", s.handleHealth)
	s.route(mux, "POST "+v+"/auth/login", s.handleLogin)
	s.route(mux, "GET "+v+"/me", s.authn(s.handleMe))
	s.route(mux, "GET "+v+"/model", s.authn(s.handleModel))
	s.route(mux, "POST "+v+"/predict", s.authn(s.handlePredict))
	s.route(mux, "POST "+v+"/predict/batch", s.authn(s.handlePredictBatch))
	s.route(mux, "GET "+v+"/predictions", s.authn(s.handlePredictions))
	s.route(mux, "GET "+v+"/ordenes", s.authn(s.handleOrdenes))
	s.route(mux, "GET "+v+"/cluster", s.authn(s.handleCluster))
	s.route(mux, "GET "+v+"/cluster/history", s.authn(s.handleClusterHistory, "admin"))
	s.route(mux, "POST "+v+"/train", s.authn(s.handleTrainStart, "admin"))
	s.route(mux, "GET "+v+"/train", s.authn(s.handleTrainList))
	s.route(mux, "GET "+v+"/train/{id}", s.authn(s.handleTrainGet))
	s.route(mux, "GET "+v+"/metrics", s.authn(s.handleMetrics))
	mux.HandleFunc("GET /openapi.yaml", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/yaml; charset=utf-8")
		w.Write(openapiYAML)
	})
	mux.HandleFunc("GET /docs", handleDocs)
	return s.secure(mux)
}

// secure: recuperación de pánicos, cabeceras de seguridad y CORS.
func (s *Server) secure(next http.Handler) http.Handler {
	allowed := map[string]bool{}
	for _, o := range s.cfg.AllowedOrigins {
		allowed[strings.TrimSpace(o)] = true
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if rec := recover(); rec != nil {
				log.Printf("[api] pánico en %s %s: %v\n%s", r.Method, r.URL.Path, rec, debug.Stack())
				writeErr(w, http.StatusInternalServerError, "error interno")
			}
		}()
		h := w.Header()
		h.Set("X-Content-Type-Options", "nosniff")
		h.Set("X-Frame-Options", "DENY")
		h.Set("Referrer-Policy", "no-referrer")
		if strings.HasPrefix(r.URL.Path, "/api/") {
			h.Set("Cache-Control", "no-store")
		}
		if origin := r.Header.Get("Origin"); origin != "" && (allowed["*"] || allowed[origin]) {
			if allowed["*"] {
				h.Set("Access-Control-Allow-Origin", "*")
			} else {
				h.Set("Access-Control-Allow-Origin", origin)
				h.Add("Vary", "Origin")
			}
			h.Set("Access-Control-Allow-Headers", "Authorization, Content-Type")
			h.Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
			h.Set("Access-Control-Expose-Headers", "X-Latency-Ms")
			h.Set("Access-Control-Max-Age", "600")
		}
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]string{"error": msg})
}

// decode lee un cuerpo JSON de hasta 1 MiB rechazando campos desconocidos.
func decode(w http.ResponseWriter, r *http.Request, v any) bool {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		writeErr(w, http.StatusBadRequest, "JSON inválido: "+err.Error())
		return false
	}
	return true
}

var docsPage = []byte(`<!doctype html><html lang="es"><head><meta charset="utf-8"><title>RiesgoGo API – Swagger</title>
<link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5.17.14/swagger-ui.css"></head>
<body><div id="ui"></div>
<script src="https://unpkg.com/swagger-ui-dist@5.17.14/swagger-ui-bundle.js"></script>
<script>window.ui=SwaggerUIBundle({url:"/openapi.yaml",dom_id:"#ui",persistAuthorization:true});</script>
</body></html>`)

func handleDocs(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(docsPage)
}

// predictionWriter agrupa las predicciones y las inserta en el repo cada 250 ms o cada 200
// elementos, para no sumar latencia de MongoDB a /predict. Un único goroutine consume el canal.
func (s *Server) predictionWriter(ctx context.Context) {
	tick := time.NewTicker(250 * time.Millisecond)
	defer tick.Stop()
	var batch []store.Prediccion
	flush := func() {
		if len(batch) == 0 {
			return
		}
		c, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		if err := s.repo.InsertPredicciones(c, batch); err != nil {
			log.Printf("[api] no se guardaron %d predicciones: %v", len(batch), err)
		}
		cancel()
		batch = batch[:0]
	}
	for {
		select {
		case p := <-s.predQ:
			batch = append(batch, p)
			if len(batch) >= 200 {
				flush()
			}
		case <-tick.C:
			flush()
		case <-ctx.Done():
			flush()
			return
		}
	}
}

// nodeSampler guarda periódicamente el estado de los nodos (entidad metricas_nodos).
func (s *Server) nodeSampler(ctx context.Context, every time.Duration) {
	tick := time.NewTicker(every)
	defer tick.Stop()
	for {
		select {
		case <-tick.C:
			now := time.Now()
			var ms []store.MetricaNodo
			for i, st := range s.cl.Status() {
				m := store.MetricaNodo{ID: fmt.Sprintf("%d-%d", now.UnixNano(), i), TS: now, Nodo: st.Addr,
					Sano: st.Healthy, LatenciaMS: st.LatencyMS}
				if st.Stats != nil {
					m.Requests, m.BusySeg, m.HeapMB, m.Goroutines = st.Stats.Requests, st.Stats.BusySec, st.Stats.HeapMB, st.Stats.Goroutines
				}
				ms = append(ms, m)
			}
			c, cancel := context.WithTimeout(ctx, 5*time.Second)
			s.repo.InsertMetricasNodos(c, ms)
			cancel()
		case <-ctx.Done():
			return
		}
	}
}

var _ sync.Locker = (*sync.Mutex)(nil)
