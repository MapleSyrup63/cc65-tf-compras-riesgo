package api

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"sync"
	"time"

	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/store"
)

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 800*time.Millisecond)
	defer cancel()
	st := map[string]any{"estado": "ok", "uptime_seg": time.Since(s.started).Seconds(), "modelo_cargado": s.models.Load() != nil}
	code := http.StatusOK
	chk := func(name string, err error) {
		if err != nil {
			st[name] = "caido: " + err.Error()
			st["estado"] = "degradado"
		} else {
			st[name] = "ok"
		}
	}
	chk("mongodb", s.repo.Ping(ctx))
	if s.cache != nil {
		chk("redis", s.cache.Ping(ctx))
	}
	if s.cl != nil {
		sano := len(s.cl.Healthy())
		st["nodos_sanos"] = sano
		st["nodos_total"] = len(s.cl.Nodes)
	}
	writeJSON(w, code, st)
}

type loginReq struct {
	Usuario  string `json:"usuario"`
	Password string `json:"password"`
}

func (s *Server) handleLogin(w http.ResponseWriter, r *http.Request) {
	if !s.loginRL.allow(clientIP(r)) {
		w.Header().Set("Retry-After", "60")
		writeErr(w, http.StatusTooManyRequests, "demasiados intentos; espere un minuto")
		return
	}
	var in loginReq
	if !decode(w, r, &in) {
		return
	}
	u, err := s.repo.GetUsuario(r.Context(), in.Usuario)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			checkPassword(&dummyUser, in.Password) // iguala tiempos
			writeErr(w, http.StatusUnauthorized, "usuario o contraseña incorrectos")
			return
		}
		writeErr(w, http.StatusServiceUnavailable, "base de datos no disponible")
		return
	}
	if !checkPassword(u, in.Password) {
		writeErr(w, http.StatusUnauthorized, "usuario o contraseña incorrectos")
		return
	}
	tok, exp := s.signJWT(u.Usuario, u.Rol)
	writeJSON(w, http.StatusOK, map[string]any{"token": tok, "tipo": "Bearer", "expira": exp, "usuario": u.Usuario, "rol": u.Rol})
}

func (s *Server) handleMe(w http.ResponseWriter, r *http.Request) {
	c := userFrom(r)
	writeJSON(w, http.StatusOK, map[string]any{"usuario": c.Sub, "rol": c.Rol, "expira": time.Unix(c.Exp, 0)})
}

// ---- modelo

func (s *Server) handleModel(w http.ResponseWriter, r *http.Request) {
	m := s.models.Load()
	if m == nil {
		writeErr(w, http.StatusServiceUnavailable, "no hay modelos cargados")
		return
	}
	out := map[string]any{"version": m.Version, "cargado": m.Loaded, "variables": m.Features, "n_variables": len(m.Features),
		"arboles": len(m.RF.Trees), "umbral_rf": m.RFThr, "umbral_lr": m.LRThr}
	if rep := s.loadReport(); rep != nil {
		out["evaluacion"] = rep
	}
	writeJSON(w, http.StatusOK, out)
}

// loadReport lee las métricas de test del último reporte disponible (clúster o PC3).
func (s *Server) loadReport() map[string]any {
	for _, name := range []string{"cluster_train_report.json", "train_report.json"} {
		b, err := os.ReadFile(filepath.Join(s.cfg.ReportsDir, name))
		if err != nil {
			continue
		}
		var rep map[string]map[string]any
		if json.Unmarshal(b, &rep) != nil {
			continue
		}
		out := map[string]any{"fuente": name}
		for _, k := range []string{"random_forest", "logistic_regression"} {
			sub := rep[k]
			if ev, ok := sub["evaluacion"].(map[string]any); ok {
				out[k] = ev["test"]
			} else if t, ok := sub["test"]; ok {
				out[k] = t
			}
		}
		return out
	}
	return nil
}

// ---- predicciones

type predictReq struct {
	Features map[string]float64 `json:"features"`
	MontoPEN *float64           `json:"monto_pen"`
}

// PredictResp es la respuesta de /predict.
type PredictResp struct {
	ScoreRF    float64 `json:"score_rf"`
	ScoreLR    float64 `json:"score_lr"`
	Riesgo     bool    `json:"riesgo"`
	Nivel      string  `json:"nivel"`
	UmbralRF   float64 `json:"umbral_rf"`
	UmbralLR   float64 `json:"umbral_lr"`
	Cache      bool    `json:"cache"`
	LatenciaMS float64 `json:"latencia_ms"`
	Modelo     string  `json:"modelo"`
}

func newID() string {
	b := make([]byte, 12)
	rand.Read(b)
	return hex.EncodeToString(b)
}

// predictOne calcula una predicción usando la caché (Redis) cuando es posible.
func (s *Server) predictOne(ctx context.Context, m *Models, in predictReq, user string) (*PredictResp, error) {
	t0 := time.Now()
	x, err := m.Vector(in.Features, in.MontoPEN)
	if err != nil {
		return nil, err
	}
	key := cacheKey(m.Version, x)
	resp := &PredictResp{UmbralRF: m.RFThr, UmbralLR: m.LRThr, Modelo: m.Version}
	if s.cache != nil {
		cctx, cancel := context.WithTimeout(ctx, 30*time.Millisecond) // Redis nunca debe frenar la API
		b, ok, err := s.cache.Get(cctx, key)
		cancel()
		if err == nil && ok && json.Unmarshal(b, resp) == nil {
			resp.Cache = true
			s.cacheHit.Add(1)
		}
	}
	if !resp.Cache {
		s.cacheMis.Add(1)
		rf, lr := m.Score(x)
		resp.ScoreRF, resp.ScoreLR = rf, lr
		resp.Riesgo = rf >= m.RFThr
		resp.Nivel = m.Nivel(rf)
		if s.cache != nil {
			if b, err := json.Marshal(resp); err == nil {
				cctx, cancel := context.WithTimeout(ctx, 30*time.Millisecond)
				s.cache.Set(cctx, key, b, s.cfg.PredictCacheTTL)
				cancel()
			}
		}
	}
	resp.LatenciaMS = float64(time.Since(t0).Microseconds()) / 1000
	s.enqueue(store.Prediccion{ID: newID(), TS: time.Now(), Usuario: user, Entrada: entradaDe(m, x, in),
		RF: resp.ScoreRF, LR: resp.ScoreLR, Riesgo: resp.Riesgo, Nivel: resp.Nivel, LatenciaMS: resp.LatenciaMS, Cache: resp.Cache})
	return resp, nil
}

// entradaDe guarda solo lo que envió el usuario (no las 47 variables rellenadas con la media).
func entradaDe(m *Models, x []float32, in predictReq) map[string]float64 {
	out := map[string]float64{}
	for k, v := range in.Features {
		out[k] = v
	}
	if in.MontoPEN != nil {
		out["monto_pen"] = *in.MontoPEN
	}
	return out
}

func cacheKey(version string, x []float32) string {
	h := sha256.New()
	h.Write([]byte(version))
	var b [4]byte
	for _, v := range x {
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
		h.Write(b[:])
	}
	return "pred:" + hex.EncodeToString(h.Sum(nil))[:32]
}

// enqueue no bloquea: si el canal está lleno descarta (y lo cuenta) en lugar de frenar /predict.
func (s *Server) enqueue(p store.Prediccion) {
	select {
	case s.predQ <- p:
	default:
		s.dropped.Add(1)
	}
}

func (s *Server) handlePredict(w http.ResponseWriter, r *http.Request) {
	m := s.models.Load()
	if m == nil {
		writeErr(w, http.StatusServiceUnavailable, "no hay modelos cargados; entrene desde POST /api/v1/train")
		return
	}
	var in predictReq
	if !decode(w, r, &in) {
		return
	}
	resp, err := s.predictOne(r.Context(), m, in, userFrom(r).Sub)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handlePredictBatch(w http.ResponseWriter, r *http.Request) {
	m := s.models.Load()
	if m == nil {
		writeErr(w, http.StatusServiceUnavailable, "no hay modelos cargados")
		return
	}
	var in struct {
		Items []predictReq `json:"items"`
	}
	if !decode(w, r, &in) {
		return
	}
	if len(in.Items) == 0 || len(in.Items) > 1000 {
		writeErr(w, http.StatusBadRequest, "items debe tener entre 1 y 1000 elementos")
		return
	}
	// Las predicciones son independientes: se calculan en paralelo con un pool acotado
	// (semáforo) y cada goroutine escribe SOLO su posición out[i] (sin mutex).
	out := make([]any, len(in.Items))
	sem := make(chan struct{}, runtime.NumCPU())
	var wg sync.WaitGroup
	user := userFrom(r).Sub
	for i := range in.Items {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int) {
			defer func() { <-sem; wg.Done() }()
			resp, err := s.predictOne(r.Context(), m, in.Items[i], user)
			if err != nil {
				out[i] = map[string]string{"error": err.Error()}
			} else {
				out[i] = resp
			}
		}(i)
	}
	wg.Wait()
	writeJSON(w, http.StatusOK, map[string]any{"total": len(out), "resultados": out})
}

func limitParam(r *http.Request, def, max int) int {
	n, err := strconv.Atoi(r.URL.Query().Get("limit"))
	if err != nil || n < 1 {
		return def
	}
	if n > max {
		return max
	}
	return n
}

func (s *Server) handlePredictions(w http.ResponseWriter, r *http.Request) {
	c := userFrom(r)
	user := c.Sub
	if c.Rol == "admin" && r.URL.Query().Get("todos") == "true" {
		user = ""
	}
	ps, err := s.repo.ListPredicciones(r.Context(), limitParam(r, 50, 500), user)
	if err != nil {
		writeErr(w, http.StatusServiceUnavailable, "base de datos no disponible")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"total": len(ps), "items": ps})
}

func (s *Server) handleOrdenes(w http.ResponseWriter, r *http.Request) {
	f := store.OrdenFiltro{Departamento: r.URL.Query().Get("departamento"), Limit: limitParam(r, 50, 500)}
	if v := r.URL.Query().Get("riesgo"); v != "" {
		b, err := strconv.ParseBool(v)
		if err != nil {
			writeErr(w, http.StatusBadRequest, "riesgo debe ser true o false")
			return
		}
		f.Riesgo = &b
	}
	if p, err := strconv.Atoi(r.URL.Query().Get("pagina")); err == nil && p > 1 {
		f.Skip = (p - 1) * f.Limit
	}
	items, total, err := s.repo.ListOrdenes(r.Context(), f)
	if err != nil {
		writeErr(w, http.StatusServiceUnavailable, "base de datos no disponible")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"total": total, "items": items})
}

// ---- clúster

func (s *Server) handleCluster(w http.ResponseWriter, r *http.Request) {
	if s.cl == nil {
		writeErr(w, http.StatusServiceUnavailable, "la API no tiene nodos configurados (NODES)")
		return
	}
	const key = "cluster:status"
	if s.cache != nil { // estado cacheado 3 s en Redis: varios paneles no saturan a los nodos
		cctx, cancel := context.WithTimeout(r.Context(), 30*time.Millisecond)
		b, ok, err := s.cache.Get(cctx, key)
		cancel()
		if err == nil && ok {
			w.Header().Set("Content-Type", "application/json; charset=utf-8")
			w.Header().Set("X-Cache", "HIT")
			w.Write(b)
			return
		}
	}
	st := s.cl.Status()
	out := map[string]any{"nodos": st, "sanos": len(s.cl.Healthy()), "total": len(st), "ts": time.Now()}
	if b, err := json.Marshal(out); err == nil && s.cache != nil {
		cctx, cancel := context.WithTimeout(r.Context(), 30*time.Millisecond)
		s.cache.Set(cctx, key, b, 3*time.Second)
		cancel()
	}
	writeJSON(w, http.StatusOK, out)
}

func (s *Server) handleClusterHistory(w http.ResponseWriter, r *http.Request) {
	ms, err := s.repo.ListMetricasNodos(r.Context(), limitParam(r, 100, 1000))
	if err != nil {
		writeErr(w, http.StatusServiceUnavailable, "base de datos no disponible")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"total": len(ms), "items": ms})
}

// ---- métricas de la API

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	rutas := map[string]LatStats{}
	for p, rec := range s.routes {
		if st := rec.stats(); st.Count > 0 {
			rutas[p] = st
		}
	}
	out := map[string]any{"uptime_seg": time.Since(s.started).Seconds(), "rutas": rutas,
		"cache_predict":            map[string]int64{"hits": s.cacheHit.Load(), "misses": s.cacheMis.Load()},
		"predicciones_descartadas": s.dropped.Load(), "goroutines": runtime.NumGoroutine()}
	if m := s.models.Load(); m != nil {
		out["modelo"] = m.Version
	}
	if c, err := s.repo.Conteos(r.Context()); err == nil {
		out["conteos_bd"] = c
	}
	writeJSON(w, http.StatusOK, out)
}
