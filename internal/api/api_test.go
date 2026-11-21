package api

import (
	"bytes"
	"context"
	"encoding/json"
	"math"
	"math/rand"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/cluster"
	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/ml"
	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/store"
)

func synth(n int) *ml.Dataset {
	d := 6
	rng := rand.New(rand.NewSource(1))
	ds := &ml.Dataset{N: n, D: d, X: make([]float32, n*d), Y: make([]uint8, n), Anulada: make([]uint8, n),
		Sobrecosto: make([]uint8, n), IsTrain: make([]bool, n),
		Names: []string{"log_monto_pen", "tc_hasta_8uit", "f2", "f3", "f4", "f5"}}
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			ds.X[i*d+j] = float32(rng.NormFloat64())
		}
		z := 1.6*float64(ds.X[i*d]) + 1.1*float64(ds.X[i*d+1]) - 0.6*float64(ds.X[i*d+2]) - 1.8
		if rng.Float64() < 1/(1+math.Exp(-z)) {
			ds.Y[i] = 1
			ds.Anulada[i] = uint8(i % 2)
			ds.Sobrecosto[i] = uint8(1 - i%2)
		}
		ds.IsTrain[i] = i%5 != 0
	}
	return ds
}

type env struct {
	srv   *httptest.Server
	api   *Server
	repo  *store.MemRepo
	cache *store.MemCache
}

func newEnv(t *testing.T, withCluster bool) *env {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	var cl *cluster.Cluster
	if withCluster {
		d := synth(8000)
		var addrs []string
		for i := 0; i < 2; i++ {
			n := cluster.NewNode("n"+string(rune('1'+i)), d, 2)
			ln, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				t.Fatal(err)
			}
			go n.Serve(ctx, ln)
			addrs = append(addrs, ln.Addr().String())
		}
		cl = cluster.NewCluster(addrs)
		if err := cl.Connect(ctx, 10*time.Second); err != nil {
			t.Fatal(err)
		}
	}
	repo, cache := store.NewMemRepo(), store.NewMemCache()
	s, err := New(Config{JWTSecret: []byte("secreto-de-prueba-1234"), ModelsDir: t.TempDir(), ReportsDir: t.TempDir(),
		AllowedOrigins: []string{"http://localhost:5173"},
		Seed:           []SeedUser{{"admin", "admin123", "admin"}, {"ana", "ana123", "analista"}}}, repo, cache, cl)
	if err != nil {
		t.Fatal(err)
	}
	s.Start(ctx)
	srv := httptest.NewServer(s.Handler())
	t.Cleanup(srv.Close)
	return &env{srv, s, repo, cache}
}

func (e *env) do(t *testing.T, method, path, token string, body any) (*http.Response, map[string]any) {
	t.Helper()
	var rd *bytes.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		rd = bytes.NewReader(b)
	} else {
		rd = bytes.NewReader(nil)
	}
	req, _ := http.NewRequest(method, e.srv.URL+path, rd)
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out map[string]any
	json.NewDecoder(resp.Body).Decode(&out)
	return resp, out
}

func (e *env) login(t *testing.T, u, p string) string {
	t.Helper()
	resp, out := e.do(t, "POST", "/api/v1/auth/login", "", map[string]string{"usuario": u, "password": p})
	if resp.StatusCode != 200 {
		t.Fatalf("login %s: %d %v", u, resp.StatusCode, out)
	}
	return out["token"].(string)
}

func TestAutenticacion(t *testing.T) {
	e := newEnv(t, false)
	if r, _ := e.do(t, "POST", "/api/v1/auth/login", "", map[string]string{"usuario": "admin", "password": "mala"}); r.StatusCode != 401 {
		t.Fatalf("contraseña mala: %d", r.StatusCode)
	}
	if r, _ := e.do(t, "POST", "/api/v1/auth/login", "", map[string]string{"usuario": "nadie", "password": "x"}); r.StatusCode != 401 {
		t.Fatalf("usuario inexistente: %d", r.StatusCode)
	}
	tok := e.login(t, "admin", "admin123")
	if r, out := e.do(t, "GET", "/api/v1/me", tok, nil); r.StatusCode != 200 || out["rol"] != "admin" {
		t.Fatalf("me: %d %v", r.StatusCode, out)
	}
	if r, _ := e.do(t, "GET", "/api/v1/me", "", nil); r.StatusCode != 401 {
		t.Fatalf("sin token: %d", r.StatusCode)
	}
	// token manipulado (firma inválida) y token con alg "none"
	if r, _ := e.do(t, "GET", "/api/v1/me", tok+"x", nil); r.StatusCode != 401 {
		t.Fatalf("firma alterada: %d", r.StatusCode)
	}
	none := b64.EncodeToString([]byte(`{"alg":"none","typ":"JWT"}`)) + "." + strings.Split(tok, ".")[1] + "."
	if r, _ := e.do(t, "GET", "/api/v1/me", none, nil); r.StatusCode != 401 {
		t.Fatalf("alg none aceptado: %d", r.StatusCode)
	}
	// token expirado
	ttl := e.api.cfg.TokenTTL
	e.api.cfg.TokenTTL = -time.Minute
	old, _ := e.api.signJWT("admin", "admin")
	e.api.cfg.TokenTTL = ttl
	if r, _ := e.do(t, "GET", "/api/v1/me", old, nil); r.StatusCode != 401 {
		t.Fatalf("token expirado: %d", r.StatusCode)
	}
	// rol insuficiente
	ana := e.login(t, "ana", "ana123")
	if r, _ := e.do(t, "POST", "/api/v1/train", ana, nil); r.StatusCode != 403 {
		t.Fatalf("analista entrenando: %d", r.StatusCode)
	}
	// la contraseña no está en claro en el repo
	u, _ := e.repo.GetUsuario(context.Background(), "admin")
	if strings.Contains(u.Hash+u.Salt, "admin123") || u.Hash == "" {
		t.Fatal("contraseña mal almacenada")
	}
}

func TestLimiteDeIntentosDeLogin(t *testing.T) {
	e := newEnv(t, false)
	var last int
	for i := 0; i < 12; i++ {
		r, _ := e.do(t, "POST", "/api/v1/auth/login", "", map[string]string{"usuario": "admin", "password": "mala"})
		last = r.StatusCode
	}
	if last != 429 {
		t.Fatalf("se esperaba 429 tras 12 intentos, fue %d", last)
	}
}

func TestSeguridadCORSyCabeceras(t *testing.T) {
	e := newEnv(t, false)
	req, _ := http.NewRequest("OPTIONS", e.srv.URL+"/api/v1/predict", nil)
	req.Header.Set("Origin", "http://localhost:5173")
	resp, _ := http.DefaultClient.Do(req)
	if resp.StatusCode != 204 || resp.Header.Get("Access-Control-Allow-Origin") != "http://localhost:5173" {
		t.Fatalf("preflight: %d %v", resp.StatusCode, resp.Header)
	}
	req2, _ := http.NewRequest("GET", e.srv.URL+"/api/v1/health", nil)
	req2.Header.Set("Origin", "http://malicioso.example")
	resp2, _ := http.DefaultClient.Do(req2)
	if resp2.Header.Get("Access-Control-Allow-Origin") != "" {
		t.Fatal("origen no permitido recibió CORS")
	}
	if resp2.Header.Get("X-Content-Type-Options") != "nosniff" || resp2.Header.Get("X-Latency-Ms") == "" {
		t.Fatalf("faltan cabeceras: %v", resp2.Header)
	}
	body := strings.Repeat("a", 2<<20)
	r, _ := e.do(t, "POST", "/api/v1/auth/login", "", map[string]string{"usuario": body, "password": "x"})
	if r.StatusCode != 400 {
		t.Fatalf("cuerpo gigante: %d", r.StatusCode)
	}
}

func TestSwaggerYDocs(t *testing.T) {
	e := newEnv(t, false)
	for path, want := range map[string]string{"/openapi.yaml": "openapi: 3.0.3", "/docs": "swagger-ui"} {
		resp, _ := http.Get(e.srv.URL + path)
		b := new(bytes.Buffer)
		b.ReadFrom(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != 200 || !strings.Contains(b.String(), want) {
			t.Fatalf("%s: %d", path, resp.StatusCode)
		}
	}
}

func TestEntrenarPredecirYCache(t *testing.T) {
	e := newEnv(t, true)
	admin := e.login(t, "admin", "admin123")

	// sin modelo → 503
	if r, _ := e.do(t, "POST", "/api/v1/predict", admin, map[string]any{"monto_pen": 100}); r.StatusCode != 503 {
		t.Fatalf("sin modelo: %d", r.StatusCode)
	}
	// parámetros inválidos → 400
	if r, _ := e.do(t, "POST", "/api/v1/train", admin, map[string]any{"trees": 9999}); r.StatusCode != 400 {
		t.Fatalf("trees inválido: %d", r.StatusCode)
	}
	r, out := e.do(t, "POST", "/api/v1/train", admin, map[string]any{"trees": 10, "depth": 6, "sample": 3000, "epochs": 2, "batch": 128})
	if r.StatusCode != 202 {
		t.Fatalf("train: %d %v", r.StatusCode, out)
	}
	id := out["id"].(string)
	// segundo entrenamiento simultáneo → 409
	if r2, _ := e.do(t, "POST", "/api/v1/train", admin, map[string]any{"trees": 10, "depth": 6, "sample": 3000}); r2.StatusCode != 409 {
		t.Logf("aviso: el primer entrenamiento ya terminó (%d)", r2.StatusCode)
	}
	var job map[string]any
	for i := 0; i < 200; i++ {
		_, job = e.do(t, "GET", "/api/v1/train/"+id, admin, nil)
		if job["estado"] != "en_curso" {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if job["estado"] != "completado" {
		t.Fatalf("entrenamiento: %v", job)
	}
	if job["auc_rf"].(float64) < 0.8 || len(job["tareas"].([]any)) == 0 {
		t.Fatalf("job incompleto: %v", job)
	}

	// /predict ya usa el modelo recién entrenado (sin reiniciar)
	body := map[string]any{"monto_pen": 5000, "features": map[string]float64{"tc_hasta_8uit": 1}}
	r, p1 := e.do(t, "POST", "/api/v1/predict", admin, body)
	if r.StatusCode != 200 || p1["cache"] != false || p1["nivel"] == "" {
		t.Fatalf("predict: %d %v", r.StatusCode, p1)
	}
	_, p2 := e.do(t, "POST", "/api/v1/predict", admin, body)
	if p2["cache"] != true || p1["score_rf"] != p2["score_rf"] {
		t.Fatalf("la segunda consulta debía salir de caché: %v", p2)
	}
	if lat := p1["latencia_ms"].(float64); lat > 100 {
		t.Fatalf("latencia %.2f ms ≥ 100 ms", lat)
	}
	// validaciones
	if r, _ := e.do(t, "POST", "/api/v1/predict", admin, map[string]any{"features": map[string]float64{"no_existe": 1}}); r.StatusCode != 400 {
		t.Fatalf("variable desconocida: %d", r.StatusCode)
	}
	if r, _ := e.do(t, "POST", "/api/v1/predict", admin, map[string]any{"monto_pen": -5}); r.StatusCode != 400 {
		t.Fatalf("monto negativo: %d", r.StatusCode)
	}
	// batch
	items := make([]map[string]any, 50)
	for i := range items {
		items[i] = map[string]any{"monto_pen": float64(100 * (i + 1))}
	}
	r, b := e.do(t, "POST", "/api/v1/predict/batch", admin, map[string]any{"items": items})
	if r.StatusCode != 200 || len(b["resultados"].([]any)) != 50 {
		t.Fatalf("batch: %d %v", r.StatusCode, b)
	}
	// historial persistido de forma asíncrona
	ok := false
	for i := 0; i < 40 && !ok; i++ {
		_, h := e.do(t, "GET", "/api/v1/predictions?limit=500", admin, nil)
		ok = h["total"].(float64) >= 50
		time.Sleep(50 * time.Millisecond)
	}
	if !ok {
		t.Fatal("las predicciones no se persistieron")
	}
	// métricas por ruta
	_, m := e.do(t, "GET", "/api/v1/metrics", admin, nil)
	rutas := m["rutas"].(map[string]any)
	if rutas["POST /api/v1/predict"] == nil || m["cache_predict"].(map[string]any)["hits"].(float64) < 1 {
		t.Fatalf("métricas: %v", m)
	}
	// entrenamiento persistido en el repo
	if _, err := e.repo.GetEntrenamiento(context.Background(), id); err != nil {
		t.Fatal(err)
	}
	// clúster
	r, c := e.do(t, "GET", "/api/v1/cluster", admin, nil)
	if r.StatusCode != 200 || c["sanos"].(float64) != 2 {
		t.Fatalf("cluster: %d %v", r.StatusCode, c)
	}
	_, h := e.do(t, "GET", "/api/v1/health", "", nil)
	if h["estado"] != "ok" || h["nodos_sanos"].(float64) != 2 {
		t.Fatalf("health: %v", h)
	}
}

// Muchas peticiones concurrentes a /predict (con -race): el modelo es inmutable y la
// escritura de historial va por canal, así que no debe haber carreras.
func TestPredictConcurrente(t *testing.T) {
	e := newEnv(t, true)
	admin := e.login(t, "admin", "admin123")
	_, out := e.do(t, "POST", "/api/v1/train", admin, map[string]any{"trees": 6, "depth": 5, "sample": 2000, "epochs": 1, "batch": 128})
	id := out["id"].(string)
	for i := 0; i < 200; i++ {
		if _, j := e.do(t, "GET", "/api/v1/train/"+id, admin, nil); j["estado"] != "en_curso" {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	var wg sync.WaitGroup
	var bad int32
	var mu sync.Mutex
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < 25; i++ {
				r, _ := e.do(t, "POST", "/api/v1/predict", admin, map[string]any{"monto_pen": float64(g*100 + i)})
				if r.StatusCode != 200 {
					mu.Lock()
					bad++
					mu.Unlock()
				}
			}
		}(g)
	}
	wg.Wait()
	if bad != 0 {
		t.Fatalf("%d respuestas fallidas", bad)
	}
}

func TestSinClusterConfigurado(t *testing.T) {
	e := newEnv(t, false)
	admin := e.login(t, "admin", "admin123")
	if r, _ := e.do(t, "GET", "/api/v1/cluster", admin, nil); r.StatusCode != 503 {
		t.Fatalf("cluster sin nodos: %d", r.StatusCode)
	}
	if r, _ := e.do(t, "POST", "/api/v1/train", admin, nil); r.StatusCode != 400 {
		t.Fatalf("train sin nodos: %d", r.StatusCode)
	}
}
