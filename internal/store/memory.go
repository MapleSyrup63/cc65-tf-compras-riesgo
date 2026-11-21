package store

import (
	"context"
	"sort"
	"strings"
	"sync"
	"time"
)

// MemRepo: Repo en memoria (pruebas y modo degradado cuando MongoDB no está disponible).
// Un único mutex protege todos los mapas/slices.
type MemRepo struct {
	mu       sync.Mutex
	usuarios map[string]Usuario
	entr     map[string]Entrenamiento
	pred     []Prediccion
	metr     []MetricaNodo
	ordenes  []OrdenMuestra
}

func NewMemRepo() *MemRepo {
	return &MemRepo{usuarios: map[string]Usuario{}, entr: map[string]Entrenamiento{}}
}

func (m *MemRepo) Ping(context.Context) error  { return nil }
func (m *MemRepo) Close(context.Context) error { return nil }

func (m *MemRepo) UpsertUsuario(_ context.Context, u Usuario) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.usuarios[u.Usuario] = u
	return nil
}

func (m *MemRepo) GetUsuario(_ context.Context, n string) (*Usuario, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	u, ok := m.usuarios[n]
	if !ok {
		return nil, ErrNotFound
	}
	return &u, nil
}

func (m *MemRepo) SaveEntrenamiento(_ context.Context, e Entrenamiento) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.entr[e.ID] = e
	return nil
}

func (m *MemRepo) GetEntrenamiento(_ context.Context, id string) (*Entrenamiento, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	e, ok := m.entr[id]
	if !ok {
		return nil, ErrNotFound
	}
	return &e, nil
}

func (m *MemRepo) ListEntrenamientos(_ context.Context, limit int) ([]Entrenamiento, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]Entrenamiento, 0, len(m.entr))
	for _, e := range m.entr {
		out = append(out, e)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Inicio.After(out[j].Inicio) })
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	return out, nil
}

func (m *MemRepo) InsertPredicciones(_ context.Context, p []Prediccion) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.pred = append(m.pred, p...)
	return nil
}

func (m *MemRepo) ListPredicciones(_ context.Context, limit int, usuario string) ([]Prediccion, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var out []Prediccion
	for i := len(m.pred) - 1; i >= 0 && (limit <= 0 || len(out) < limit); i-- {
		if usuario == "" || m.pred[i].Usuario == usuario {
			out = append(out, m.pred[i])
		}
	}
	return out, nil
}

func (m *MemRepo) InsertMetricasNodos(_ context.Context, x []MetricaNodo) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.metr = append(m.metr, x...)
	return nil
}

func (m *MemRepo) ListMetricasNodos(_ context.Context, limit int) ([]MetricaNodo, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var out []MetricaNodo
	for i := len(m.metr) - 1; i >= 0 && (limit <= 0 || len(out) < limit); i-- {
		out = append(out, m.metr[i])
	}
	return out, nil
}

func (m *MemRepo) InsertOrdenes(_ context.Context, o []OrdenMuestra) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.ordenes = append(m.ordenes, o...)
	return nil
}

func (m *MemRepo) ListOrdenes(_ context.Context, f OrdenFiltro) ([]OrdenMuestra, int64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var all []OrdenMuestra
	for _, o := range m.ordenes {
		if f.Departamento != "" && !strings.EqualFold(o.Departamento, f.Departamento) {
			continue
		}
		if f.Riesgo != nil && o.Riesgo != *f.Riesgo {
			continue
		}
		all = append(all, o)
	}
	total := int64(len(all))
	if f.Skip > len(all) {
		f.Skip = len(all)
	}
	all = all[f.Skip:]
	if f.Limit > 0 && len(all) > f.Limit {
		all = all[:f.Limit]
	}
	return all, total, nil
}

func (m *MemRepo) Conteos(context.Context) (map[string]int64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return map[string]int64{"usuarios": int64(len(m.usuarios)), "entrenamientos": int64(len(m.entr)),
		"predicciones": int64(len(m.pred)), "metricas_nodos": int64(len(m.metr)), "ordenes_muestra": int64(len(m.ordenes))}, nil
}

// MemCache: Cache en memoria con expiración.
type MemCache struct {
	mu sync.Mutex
	m  map[string]memItem
}

type memItem struct {
	v   []byte
	exp time.Time
}

func NewMemCache() *MemCache { return &MemCache{m: map[string]memItem{}} }

func (c *MemCache) Ping(context.Context) error { return nil }
func (c *MemCache) Close() error               { return nil }

func (c *MemCache) Get(_ context.Context, k string) ([]byte, bool, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	it, ok := c.m[k]
	if !ok || time.Now().After(it.exp) {
		delete(c.m, k)
		return nil, false, nil
	}
	return it.v, true, nil
}

func (c *MemCache) Set(_ context.Context, k string, v []byte, ttl time.Duration) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.m[k] = memItem{append([]byte(nil), v...), time.Now().Add(ttl)}
	return nil
}
