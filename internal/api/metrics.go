package api

import (
	"fmt"
	"net/http"
	"sort"
	"sync"
	"time"
)

// latRec guarda las últimas N latencias de una ruta (buffer circular) para p50/p95/p99.
type latRec struct {
	mu    sync.Mutex
	buf   []float64
	next  int
	full  bool
	count int64
	sum   float64
	max   float64
}

func newLatRec(n int) *latRec { return &latRec{buf: make([]float64, n)} }

func (l *latRec) add(ms float64) {
	l.mu.Lock()
	l.buf[l.next] = ms
	l.next++
	if l.next == len(l.buf) {
		l.next, l.full = 0, true
	}
	l.count++
	l.sum += ms
	if ms > l.max {
		l.max = ms
	}
	l.mu.Unlock()
}

// LatStats es el resumen de latencias de una ruta.
type LatStats struct {
	Count  int64   `json:"count"`
	MeanMS float64 `json:"mean_ms"`
	P50MS  float64 `json:"p50_ms"`
	P95MS  float64 `json:"p95_ms"`
	P99MS  float64 `json:"p99_ms"`
	MaxMS  float64 `json:"max_ms"`
}

func (l *latRec) stats() LatStats {
	l.mu.Lock()
	n := l.next
	if l.full {
		n = len(l.buf)
	}
	cp := append([]float64(nil), l.buf[:n]...)
	st := LatStats{Count: l.count, MaxMS: l.max}
	if l.count > 0 {
		st.MeanMS = l.sum / float64(l.count)
	}
	l.mu.Unlock()
	if len(cp) == 0 {
		return st
	}
	sort.Float64s(cp)
	q := func(p float64) float64 { return cp[int(p*float64(len(cp)-1))] }
	st.P50MS, st.P95MS, st.P99MS = q(0.50), q(0.95), q(0.99)
	return st
}

// statusWriter agrega el encabezado X-Latency-Ms justo antes de enviar la respuesta
// (útil para las capturas de Postman) y recuerda el código de estado.
type statusWriter struct {
	http.ResponseWriter
	start  time.Time
	status int
}

func (w *statusWriter) WriteHeader(code int) {
	if w.status == 0 {
		w.status = code
		w.Header().Set("X-Latency-Ms", fmt.Sprintf("%.3f", float64(time.Since(w.start).Microseconds())/1000))
	}
	w.ResponseWriter.WriteHeader(code)
}

func (w *statusWriter) Write(b []byte) (int, error) {
	if w.status == 0 {
		w.WriteHeader(http.StatusOK)
	}
	return w.ResponseWriter.Write(b)
}
