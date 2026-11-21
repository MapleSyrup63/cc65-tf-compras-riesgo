package api

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/http"
	"path/filepath"
	"sync"
	"time"

	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/cluster"
	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/store"
)

var errBusy = errors.New("ya hay un entrenamiento en curso")

// jobManager ejecuta UN entrenamiento a la vez. El mutex protege el mapa de jobs y el
// campo running; cada job se modifica solo bajo ese mutex (update).
type jobManager struct {
	s       *Server
	mu      sync.Mutex
	jobs    map[string]*store.Entrenamiento
	running string
}

func newJobManager(s *Server) *jobManager {
	return &jobManager{s: s, jobs: map[string]*store.Entrenamiento{}}
}

func (j *jobManager) update(id string, fn func(e *store.Entrenamiento)) store.Entrenamiento {
	j.mu.Lock()
	defer j.mu.Unlock()
	e := j.jobs[id]
	fn(e)
	cp := *e
	cp.Log = append([]string(nil), e.Log...)
	return cp
}

func (j *jobManager) persist(e store.Entrenamiento) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := j.s.repo.SaveEntrenamiento(ctx, e); err != nil {
		log.Printf("[api] no se pudo guardar el entrenamiento %s: %v", e.ID, err)
	}
}

// Start lanza el entrenamiento distribuido en segundo plano y devuelve su id.
func (j *jobManager) Start(user string, p cluster.TrainParams) (string, error) {
	if j.s.cl == nil {
		return "", errors.New("la API no tiene nodos configurados (NODES)")
	}
	if err := p.Normalize(); err != nil {
		return "", err
	}
	j.mu.Lock()
	if j.running != "" {
		j.mu.Unlock()
		return "", errBusy
	}
	id := newID()
	e := &store.Entrenamiento{ID: id, Estado: "en_curso", Usuario: user, Inicio: time.Now(), Nodos: len(j.s.cl.Nodes),
		Parametros: map[string]any{"trees": p.Trees, "depth": p.Depth, "sample": p.Sample, "epochs": p.Epochs, "batch": p.Batch}}
	j.jobs[id] = e
	j.running = id
	snap := *e
	j.mu.Unlock()
	j.persist(snap)
	go j.run(id, p)
	return id, nil
}

func (j *jobManager) run(id string, p cluster.TrainParams) {
	logf := func(format string, a ...any) {
		line := fmt.Sprintf("%s %s", time.Now().Format("15:04:05"), fmt.Sprintf(format, a...))
		log.Printf("[train %s] %s", id[:6], line)
		j.update(id, func(e *store.Entrenamiento) { e.Log = append(e.Log, line) })
	}
	res, err := j.s.cl.TrainAll(p, logf)
	if err == nil {
		if err = res.SaveModels(j.s.cfg.ModelsDir); err == nil {
			err = res.SaveReport(filepath.Join(j.s.cfg.ReportsDir, "cluster_train_report.json"))
		}
	}
	if err == nil {
		if m, e2 := LoadModels(j.s.cfg.ModelsDir); e2 != nil {
			err = e2
		} else {
			j.s.models.Store(m) // intercambio atómico: /predict usa ya el modelo nuevo
			logf("modelo recargado: %s", m.Version)
		}
	}
	fin := time.Now()
	snap := j.update(id, func(e *store.Entrenamiento) {
		e.Fin = &fin
		e.Segundos = fin.Sub(e.Inicio).Seconds()
		if err != nil {
			e.Estado, e.Error = "error", err.Error()
			return
		}
		e.Estado = "completado"
		e.AUCRF, e.AUCLR = res.RFEval.Test.AUC, res.LREval.Test.AUC
		for _, t := range res.RFTasks {
			e.Tareas = append(e.Tareas, tarea("rf", t))
		}
		for _, t := range res.LRTasks {
			e.Tareas = append(e.Tareas, tarea("lr", t))
		}
	})
	j.mu.Lock()
	j.running = ""
	j.mu.Unlock()
	j.persist(snap)
}

func tarea(tipo string, t cluster.TaskReport) store.TareaNodo {
	return store.TareaNodo{Tipo: tipo, Tarea: t.Task, Nodo: t.Node, Items: t.Items, SegundosNodo: t.NodeSec,
		SegundosTotal: t.RoundTrip, Reasignaciones: t.Retries}
}

func (j *jobManager) get(id string) (*store.Entrenamiento, bool) {
	j.mu.Lock()
	defer j.mu.Unlock()
	e, ok := j.jobs[id]
	if !ok {
		return nil, false
	}
	cp := *e
	cp.Log = append([]string(nil), e.Log...)
	return &cp, true
}

func (s *Server) handleTrainStart(w http.ResponseWriter, r *http.Request) {
	var p cluster.TrainParams
	if r.ContentLength != 0 && !decode(w, r, &p) {
		return
	}
	id, err := s.jobs.Start(userFrom(r).Sub, p)
	switch {
	case errors.Is(err, errBusy):
		writeErr(w, http.StatusConflict, err.Error())
	case err != nil:
		writeErr(w, http.StatusBadRequest, err.Error())
	default:
		w.Header().Set("Location", "/api/v1/train/"+id)
		writeJSON(w, http.StatusAccepted, map[string]string{"id": id, "estado": "en_curso", "consulta": "/api/v1/train/" + id})
	}
}

func (s *Server) handleTrainGet(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if e, ok := s.jobs.get(id); ok {
		writeJSON(w, http.StatusOK, e)
		return
	}
	e, err := s.repo.GetEntrenamiento(r.Context(), id)
	if errors.Is(err, store.ErrNotFound) {
		writeErr(w, http.StatusNotFound, "entrenamiento no encontrado")
		return
	}
	if err != nil {
		writeErr(w, http.StatusServiceUnavailable, "base de datos no disponible")
		return
	}
	writeJSON(w, http.StatusOK, e)
}

func (s *Server) handleTrainList(w http.ResponseWriter, r *http.Request) {
	es, err := s.repo.ListEntrenamientos(r.Context(), limitParam(r, 20, 100))
	if err != nil {
		writeErr(w, http.StatusServiceUnavailable, "base de datos no disponible")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"total": len(es), "items": es})
}
