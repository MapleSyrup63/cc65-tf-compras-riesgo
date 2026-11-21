package cluster

import (
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/ml"
)

// TaskReport resume una tarea ejecutada por un nodo (evidencia del reparto).
type TaskReport struct {
	Task      int     `json:"tarea"`
	Node      string  `json:"nodo"`
	Items     int     `json:"items"` // árboles o pasos de gradiente
	ShardRows int     `json:"filas_shard"`
	NodeSec   float64 `json:"segundos_nodo"`  // cómputo dentro del nodo
	RoundTrip float64 `json:"segundos_total"` // incluye red y (de)serialización
	Retries   int     `json:"reasignaciones"`
}

// ForestResult es el modelo armado por el coordinador (más el binner para predecir).
type ForestResult struct {
	Forest  *ml.Forest
	Binner  *ml.Binner
	Reports []TaskReport
	Seconds float64
}

// TrainForest distribuye el Random Forest: un shard por tarea y fan-in ordenado.
func (c *Cluster) TrainForest(cfg ml.ForestConfig, shards int) (*ForestResult, error) {
	info, err := c.info()
	if err != nil {
		return nil, err
	}
	if shards < 1 {
		shards = len(c.Nodes)
	}
	t0 := time.Now()
	type out struct {
		resp *TreesResp
		rep  TaskReport
		err  error
	}
	results := make([]out, shards)
	var wg sync.WaitGroup
	for k := 0; k < shards; k++ {
		n := cfg.Trees / shards
		if k < cfg.Trees%shards {
			n++
		}
		wg.Add(1)
		go func(k, n int) {
			defer wg.Done()
			ts := time.Now()
			req := &Request{Op: OpTrees, Trees: &TreesReq{Ref: ShardRef{k, shards}, NTrees: n, Cfg: cfg,
				Seed: cfg.Seed + int64(k)*7919}}
			resp, node, retries, err := c.exec(k, req)
			if err != nil {
				results[k] = out{err: err}
				return
			}
			results[k] = out{resp: resp.Trees, rep: TaskReport{Task: k, Node: node, Items: len(resp.Trees.Trees),
				ShardRows: resp.Trees.ShardRows, NodeSec: resp.Trees.Seconds,
				RoundTrip: time.Since(ts).Seconds(), Retries: retries}}
		}(k, n)
	}
	wg.Wait()

	f := &ml.Forest{D: info.D, Importance: make([]float64, info.D)}
	res := &ForestResult{Forest: f}
	for _, r := range results { // fan-in en orden de shard ⇒ bosque determinístico
		if r.err != nil {
			return nil, r.err
		}
		f.Trees = append(f.Trees, r.resp.Trees...)
		for j, v := range r.resp.Importance {
			f.Importance[j] += v
		}
		res.Reports = append(res.Reports, r.rep)
	}
	var tot float64
	for _, v := range f.Importance {
		tot += v
	}
	for j := range f.Importance {
		f.Importance[j] /= tot
	}
	resp, _, _, err := c.exec(0, &Request{Op: OpBinner})
	if err != nil {
		return nil, err
	}
	res.Binner = resp.Binner
	res.Seconds = time.Since(t0).Seconds()
	return res, nil
}

// LogRegResult es la regresión logística entrenada de forma distribuida.
type LogRegResult struct {
	Model   *ml.LogReg
	Reports []TaskReport
	Steps   int
	Seconds float64
}

// TrainLogReg: SGD síncrono data-parallel con el coordinador como parameter server.
//  1. Scaler: cada nodo manda Σx y Σx² de su shard; el coordinador hace el reduce.
//  2. En cada paso el coordinador envía los pesos a todos los nodos (broadcast), cada
//     uno devuelve el gradiente de su mini-batch y el coordinador espera a TODOS
//     (barrera), suma los gradientes y actualiza los pesos con Adam.
//
// Los pesos solo los modifica el coordinador entre barreras: no hay carreras.
func (c *Cluster) TrainLogReg(cfg ml.LogRegConfig, shards int, posWeight float64) (*LogRegResult, error) {
	info, err := c.info()
	if err != nil {
		return nil, err
	}
	if shards < 1 {
		shards = len(c.Nodes)
	}
	D := info.D
	t0 := time.Now()

	// --- 1) scaler distribuido (map en nodos, reduce aquí)
	sc := make([]*ScalerResp, shards)
	errs := make([]error, shards)
	var wg sync.WaitGroup
	for k := 0; k < shards; k++ {
		wg.Add(1)
		go func(k int) {
			defer wg.Done()
			resp, _, _, err := c.exec(k, &Request{Op: OpScaler, Ref: ShardRef{k, shards}})
			if err != nil {
				errs[k] = err
				return
			}
			sc[k] = resp.Scaler
		}(k)
	}
	wg.Wait()
	mean, std := make([]float64, D), make([]float64, D)
	var n float64
	for k := range sc {
		if errs[k] != nil {
			return nil, errs[k]
		}
		for j := 0; j < D; j++ {
			mean[j] += sc[k].S[j]
			std[j] += sc[k].S2[j]
		}
		n += float64(sc[k].N)
	}
	for j := 0; j < D; j++ {
		mean[j] /= n
		v := std[j]/n - mean[j]*mean[j]
		std[j] = math.Sqrt(math.Max(v, 1e-12))
	}

	// --- 2) SGD síncrono
	m := &ml.LogReg{W: make([]float64, D), Mean: mean, Std: std}
	shardSize := (info.TrainRows + shards - 1) / shards
	stepsPerEpoch := (shardSize + cfg.BatchSize - 1) / cfg.BatchSize
	mw, vw := make([]float64, D+1), make([]float64, D+1)
	const b1, b2, eps = 0.9, 0.999, 1e-8
	reps := make([]TaskReport, shards)
	for k := range reps {
		reps[k].Task = k
	}
	step := 0
	for ep := 0; ep < cfg.Epochs; ep++ {
		var epochLoss float64
		var epochN int
		for s := 0; s < stepsPerEpoch; s++ {
			wSnap := append([]float64(nil), m.W...) // copia inmutable para los nodos
			grads := make([]*GradResp, shards)
			gerr := make([]error, shards)
			for k := 0; k < shards; k++ {
				wg.Add(1)
				go func(k int) {
					defer wg.Done()
					ts := time.Now()
					req := &Request{Op: OpGrad, Grad: &GradReq{Ref: ShardRef{k, shards}, Step: step,
						Batch: cfg.BatchSize, W: wSnap, B: m.B, Mean: mean, Std: std, PosWeight: posWeight,
						Seed: cfg.Seed}}
					resp, node, retries, err := c.exec(k, req)
					if err != nil {
						gerr[k] = err
						return
					}
					grads[k] = resp.Grad
					// cada goroutine escribe SOLO reps[k]
					reps[k].Node, reps[k].Items = node, reps[k].Items+1
					reps[k].ShardRows = shardSize
					reps[k].NodeSec += resp.Grad.Seconds
					reps[k].RoundTrip += time.Since(ts).Seconds()
					reps[k].Retries += retries
				}(k)
			}
			wg.Wait() // barrera: all-reduce síncrono
			gw := make([]float64, D)
			var gb float64
			var cnt int
			for k := 0; k < shards; k++ {
				if gerr[k] != nil {
					return nil, fmt.Errorf("paso %d: %w", step, gerr[k])
				}
				g := grads[k].Grad
				for j := range gw {
					gw[j] += g.GradW[j]
				}
				gb += g.GradB
				cnt += g.N
				epochLoss += g.Loss
			}
			epochN += cnt
			step++
			for j := 0; j <= D; j++ {
				var grad float64
				if j < D {
					grad = gw[j]/float64(cnt) + cfg.L2*m.W[j]
				} else {
					grad = gb / float64(cnt)
				}
				mw[j] = b1*mw[j] + (1-b1)*grad
				vw[j] = b2*vw[j] + (1-b2)*grad*grad
				mh := mw[j] / (1 - math.Pow(b1, float64(step)))
				vh := vw[j] / (1 - math.Pow(b2, float64(step)))
				upd := cfg.LR * mh / (math.Sqrt(vh) + eps)
				if j < D {
					m.W[j] -= upd
				} else {
					m.B -= upd
				}
			}
		}
		m.Loss = append(m.Loss, epochLoss/float64(epochN))
	}
	return &LogRegResult{Model: m, Reports: reps, Steps: step, Seconds: time.Since(t0).Seconds()}, nil
}

// ModelEval son las métricas de un modelo sobre el conjunto de prueba.
type ModelEval struct {
	Threshold float64    `json:"umbral"`
	Test      ml.Metrics `json:"test"`
	AUCAnul   float64    `json:"auc_solo_anulaciones"`
}

// EvalResult agrupa la evaluación distribuida.
type EvalResult struct {
	RF, LR  *ModelEval
	Seconds float64
	Reports []TaskReport
}

// Evaluate reparte la puntuación de validación y prueba entre los nodos (cada uno
// puntúa su porción), junta los puntajes y calcula umbral (en validación) y métricas.
func (c *Cluster) Evaluate(f *ml.Forest, lr *ml.LogReg, shards int) (*EvalResult, error) {
	if shards < 1 {
		shards = len(c.Nodes)
	}
	t0 := time.Now()
	resps := make([]*EvalResp, shards)
	errs := make([]error, shards)
	reps := make([]TaskReport, shards)
	var wg sync.WaitGroup
	for k := 0; k < shards; k++ {
		wg.Add(1)
		go func(k int) {
			defer wg.Done()
			ts := time.Now()
			resp, node, retries, err := c.exec(k, &Request{Op: OpEval, Eval: &EvalReq{Ref: ShardRef{k, shards}, Forest: f, LogReg: lr}})
			if err != nil {
				errs[k] = err
				return
			}
			resps[k] = resp.Eval
			reps[k] = TaskReport{Task: k, Node: node, Items: len(resp.Eval.Test.Y), NodeSec: resp.Eval.Seconds,
				RoundTrip: time.Since(ts).Seconds(), Retries: retries}
		}(k)
	}
	wg.Wait()
	var val, test Scores
	for k, r := range resps {
		if errs[k] != nil {
			return nil, errs[k]
		}
		val.RF, val.LR, val.Y = append(val.RF, r.Val.RF...), append(val.LR, r.Val.LR...), append(val.Y, r.Val.Y...)
		test.RF, test.LR = append(test.RF, r.Test.RF...), append(test.LR, r.Test.LR...)
		test.Y, test.Anul, test.Sob = append(test.Y, r.Test.Y...), append(test.Anul, r.Test.Anul...), append(test.Sob, r.Test.Sob...)
	}
	mk := func(valS, testS []float64) *ModelEval {
		thr := ml.BestF1Threshold(valS, val.Y)
		m := ml.Evaluate(testS, test.Y, test.Anul, test.Sob, thr)
		m.AUC, m.AP = ml.AUCAP(testS, test.Y)
		an, _ := ml.AUCAP(testS, test.Anul)
		return &ModelEval{Threshold: thr, Test: m, AUCAnul: an}
	}
	out := &EvalResult{Reports: reps}
	if f != nil {
		out.RF = mk(val.RF, test.RF)
	}
	if lr != nil {
		out.LR = mk(val.LR, test.LR)
	}
	out.Seconds = time.Since(t0).Seconds()
	return out, nil
}
