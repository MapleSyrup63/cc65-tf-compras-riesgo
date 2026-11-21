package cluster

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/ml"
)

// TrainParams son los hiperparámetros del entrenamiento distribuido completo.
type TrainParams struct {
	Trees  int `json:"trees"`
	Depth  int `json:"depth"`
	Sample int `json:"sample"`
	Epochs int `json:"epochs"`
	Batch  int `json:"batch"`
	Shards int `json:"shards"` // 0 = un shard por nodo
}

// DefaultTrainParams valores usados en la PC3 (adaptados a 3 nodos).
func DefaultTrainParams() TrainParams {
	return TrainParams{Trees: 60, Depth: 14, Sample: 250000, Epochs: 4, Batch: 2048}
}

// Normalize rellena los valores faltantes y valida rangos razonables.
func (p *TrainParams) Normalize() error {
	d := DefaultTrainParams()
	if p.Trees == 0 {
		p.Trees = d.Trees
	}
	if p.Depth == 0 {
		p.Depth = d.Depth
	}
	if p.Sample == 0 {
		p.Sample = d.Sample
	}
	if p.Epochs == 0 {
		p.Epochs = d.Epochs
	}
	if p.Batch == 0 {
		p.Batch = d.Batch
	}
	switch {
	case p.Trees < 1 || p.Trees > 500:
		return fmt.Errorf("trees debe estar entre 1 y 500")
	case p.Depth < 1 || p.Depth > 24:
		return fmt.Errorf("depth debe estar entre 1 y 24")
	case p.Epochs < 1 || p.Epochs > 50:
		return fmt.Errorf("epochs debe estar entre 1 y 50")
	case p.Batch < 16 || p.Batch > 65536:
		return fmt.Errorf("batch debe estar entre 16 y 65536")
	case p.Sample < 1000:
		return fmt.Errorf("sample debe ser >= 1000")
	}
	return nil
}

// TrainOutput es el resultado completo (modelos + reportes + métricas).
type TrainOutput struct {
	Params   TrainParams   `json:"params"`
	Nodes    int           `json:"nodos"`
	Info     HelloResp     `json:"dataset"`
	RF       *ForestResult `json:"-"`
	LR       *LogRegResult `json:"-"`
	Eval     *EvalResult   `json:"-"`
	RFTrain  float64       `json:"rf_segundos"`
	LRTrain  float64       `json:"lr_segundos"`
	Total    float64       `json:"total_segundos"`
	RFTasks  []TaskReport  `json:"rf_tareas"`
	LRTasks  []TaskReport  `json:"lr_tareas"`
	RFEval   *ModelEval    `json:"rf_evaluacion"`
	LREval   *ModelEval    `json:"lr_evaluacion"`
	LRLoss   []float64     `json:"lr_loss_por_epoca"`
	Finished time.Time     `json:"finalizado"`
}

// posRate: tasa de positivos del dataset de la PC3 (171,444 / 2,320,035).
const posRate = 171444.0 / 2320035.0

// TrainAll entrena RF y regresión logística en el clúster y los evalúa.
// logf recibe el avance (puede ser log.Printf o un escritor de estado de job).
func (c *Cluster) TrainAll(p TrainParams, logf func(format string, a ...any)) (*TrainOutput, error) {
	if logf == nil {
		logf = func(string, ...any) {}
	}
	if err := p.Normalize(); err != nil {
		return nil, err
	}
	info, err := c.info()
	if err != nil {
		return nil, err
	}
	W := p.Shards
	if W < 1 {
		W = len(c.Nodes)
	}
	t0 := time.Now()
	logf("clúster: %d nodos, %d shards | dataset N=%d D=%d train=%d", len(c.Nodes), W, info.N, info.D, info.TrainRows)

	cfg := ml.ForestConfig{Trees: p.Trees, Workers: W, SampleSize: p.Sample, MaxDepth: p.Depth, MinLeaf: 50,
		Mtry: int(math.Round(math.Sqrt(float64(info.D)))), Seed: 2026}
	rf, err := c.TrainForest(cfg, W)
	if err != nil {
		return nil, fmt.Errorf("random forest: %w", err)
	}
	logf("[RF] %d árboles en %.2fs", len(rf.Forest.Trees), rf.Seconds)
	for _, r := range rf.Reports {
		logf("     tarea %d → %s: %d árboles, nodo %.2fs, total %.2fs, reasignaciones %d", r.Task, r.Node, r.Items, r.NodeSec, r.RoundTrip, r.Retries)
	}

	lrCfg := ml.LogRegConfig{Epochs: p.Epochs, BatchSize: p.Batch, LR: 0.01, L2: 1e-5, Seed: 2026}
	lr, err := c.TrainLogReg(lrCfg, W, (1-posRate)/posRate)
	if err != nil {
		return nil, fmt.Errorf("regresión logística: %w", err)
	}
	logf("[LR] %d pasos en %.2fs, pérdida por época %v", lr.Steps, lr.Seconds, lr.Model.Loss)

	ev, err := c.Evaluate(rf.Forest, lr.Model, W)
	if err != nil {
		return nil, fmt.Errorf("evaluación: %w", err)
	}
	logf("[RF] test AUC=%.4f AP=%.4f F1=%.4f umbral=%.2f", ev.RF.Test.AUC, ev.RF.Test.AP, ev.RF.Test.F1, ev.RF.Threshold)
	logf("[LR] test AUC=%.4f AP=%.4f F1=%.4f umbral=%.2f", ev.LR.Test.AUC, ev.LR.Test.AP, ev.LR.Test.F1, ev.LR.Threshold)

	return &TrainOutput{Params: p, Nodes: len(c.Nodes), Info: info, RF: rf, LR: lr, Eval: ev,
		RFTrain: rf.Seconds, LRTrain: lr.Seconds, Total: time.Since(t0).Seconds(),
		RFTasks: rf.Reports, LRTasks: lr.Reports, RFEval: ev.RF, LREval: ev.LR, LRLoss: lr.Model.Loss,
		Finished: time.Now()}, nil
}

// SaveModels escribe models/random_forest.json y models/logistic_regression.json
// (mismo formato que cmd/train) de forma atómica (archivo temporal + rename).
func (o *TrainOutput) SaveModels(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	if err := writeJSONAtomic(filepath.Join(dir, "random_forest.json"), map[string]any{
		"binner": o.RF.Binner, "forest": o.RF.Forest, "umbral": o.RFEval.Threshold, "features": o.Info.Names}); err != nil {
		return err
	}
	return writeJSONAtomic(filepath.Join(dir, "logistic_regression.json"), map[string]any{
		"model": o.LR.Model, "umbral": o.LREval.Threshold, "features": o.Info.Names})
}

// SaveReport escribe el reporte de evidencias de la ejecución.
func (o *TrainOutput) SaveReport(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return writeJSONAtomic(path, map[string]any{
		"parametros": o.Params, "nodos": o.Nodes,
		"dataset":             map[string]int{"N": o.Info.N, "D": o.Info.D, "train": o.Info.TrainRows, "val": o.Info.ValRows, "test": o.Info.TestRows},
		"random_forest":       map[string]any{"segundos": o.RFTrain, "tareas": o.RFTasks, "evaluacion": o.RFEval},
		"logistic_regression": map[string]any{"segundos": o.LRTrain, "tareas": o.LRTasks, "loss_por_epoca": o.LRLoss, "evaluacion": o.LREval},
		"total_segundos":      o.Total, "finalizado": o.Finished,
	})
}

func writeJSONAtomic(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}
