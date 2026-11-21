package cluster

import (
	"context"
	"math"
	"math/rand"
	"net"
	"testing"
	"time"

	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/ml"
)

// synth genera un dataset sintético reproducible (solo para pruebas).
func synth(n, d int) *ml.Dataset {
	rng := rand.New(rand.NewSource(1))
	ds := &ml.Dataset{N: n, D: d, X: make([]float32, n*d), Y: make([]uint8, n), Anulada: make([]uint8, n),
		Sobrecosto: make([]uint8, n), IsTrain: make([]bool, n)}
	for j := 0; j < d; j++ {
		ds.Names = append(ds.Names, "f"+string(rune('0'+j)))
	}
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			ds.X[i*d+j] = float32(rng.NormFloat64())
		}
		z := 1.6*float64(ds.X[i*d]) + 1.1*float64(ds.X[i*d+1]) - 0.6*float64(ds.X[i*d+2]) - 1.8
		if rng.Float64() < 1/(1+math.Exp(-z)) {
			ds.Y[i] = 1
			if i%2 == 0 {
				ds.Anulada[i] = 1
			} else {
				ds.Sobrecosto[i] = 1
			}
		}
		ds.IsTrain[i] = i%5 != 0
	}
	return ds
}

// startNodes levanta k nodos TCP en 127.0.0.1 con puertos libres.
func startNodes(t *testing.T, k int) ([]*Node, []string, context.CancelFunc) {
	t.Helper()
	d := synth(8000, 6)
	ctx, cancel := context.WithCancel(context.Background())
	var nodes []*Node
	var addrs []string
	for i := 0; i < k; i++ {
		n := NewNode("nodo"+string(rune('1'+i)), d, 2)
		ln, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatal(err)
		}
		go n.Serve(ctx, ln)
		nodes = append(nodes, n)
		addrs = append(addrs, ln.Addr().String())
	}
	return nodes, addrs, cancel
}

var rfCfg = ml.ForestConfig{Trees: 12, SampleSize: 2000, MaxDepth: 6, MinLeaf: 20, Mtry: 3, Seed: 2026}
var lrCfg = ml.LogRegConfig{Epochs: 3, BatchSize: 256, LR: 0.05, L2: 1e-5, Seed: 2026}

func connect(t *testing.T, addrs []string) *Cluster {
	t.Helper()
	c := NewCluster(addrs)
	c.Timeout = 30 * time.Second
	if err := c.Connect(context.Background(), 10*time.Second); err != nil {
		t.Fatal(err)
	}
	return c
}

func TestRandomForestDistribuido(t *testing.T) {
	_, addrs, stop := startNodes(t, 3)
	defer stop()
	c := connect(t, addrs)
	res, err := c.TrainForest(rfCfg, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Forest.Trees) != rfCfg.Trees {
		t.Fatalf("árboles=%d, esperado %d", len(res.Forest.Trees), rfCfg.Trees)
	}
	if len(res.Reports) != 3 {
		t.Fatalf("reportes=%d", len(res.Reports))
	}
	ev, err := c.Evaluate(res.Forest, nil, 3)
	if err != nil {
		t.Fatal(err)
	}
	if ev.RF.Test.AUC < 0.80 {
		t.Fatalf("AUC RF=%.3f < 0.80", ev.RF.Test.AUC)
	}
	// determinismo: mismo resultado en una segunda corrida
	res2, _ := c.TrainForest(rfCfg, 3)
	ev2, _ := c.Evaluate(res2.Forest, nil, 3)
	if ev.RF.Test.AUC != ev2.RF.Test.AUC {
		t.Fatalf("no determinístico: %.6f vs %.6f", ev.RF.Test.AUC, ev2.RF.Test.AUC)
	}
}

func TestRegresionLogisticaDistribuida(t *testing.T) {
	_, addrs, stop := startNodes(t, 3)
	defer stop()
	c := connect(t, addrs)
	res, err := c.TrainLogReg(lrCfg, 3, 3.0)
	if err != nil {
		t.Fatal(err)
	}
	if res.Steps == 0 || len(res.Model.Loss) != lrCfg.Epochs {
		t.Fatalf("pasos=%d loss=%v", res.Steps, res.Model.Loss)
	}
	if res.Model.Loss[len(res.Model.Loss)-1] >= res.Model.Loss[0] {
		t.Fatalf("la pérdida no baja: %v", res.Model.Loss)
	}
	ev, err := c.Evaluate(nil, res.Model, 3)
	if err != nil {
		t.Fatal(err)
	}
	if ev.LR.Test.AUC < 0.80 {
		t.Fatalf("AUC LR=%.3f < 0.80", ev.LR.Test.AUC)
	}
}

// El scaler distribuido (map en nodos + reduce) debe coincidir con ml.FitScaler.
func TestScalerDistribuidoIgualSecuencial(t *testing.T) {
	nodes, addrs, stop := startNodes(t, 3)
	defer stop()
	c := connect(t, addrs)
	res, err := c.TrainLogReg(ml.LogRegConfig{Epochs: 1, BatchSize: 64, LR: 0.01, Seed: 1}, 3, 1)
	if err != nil {
		t.Fatal(err)
	}
	mean, std := ml.FitScaler(nodes[0].d, nodes[0].perm, 2)
	for j := range mean {
		if math.Abs(mean[j]-res.Model.Mean[j]) > 1e-9 || math.Abs(std[j]-res.Model.Std[j]) > 1e-9 {
			t.Fatalf("feature %d: media %.9f vs %.9f, std %.9f vs %.9f", j, mean[j], res.Model.Mean[j], std[j], res.Model.Std[j])
		}
	}
}

// Si un nodo cae, sus tareas se reasignan a otro y el entrenamiento termina.
func TestToleranciaAFallos(t *testing.T) {
	nodes, addrs, stop := startNodes(t, 3)
	defer stop()
	c := connect(t, addrs)
	nodes[2].Close() // cae el tercer nodo
	res, err := c.TrainForest(rfCfg, 3)
	if err != nil {
		t.Fatalf("debió tolerar la caída: %v", err)
	}
	if len(res.Forest.Trees) != rfCfg.Trees {
		t.Fatalf("árboles=%d", len(res.Forest.Trees))
	}
	var retries int
	for _, r := range res.Reports {
		retries += r.Retries
		if r.Node == addrs[2] {
			t.Fatalf("una tarea quedó asignada al nodo caído")
		}
	}
	if retries == 0 {
		t.Fatal("se esperaba al menos una reasignación")
	}
	st := c.Status()
	if st[2].Healthy {
		t.Fatal("el nodo caído figura como sano")
	}
}

func TestEstadoDelClusterYMensajeInvalido(t *testing.T) {
	_, addrs, stop := startNodes(t, 2)
	defer stop()
	c := connect(t, addrs)
	for _, s := range c.Status() {
		if !s.Healthy || s.Stats == nil || s.Stats.TrainRows == 0 {
			t.Fatalf("estado inválido: %+v", s)
		}
	}
	// operación desconocida y shard inválido devuelven error, no tumban el nodo
	if _, err := c.Nodes[0].Call(&Request{Op: "nope"}, time.Second); err == nil {
		t.Fatal("se esperaba error")
	}
	if _, err := c.Nodes[0].Call(&Request{Op: OpScaler, Ref: ShardRef{5, 2}}, time.Second); err == nil {
		t.Fatal("se esperaba error por shard inválido")
	}
	if _, err := c.Nodes[0].Call(&Request{Op: OpPing}, time.Second); err != nil {
		t.Fatalf("el nodo no sobrevivió: %v", err)
	}
}
