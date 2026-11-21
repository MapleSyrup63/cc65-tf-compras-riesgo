// Package cluster implementa la Distribución de la PC4: nodos de cómputo ML que
// atienden trabajo por TCP y un coordinador que reparte shards, hace fan-in de
// resultados parciales y tolera la caída de un nodo.
//
// Topología: estrella (coordinador ↔ N nodos). Cada nodo mantiene una conexión TCP
// persistente con el coordinador; entre nodos no hay tráfico (no hay malla).
// Protocolo: solicitud/respuesta síncrona, mensajes encoding/gob sobre TCP.
package cluster

import "github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/ml"

// Operaciones del protocolo.
const (
	OpHello  = "hello"       // metadatos del nodo y del dataset que cargó
	OpPing   = "ping"        // latido
	OpStats  = "stats"       // métricas del nodo (CPU, memoria, solicitudes)
	OpBinner = "binner"      // cuantiles de discretización (idénticos en todos los nodos)
	OpScaler = "scaler"      // sumas parciales para media/desviación (map-reduce)
	OpTrees  = "train_trees" // entrena árboles del Random Forest sobre un shard
	OpGrad   = "gradient"    // gradiente local de la regresión logística
	OpEval   = "eval"        // puntúa su porción de validación y prueba
)

// ShardRef identifica un shard lógico: el shard Shard de NShards.
type ShardRef struct {
	Shard, NShards int
}

// Request es el mensaje coordinador → nodo.
type Request struct {
	Op    string
	Trees *TreesReq
	Grad  *GradReq
	Eval  *EvalReq
	Ref   ShardRef // usado por OpScaler
}

// Response es el mensaje nodo → coordinador.
type Response struct {
	Err    string
	Hello  *HelloResp
	Stats  *StatsResp
	Binner *ml.Binner
	Scaler *ScalerResp
	Trees  *TreesResp
	Grad   *GradResp
	Eval   *EvalResp
}

type HelloResp struct {
	ID           string
	N, D         int
	TrainRows    int
	ValRows      int
	TestRows     int
	Names        []string
	NumCPU       int
	LocalWorkers int
}

type StatsResp struct {
	ID         string
	UptimeSec  float64
	Requests   int64
	Active     int64
	BusySec    float64 // tiempo acumulado de cómputo útil
	Goroutines int
	NumCPU     int
	HeapMB     float64
	TrainRows  int
}

type TreesReq struct {
	Ref    ShardRef
	NTrees int
	Cfg    ml.ForestConfig
	Seed   int64
}

type TreesResp struct {
	Trees      []ml.Tree
	Importance []float64
	ShardRows  int
	Seconds    float64
}

type ScalerResp struct {
	S, S2 []float64
	N     int
}

type GradReq struct {
	Ref       ShardRef
	Step      int
	Batch     int
	W         []float64
	B         float64
	Mean, Std []float64
	PosWeight float64
	Seed      int64
}

type GradResp struct {
	Grad    ml.GradMsg
	Seconds float64
}

// EvalReq lleva el/los modelos entrenados; el nodo puntúa su porción.
type EvalReq struct {
	Ref    ShardRef
	Forest *ml.Forest
	LogReg *ml.LogReg
}

// Scores agrupa puntajes y etiquetas de una porción de datos.
type Scores struct {
	RF, LR       []float64
	Y, Anul, Sob []uint8
}

type EvalResp struct {
	Val, Test Scores
	Seconds   float64
}
