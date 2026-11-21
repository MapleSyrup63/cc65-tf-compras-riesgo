package cluster

import (
	"bufio"
	"context"
	"encoding/gob"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"net"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/ml"
)

// Constantes de reproducibilidad: TODOS los nodos derivan exactamente la misma
// partición train/val/test, los mismos bins y los mismos shards. Por eso cualquier
// nodo puede reemplazar a otro (tolerancia a fallos) sin transferir datos.
const (
	valSeed      = 42
	binnerSample = 200000
	binnerSeed   = 7
	shardSeed    = 2026
)

// Node es un nodo de cómputo ML. Tras NewNode su estado es de SOLO LECTURA, por lo que
// las conexiones se atienden en goroutines concurrentes sin mutex (solo contadores atómicos).
type Node struct {
	ID           string
	LocalWorkers int

	d        *ml.Dataset
	xb       []uint8 // dataset discretizado (para el Random Forest)
	binner   *ml.Binner
	perm     []int // train barajado con shardSeed; los shards son porciones contiguas
	val      []int
	test     []int
	started  time.Time
	requests atomic.Int64
	active   atomic.Int64
	busyNS   atomic.Int64

	mu    sync.Mutex // protege ln y conns
	ln    net.Listener
	conns map[net.Conn]struct{}
}

// NewNode prepara el nodo: partición determinística, binning y shards.
func NewNode(id string, d *ml.Dataset, localWorkers int) *Node {
	if localWorkers < 1 {
		localWorkers = 1
	}
	trainAll, test := d.Split()
	rng := rand.New(rand.NewSource(valSeed))
	rng.Shuffle(len(trainAll), func(i, j int) { trainAll[i], trainAll[j] = trainAll[j], trainAll[i] })
	nVal := len(trainAll) / 10
	val := append([]int(nil), trainAll[:nVal]...)
	train := append([]int(nil), trainAll[nVal:]...)
	sort.Ints(train)

	n := &Node{ID: id, LocalWorkers: localWorkers, d: d, val: val, test: test, started: time.Now(),
		conns: map[net.Conn]struct{}{}}
	n.binner = ml.FitBinner(d, train, binnerSample, binnerSeed)
	n.xb = n.binner.Transform(d, localWorkers)
	n.perm = append([]int(nil), train...)
	rng2 := rand.New(rand.NewSource(shardSeed))
	rng2.Shuffle(len(n.perm), func(i, j int) { n.perm[i], n.perm[j] = n.perm[j], n.perm[i] })
	return n
}

// shardRows devuelve las filas de train del shard k de W (porciones contiguas de perm).
func (n *Node) shardRows(ref ShardRef) ([]int, error) {
	if ref.NShards < 1 || ref.Shard < 0 || ref.Shard >= ref.NShards {
		return nil, fmt.Errorf("shard inválido %d/%d", ref.Shard, ref.NShards)
	}
	return slicePart(n.perm, ref), nil
}

func slicePart(a []int, ref ShardRef) []int {
	size := (len(a) + ref.NShards - 1) / ref.NShards
	lo, hi := ref.Shard*size, (ref.Shard+1)*size
	if lo > len(a) {
		lo = len(a)
	}
	if hi > len(a) {
		hi = len(a)
	}
	return a[lo:hi]
}

func (n *Node) hello() *HelloResp {
	return &HelloResp{ID: n.ID, N: n.d.N, D: n.d.D, TrainRows: len(n.perm), ValRows: len(n.val),
		TestRows: len(n.test), Names: n.d.Names, NumCPU: runtime.NumCPU(), LocalWorkers: n.LocalWorkers}
}

func (n *Node) stats() *StatsResp {
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	return &StatsResp{ID: n.ID, UptimeSec: time.Since(n.started).Seconds(), Requests: n.requests.Load(),
		Active: n.active.Load(), BusySec: float64(n.busyNS.Load()) / 1e9, Goroutines: runtime.NumGoroutine(),
		NumCPU: runtime.NumCPU(), HeapMB: float64(ms.HeapAlloc) / (1 << 20), TrainRows: len(n.perm)}
}

// ---------------------------------------------------------------- servidor TCP

// Serve atiende conexiones hasta que ctx se cancela. Una goroutine por conexión.
func (n *Node) Serve(ctx context.Context, ln net.Listener) error {
	n.mu.Lock()
	n.ln = ln
	n.mu.Unlock()
	go func() { <-ctx.Done(); n.Close() }()
	for {
		c, err := ln.Accept()
		if err != nil {
			if ctx.Err() != nil || errors.Is(err, net.ErrClosed) {
				return nil
			}
			return err
		}
		n.mu.Lock()
		n.conns[c] = struct{}{}
		n.mu.Unlock()
		go n.handle(c)
	}
}

// Close cierra el listener y todas las conexiones activas (simula la caída del nodo).
func (n *Node) Close() {
	n.mu.Lock()
	defer n.mu.Unlock()
	if n.ln != nil {
		n.ln.Close()
	}
	for c := range n.conns {
		c.Close()
	}
}

func (n *Node) handle(c net.Conn) {
	defer func() {
		c.Close()
		n.mu.Lock()
		delete(n.conns, c)
		n.mu.Unlock()
	}()
	if tc, ok := c.(*net.TCPConn); ok {
		tc.SetNoDelay(true)
		tc.SetKeepAlive(true)
	}
	dec := gob.NewDecoder(bufio.NewReaderSize(c, 1<<20))
	bw := bufio.NewWriterSize(c, 1<<20)
	enc := gob.NewEncoder(bw)
	for {
		var req Request
		if err := dec.Decode(&req); err != nil {
			if err != io.EOF && !errors.Is(err, net.ErrClosed) {
				log.Printf("[%s] conexión %v cerrada: %v", n.ID, c.RemoteAddr(), err)
			}
			return
		}
		resp := n.dispatch(&req)
		if err := enc.Encode(resp); err != nil {
			log.Printf("[%s] error al responder: %v", n.ID, err)
			return
		}
		if err := bw.Flush(); err != nil {
			return
		}
	}
}

func (n *Node) dispatch(req *Request) (resp *Response) {
	n.requests.Add(1)
	n.active.Add(1)
	t0 := time.Now()
	defer func() {
		n.active.Add(-1)
		if req.Op != OpPing && req.Op != OpStats && req.Op != OpHello {
			n.busyNS.Add(int64(time.Since(t0)))
		}
		if r := recover(); r != nil { // un mensaje malformado no debe tumbar el nodo
			resp = &Response{Err: fmt.Sprintf("panic en %s: %v", req.Op, r)}
		}
	}()
	switch req.Op {
	case OpPing:
		return &Response{}
	case OpHello:
		return &Response{Hello: n.hello()}
	case OpStats:
		return &Response{Stats: n.stats()}
	case OpBinner:
		return &Response{Binner: n.binner}
	case OpScaler:
		return n.doScaler(req)
	case OpTrees:
		return n.doTrees(req)
	case OpGrad:
		return n.doGrad(req)
	case OpEval:
		return n.doEval(req)
	}
	return &Response{Err: "operación desconocida: " + req.Op}
}

// doScaler: sumas parciales Σx y Σx² sobre el shard (fase map; el coordinador hace el reduce).
func (n *Node) doScaler(req *Request) *Response {
	rows, err := n.shardRows(req.Ref)
	if err != nil {
		return &Response{Err: err.Error()}
	}
	D := n.d.D
	type part struct{ s, s2 []float64 }
	parts := make([]part, n.LocalWorkers)
	ml.ParallelFor(len(rows), n.LocalWorkers, func(lo, hi, w int) {
		p := part{make([]float64, D), make([]float64, D)}
		for _, r := range rows[lo:hi] {
			x := n.d.Row(r)
			for j := 0; j < D; j++ {
				v := float64(x[j])
				p.s[j] += v
				p.s2[j] += v * v
			}
		}
		parts[w] = p // cada goroutine escribe SOLO su posición
	})
	out := &ScalerResp{S: make([]float64, D), S2: make([]float64, D), N: len(rows)}
	for _, p := range parts {
		if p.s == nil {
			continue
		}
		for j := 0; j < D; j++ {
			out.S[j] += p.s[j]
			out.S2[j] += p.s2[j]
		}
	}
	return &Response{Scaler: out}
}

// doTrees: bagging sobre el shard. Los árboles del nodo se reparten entre sus
// goroutines locales (paralelismo en dos niveles: entre nodos y dentro del nodo).
func (n *Node) doTrees(req *Request) *Response {
	tr := req.Trees
	if tr == nil {
		return &Response{Err: "falta TreesReq"}
	}
	rows, err := n.shardRows(tr.Ref)
	if err != nil {
		return &Response{Err: err.Error()}
	}
	if len(rows) == 0 || tr.NTrees < 1 {
		return &Response{Err: "shard vacío o sin árboles"}
	}
	t0 := time.Now()
	L := n.LocalWorkers
	if L > tr.NTrees {
		L = tr.NTrees
	}
	type res struct {
		trees []ml.Tree
		imp   []float64
	}
	out := make([]res, L)
	var wg sync.WaitGroup
	for j := 0; j < L; j++ {
		cnt := tr.NTrees / L
		if j < tr.NTrees%L {
			cnt++
		}
		wg.Add(1)
		go func(j, cnt int) {
			defer wg.Done()
			t, imp := ml.TrainTreesOnShard(n.xb, n.d.Y, n.d.D, rows, cnt, tr.Cfg, tr.Seed+int64(j)*104729)
			out[j] = res{t, imp} // cada goroutine escribe SOLO su posición
		}(j, cnt)
	}
	wg.Wait()
	resp := &TreesResp{Importance: make([]float64, n.d.D), ShardRows: len(rows)}
	for _, r := range out { // orden fijo ⇒ resultado determinístico
		resp.Trees = append(resp.Trees, r.trees...)
		for j, v := range r.imp {
			resp.Importance[j] += v
		}
	}
	resp.Seconds = time.Since(t0).Seconds()
	return &Response{Trees: resp}
}

// doGrad: gradiente local de la log-loss sobre un mini-batch muestreado de SU shard.
func (n *Node) doGrad(req *Request) *Response {
	g := req.Grad
	if g == nil {
		return &Response{Err: "falta GradReq"}
	}
	rows, err := n.shardRows(g.Ref)
	if err != nil {
		return &Response{Err: err.Error()}
	}
	if len(rows) == 0 || len(g.W) != n.d.D {
		return &Response{Err: "shard vacío o pesos con dimensión incorrecta"}
	}
	t0 := time.Now()
	rng := rand.New(rand.NewSource(g.Seed + int64(g.Step)*1000003 + int64(g.Ref.Shard)))
	b := g.Batch
	if b > len(rows) {
		b = len(rows)
	}
	batch := make([]int, b)
	for i := range batch {
		batch[i] = rows[rng.Intn(len(rows))]
	}
	gm := ml.LocalGradient(n.d, batch, g.W, g.B, g.Mean, g.Std, g.PosWeight)
	gm.Worker = g.Ref.Shard
	return &Response{Grad: &GradResp{Grad: gm, Seconds: time.Since(t0).Seconds()}}
}

// doEval: puntúa su porción de validación y de prueba con los modelos recibidos.
func (n *Node) doEval(req *Request) *Response {
	e := req.Eval
	if e == nil || (e.Forest == nil && e.LogReg == nil) {
		return &Response{Err: "falta modelo en EvalReq"}
	}
	t0 := time.Now()
	score := func(idx []int) Scores {
		s := Scores{Y: pick(n.d.Y, idx), Anul: pick(n.d.Anulada, idx), Sob: pick(n.d.Sobrecosto, idx)}
		if e.Forest != nil {
			s.RF = e.Forest.Predict(n.xb, n.d.D, idx, n.LocalWorkers)
		}
		if e.LogReg != nil {
			s.LR = e.LogReg.Predict(n.d, idx, n.LocalWorkers)
		}
		return s
	}
	return &Response{Eval: &EvalResp{
		Val:     score(slicePart(n.val, e.Ref)),
		Test:    score(slicePart(n.test, e.Ref)),
		Seconds: time.Since(t0).Seconds(),
	}}
}

func pick(a []uint8, idx []int) []uint8 {
	out := make([]uint8, len(idx))
	for i, r := range idx {
		out[i] = a[r]
	}
	return out
}

var _ = math.Sqrt
