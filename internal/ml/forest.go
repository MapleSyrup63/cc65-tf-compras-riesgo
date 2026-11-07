package ml

import (
	"math/rand"
	"time"
)

// Node de un árbol CART binario. Si Leaf, Prob es P(riesgo=1).
type Node struct {
	Feat  int     `json:"f"`
	Bin   uint8   `json:"b"` // va a la izquierda si bin(x[f]) <= Bin
	Left  int32   `json:"l"`
	Right int32   `json:"r"`
	Leaf  bool    `json:"leaf"`
	Prob  float32 `json:"p"`
}

type Tree struct {
	Nodes []Node `json:"nodes"`
}

// ForestConfig parámetros del Random Forest distribuido.
type ForestConfig struct {
	Trees      int   // total de árboles
	Workers    int   // goroutines (en PC4: nodos del clúster)
	SampleSize int   // filas bootstrap por árbol
	MaxDepth   int   // profundidad máxima
	MinLeaf    int   // mínimo de filas por hoja
	Mtry       int   // features candidatas por nodo (≈ sqrt(D))
	Seed       int64 // semilla
}

// Forest es el ensamble final (lo arma el coordinador).
type Forest struct {
	Trees      []Tree    `json:"trees"`
	D          int       `json:"d"`
	Importance []float64 `json:"importance"`
}

// WorkerReport resume el trabajo de cada worker (evidencia del paralelismo).
type WorkerReport struct {
	Worker   int     `json:"worker"`
	Arboles  int     `json:"arboles"`
	Shard    int     `json:"filas_shard"`
	Segundos float64 `json:"segundos"`
}

type treeBuilder struct {
	X   []uint8
	Y   []uint8
	D   int
	cfg ForestConfig
	rng *rand.Rand
	imp []float64
	cnt [MaxBins]float64
	pos [MaxBins]float64
}

func gini(pos, n float64) float64 {
	if n == 0 {
		return 0
	}
	p := pos / n
	return 2 * p * (1 - p)
}

// build construye el árbol recursivamente sobre idx (se reordena in-place).
func (tb *treeBuilder) build(t *Tree, idx []int, depth int) int32 {
	var pos float64
	for _, i := range idx {
		pos += float64(tb.Y[i])
	}
	n := float64(len(idx))
	me := int32(len(t.Nodes))
	t.Nodes = append(t.Nodes, Node{Leaf: true, Prob: float32(pos / n)})
	if depth >= tb.cfg.MaxDepth || len(idx) < 2*tb.cfg.MinLeaf || pos == 0 || pos == n {
		return me
	}
	parent := gini(pos, n)
	bestGain, bestF, bestB := 1e-7, -1, uint8(0)
	// muestreo de mtry features sin reemplazo (Fisher-Yates parcial)
	feats := tb.rng.Perm(tb.D)[:tb.cfg.Mtry]
	for _, f := range feats {
		tb.cnt, tb.pos = [MaxBins]float64{}, [MaxBins]float64{}
		for _, i := range idx {
			b := tb.X[i*tb.D+f]
			tb.cnt[b]++
			tb.pos[b] += float64(tb.Y[i])
		}
		var lc, lp float64
		for b := 0; b < MaxBins-1; b++ {
			lc += tb.cnt[b]
			lp += tb.pos[b]
			rc := n - lc
			if lc < float64(tb.cfg.MinLeaf) {
				continue
			}
			if rc < float64(tb.cfg.MinLeaf) {
				break
			}
			gain := parent - (lc/n)*gini(lp, lc) - (rc/n)*gini(pos-lp, rc)
			if gain > bestGain {
				bestGain, bestF, bestB = gain, f, uint8(b)
			}
		}
	}
	if bestF < 0 {
		return me
	}
	tb.imp[bestF] += bestGain * n
	// partición in-place
	l, r := 0, len(idx)-1
	for l <= r {
		if tb.X[idx[l]*tb.D+bestF] <= bestB {
			l++
		} else {
			idx[l], idx[r] = idx[r], idx[l]
			r--
		}
	}
	left := tb.build(t, idx[:l], depth+1)
	right := tb.build(t, idx[l:], depth+1)
	t.Nodes[me] = Node{Feat: bestF, Bin: bestB, Left: left, Right: right}
	return me
}

// TrainTreesOnShard es la unidad de trabajo de un worker: entrena `nTrees`
// árboles con bootstrap sobre SU shard de datos. Es una función pura (solo lee
// X/Y y escribe estructuras propias), por lo que es segura en goroutines y
// en la PC4 se ejecutará en un nodo remoto.
func TrainTreesOnShard(X, Y []uint8, D int, shard []int, nTrees int, cfg ForestConfig, seed int64) ([]Tree, []float64) {
	tb := &treeBuilder{X: X, Y: Y, D: D, cfg: cfg, rng: rand.New(rand.NewSource(seed)), imp: make([]float64, D)}
	trees := make([]Tree, 0, nTrees)
	sample := cfg.SampleSize
	if sample > len(shard) {
		sample = len(shard)
	}
	idx := make([]int, sample)
	for t := 0; t < nTrees; t++ {
		for i := range idx { // bootstrap (muestreo con reemplazo)
			idx[i] = shard[tb.rng.Intn(len(shard))]
		}
		var tr Tree
		tb.build(&tr, idx, 0)
		trees = append(trees, tr)
	}
	return trees, tb.imp
}

type treeMsg struct {
	worker int
	trees  []Tree
	imp    []float64
	rep    WorkerReport
}

// TrainForest: patrón coordinador/workers.
//  1. El coordinador baraja el train y lo parte en `Workers` shards (sharding).
//  2. Cada worker (goroutine) entrena Trees/Workers árboles sobre su shard.
//  3. Los árboles regresan por un channel; el coordinador arma el ensamble.
func TrainForest(X, Y []uint8, D int, train []int, cfg ForestConfig) (*Forest, []WorkerReport) {
	rng := rand.New(rand.NewSource(cfg.Seed))
	perm := make([]int, len(train))
	copy(perm, train)
	rng.Shuffle(len(perm), func(i, j int) { perm[i], perm[j] = perm[j], perm[i] })

	W := cfg.Workers
	out := make(chan treeMsg, W)
	shardSize := (len(perm) + W - 1) / W
	for w := 0; w < W; w++ {
		lo, hi := w*shardSize, (w+1)*shardSize
		if hi > len(perm) {
			hi = len(perm)
		}
		n := cfg.Trees / W
		if w < cfg.Trees%W {
			n++
		}
		go func(w int, shard []int, n int) {
			t0 := time.Now()
			trees, imp := TrainTreesOnShard(X, Y, D, shard, n, cfg, cfg.Seed+int64(w)*7919)
			out <- treeMsg{w, trees, imp, WorkerReport{w, n, len(shard), time.Since(t0).Seconds()}}
		}(w, perm[lo:hi], n)
	}
	f := &Forest{D: D, Importance: make([]float64, D)}
	reps := make([]WorkerReport, W)
	byWorker := make([][]Tree, W)
	for i := 0; i < W; i++ { // fan-in
		m := <-out
		byWorker[m.worker] = m.trees
		for j, v := range m.imp {
			f.Importance[j] += v
		}
		reps[m.worker] = m.rep
	}
	for _, ts := range byWorker { // orden determinístico
		f.Trees = append(f.Trees, ts...)
	}
	var tot float64
	for _, v := range f.Importance {
		tot += v
	}
	for j := range f.Importance {
		f.Importance[j] /= tot
	}
	return f, reps
}

// PredictRow promedia la probabilidad de las hojas (votación suave).
func (f *Forest) PredictRow(xb []uint8) float64 {
	var s float64
	for t := range f.Trees {
		nodes := f.Trees[t].Nodes
		k := int32(0)
		for !nodes[k].Leaf {
			if xb[nodes[k].Feat] <= nodes[k].Bin {
				k = nodes[k].Left
			} else {
				k = nodes[k].Right
			}
		}
		s += float64(nodes[k].Prob)
	}
	return s / float64(len(f.Trees))
}

// Predict en paralelo sobre los índices dados.
func (f *Forest) Predict(Xb []uint8, D int, idx []int, workers int) []float64 {
	out := make([]float64, len(idx))
	ParallelFor(len(idx), workers, func(lo, hi, _ int) {
		for i := lo; i < hi; i++ {
			r := idx[i]
			out[i] = f.PredictRow(Xb[r*D : (r+1)*D])
		}
	})
	return out
}
