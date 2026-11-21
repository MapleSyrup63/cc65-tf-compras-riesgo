package ml

import (
	"math"
	"math/rand"
	"time"
)

// LogRegConfig parámetros de la regresión logística distribuida.
type LogRegConfig struct {
	Workers   int     // goroutines (en PC4: nodos del clúster)
	Epochs    int     // pasadas sobre el train
	BatchSize int     // tamaño de mini-batch POR worker
	LR        float64 // tasa de aprendizaje (Adam)
	L2        float64 // regularización
	PosWeight float64 // peso de la clase positiva (desbalance)
	Seed      int64
}

// LogReg modelo entrenado + parámetros de estandarización.
type LogReg struct {
	W    []float64 `json:"w"`
	B    float64   `json:"b"`
	Mean []float64 `json:"mean"`
	Std  []float64 `json:"std"`
	Loss []float64 `json:"loss_por_epoca"`
}

func sigmoid(z float64) float64 { return 1 / (1 + math.Exp(-z)) }

// FitScaler calcula media y desviación con map-reduce (sumas parciales por worker).
func FitScaler(d *Dataset, idx []int, workers int) (mean, std []float64) {
	D := d.D
	type part struct{ s, s2 []float64 }
	parts := make([]part, workers)
	ParallelFor(len(idx), workers, func(lo, hi, w int) {
		s, s2 := make([]float64, D), make([]float64, D)
		for _, r := range idx[lo:hi] {
			x := d.Row(r)
			for j := 0; j < D; j++ {
				v := float64(x[j])
				s[j] += v
				s2[j] += v * v
			}
		}
		parts[w] = part{s, s2}
	})
	mean, std = make([]float64, D), make([]float64, D)
	n := float64(len(idx))
	for _, p := range parts {
		if p.s == nil {
			continue
		}
		for j := 0; j < D; j++ {
			mean[j] += p.s[j]
			std[j] += p.s2[j]
		}
	}
	for j := 0; j < D; j++ {
		mean[j] /= n
		v := std[j]/n - mean[j]*mean[j]
		std[j] = math.Sqrt(math.Max(v, 1e-12))
	}
	return
}

// GradMsg es lo que un worker envía al coordinador en cada paso
// (en PC4 viajará serializado por TCP).
type GradMsg struct {
	Worker int
	GradW  []float64
	GradB  float64
	Loss   float64
	N      int
}

// LocalGradient: unidad de trabajo del worker. Calcula el gradiente de la
// log-loss ponderada sobre un mini-batch usando una COPIA de solo lectura de los
// pesos, por lo que no hay condición de carrera con el coordinador.
func LocalGradient(d *Dataset, batch []int, w []float64, b float64, mean, std []float64, posW float64) GradMsg {
	D := d.D
	g := GradMsg{GradW: make([]float64, D), N: len(batch)}
	xs := make([]float64, D)
	for _, r := range batch {
		x := d.Row(r)
		z := b
		for j := 0; j < D; j++ {
			xs[j] = (float64(x[j]) - mean[j]) / std[j]
			z += w[j] * xs[j]
		}
		p := sigmoid(z)
		y := float64(d.Y[r])
		wt := 1.0
		if y == 1 {
			wt = posW
		}
		e := (p - y) * wt
		for j := 0; j < D; j++ {
			g.GradW[j] += e * xs[j]
		}
		g.GradB += e
		p = math.Min(math.Max(p, 1e-12), 1-1e-12)
		g.Loss += -wt * (y*math.Log(p) + (1-y)*math.Log(1-p))
	}
	return g
}

type stepMsg struct {
	w     []float64
	b     float64
	batch []int
}

// TrainLogReg: descenso de gradiente sincrónico data-parallel (patrón
// parameter server / all-reduce):
//
//	coordinador ──stepMsg(pesos, batch)──► worker_k   (k = 1..W, en paralelo)
//	coordinador ◄──────GradMsg─────────── worker_k
//	coordinador: promedia gradientes (barrera) y actualiza pesos con Adam.
//
// Los pesos solo los modifica el coordinador, después de recibir TODOS los
// gradientes del paso: no hay escrituras concurrentes sobre el modelo.
func TrainLogReg(d *Dataset, train []int, cfg LogRegConfig) (*LogReg, []WorkerReport) {
	W, D := cfg.Workers, d.D
	m := &LogReg{W: make([]float64, D)}
	m.Mean, m.Std = FitScaler(d, train, W)

	rng := rand.New(rand.NewSource(cfg.Seed))
	perm := make([]int, len(train))
	copy(perm, train)
	shardSize := (len(perm) + W - 1) / W

	in := make([]chan stepMsg, W)
	out := make(chan GradMsg, W)
	busy := make([]time.Duration, W)
	for k := 0; k < W; k++ {
		in[k] = make(chan stepMsg)
		go func(k int) { // worker persistente
			for s := range in[k] {
				t := time.Now()
				g := LocalGradient(d, s.batch, s.w, s.b, m.Mean, m.Std, cfg.PosWeight)
				g.Worker = k
				busy[k] += time.Since(t) // solo este goroutine escribe busy[k]
				out <- g
			}
		}(k)
	}

	// Adam
	mw, vw := make([]float64, D+1), make([]float64, D+1)
	const b1, b2, eps = 0.9, 0.999, 1e-8
	step := 0
	for ep := 0; ep < cfg.Epochs; ep++ {
		rng.Shuffle(len(perm), func(i, j int) { perm[i], perm[j] = perm[j], perm[i] })
		var epochLoss float64
		var epochN int
		for off := 0; off < shardSize; off += cfg.BatchSize {
			wSnap := append([]float64(nil), m.W...) // snapshot inmutable para los workers
			sent := 0
			for k := 0; k < W; k++ {
				lo := k*shardSize + off
				hi := lo + cfg.BatchSize
				if end := (k + 1) * shardSize; hi > end {
					hi = end
				}
				if hi > len(perm) {
					hi = len(perm)
				}
				if lo >= hi {
					continue
				}
				in[k] <- stepMsg{w: wSnap, b: m.B, batch: perm[lo:hi]}
				sent++
			}
			if sent == 0 {
				break
			}
			// barrera: All-Reduce de gradientes
			gw := make([]float64, D)
			var gb float64
			var n int
			for i := 0; i < sent; i++ {
				g := <-out
				for j := range gw {
					gw[j] += g.GradW[j]
				}
				gb += g.GradB
				n += g.N
				epochLoss += g.Loss
			}
			epochN += n
			step++
			for j := 0; j <= D; j++ {
				var grad float64
				if j < D {
					grad = gw[j]/float64(n) + cfg.L2*m.W[j]
				} else {
					grad = gb / float64(n)
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
	for k := 0; k < W; k++ {
		close(in[k])
	}
	reps := make([]WorkerReport, W)
	for k := 0; k < W; k++ {
		hi := (k + 1) * shardSize
		if hi > len(perm) {
			hi = len(perm)
		}
		reps[k] = WorkerReport{Worker: k, Shard: hi - k*shardSize, Segundos: busy[k].Seconds()}
	}
	return m, reps
}

// PredictRow probabilidad de riesgo para una fila sin estandarizar.
func (m *LogReg) PredictRow(x []float32) float64 {
	z := m.B
	for j, v := range x {
		z += m.W[j] * (float64(v) - m.Mean[j]) / m.Std[j]
	}
	return sigmoid(z)
}

// Predict en paralelo.
func (m *LogReg) Predict(d *Dataset, idx []int, workers int) []float64 {
	out := make([]float64, len(idx))
	ParallelFor(len(idx), workers, func(lo, hi, _ int) {
		for i := lo; i < hi; i++ {
			out[i] = m.PredictRow(d.Row(idx[i]))
		}
	})
	return out
}
