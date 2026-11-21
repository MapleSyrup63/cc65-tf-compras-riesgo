package ml

import (
	"math/rand"
	"sort"
)

// Binner discretiza cada feature en hasta MaxBins cuantiles (histogram-based
// trees, como LightGBM/XGBoost-hist). Reduce el costo de buscar el mejor corte
// de O(n log n) a O(n + bins) por nodo.
type Binner struct {
	Edges [][]float32 `json:"edges"` // Edges[f][b] = límite superior del bin b
}

const MaxBins = 32

// FitBinner calcula los cuantiles sobre una muestra de filas de entrenamiento.
func FitBinner(d *Dataset, idx []int, sample int, seed int64) *Binner {
	rng := rand.New(rand.NewSource(seed))
	if sample > len(idx) {
		sample = len(idx)
	}
	rows := make([]int, sample)
	for i := range rows {
		rows[i] = idx[rng.Intn(len(idx))]
	}
	b := &Binner{Edges: make([][]float32, d.D)}
	ParallelFor(d.D, d.D, func(lo, hi, _ int) {
		for f := lo; f < hi; f++ {
			vals := make([]float32, len(rows))
			for i, r := range rows {
				vals[i] = d.X[r*d.D+f]
			}
			sort.Slice(vals, func(i, j int) bool { return vals[i] < vals[j] })
			var edges []float32
			for q := 1; q < MaxBins; q++ {
				v := vals[q*len(vals)/MaxBins]
				if len(edges) == 0 || v > edges[len(edges)-1] {
					edges = append(edges, v)
				}
			}
			if len(edges) == 0 || edges[len(edges)-1] < vals[len(vals)-1] {
				edges = append(edges, vals[len(vals)-1])
			}
			b.Edges[f] = edges
		}
	})
	return b
}

// BinValue devuelve el índice de bin de un valor.
func (b *Binner) BinValue(f int, v float32) uint8 {
	e := b.Edges[f]
	i := sort.Search(len(e), func(i int) bool { return v <= e[i] })
	if i >= len(e) {
		i = len(e) - 1
	}
	return uint8(i)
}

// Transform binariza todo el dataset en paralelo (cada goroutine su rango de filas).
func (b *Binner) Transform(d *Dataset, workers int) []uint8 {
	out := make([]uint8, d.N*d.D)
	ParallelFor(d.N, workers, func(lo, hi, _ int) {
		for i := lo; i < hi; i++ {
			for f := 0; f < d.D; f++ {
				out[i*d.D+f] = b.BinValue(f, d.X[i*d.D+f])
			}
		}
	})
	return out
}
