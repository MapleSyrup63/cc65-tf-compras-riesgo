package ml

import (
	"math"
	"math/rand"
	"testing"
)

// synthetic genera un problema separable: y = 1 si x0 + 0.5*x1 > 1.
func synthetic(n int, seed int64) *Dataset {
	rng := rand.New(rand.NewSource(seed))
	D := 4
	d := &Dataset{N: n, D: D, X: make([]float32, n*D), Y: make([]uint8, n),
		Anulada: make([]uint8, n), Sobrecosto: make([]uint8, n), IsTrain: make([]bool, n)}
	for i := 0; i < n; i++ {
		for j := 0; j < D; j++ {
			d.X[i*D+j] = float32(rng.Float64() * 2)
		}
		if d.X[i*D]+0.5*d.X[i*D+1] > 1.5 {
			d.Y[i] = 1
		}
		d.IsTrain[i] = i%5 != 0
	}
	return d
}

func TestAUC(t *testing.T) {
	auc, ap := AUCAP([]float64{0.9, 0.8, 0.3, 0.1}, []uint8{1, 1, 0, 0})
	if auc != 1 || ap != 1 {
		t.Fatalf("AUC/AP perfectos esperados, obtuve %v %v", auc, ap)
	}
	auc, _ = AUCAP([]float64{0.1, 0.2, 0.8, 0.9}, []uint8{1, 1, 0, 0})
	if auc != 0 {
		t.Fatalf("AUC=0 esperado, obtuve %v", auc)
	}
}

func TestForestAprende(t *testing.T) {
	d := synthetic(20000, 1)
	train, test := d.Split()
	b := FitBinner(d, train, 5000, 1)
	Xb := b.Transform(d, 4)
	f, reps := TrainForest(Xb, d.Y, d.D, train, ForestConfig{Trees: 12, Workers: 4, SampleSize: 5000, MaxDepth: 8, MinLeaf: 10, Mtry: 2, Seed: 1})
	if len(f.Trees) != 12 || len(reps) != 4 {
		t.Fatalf("esperaba 12 árboles de 4 workers, got %d/%d", len(f.Trees), len(reps))
	}
	y := make([]uint8, len(test))
	for i, r := range test {
		y[i] = d.Y[r]
	}
	auc, _ := AUCAP(f.Predict(Xb, d.D, test, 4), y)
	if auc < 0.97 {
		t.Fatalf("AUC bajo en problema separable: %.3f", auc)
	}
}

// El resultado no debe depender de la planificación de goroutines.
func TestForestDeterministico(t *testing.T) {
	d := synthetic(5000, 2)
	train, _ := d.Split()
	Xb := FitBinner(d, train, 2000, 1).Transform(d, 2)
	cfg := ForestConfig{Trees: 6, Workers: 3, SampleSize: 2000, MaxDepth: 6, MinLeaf: 5, Mtry: 2, Seed: 9}
	a, _ := TrainForest(Xb, d.Y, d.D, train, cfg)
	b, _ := TrainForest(Xb, d.Y, d.D, train, cfg)
	for i := 0; i < 50; i++ {
		if a.PredictRow(Xb[i*d.D:(i+1)*d.D]) != b.PredictRow(Xb[i*d.D:(i+1)*d.D]) {
			t.Fatal("predicciones distintas entre ejecuciones")
		}
	}
}

func TestLogRegAprende(t *testing.T) {
	d := synthetic(20000, 3)
	train, test := d.Split()
	m, _ := TrainLogReg(d, train, LogRegConfig{Workers: 4, Epochs: 5, BatchSize: 256, LR: 0.05, L2: 1e-6, PosWeight: 1, Seed: 1})
	y := make([]uint8, len(test))
	for i, r := range test {
		y[i] = d.Y[r]
	}
	auc, _ := AUCAP(m.Predict(d, test, 4), y)
	if auc < 0.97 {
		t.Fatalf("AUC bajo: %.3f", auc)
	}
	if m.Loss[len(m.Loss)-1] >= m.Loss[0] {
		t.Errorf("la pérdida no disminuye: %v", m.Loss)
	}
}

// El gradiente agregado de W workers debe ser igual al gradiente secuencial.
func TestGradienteDistribuidoIgualSecuencial(t *testing.T) {
	d := synthetic(3000, 4)
	idx := make([]int, d.N)
	for i := range idx {
		idx[i] = i
	}
	mean, std := FitScaler(d, idx, 3)
	w := []float64{0.1, -0.2, 0.3, 0.05}
	full := LocalGradient(d, idx, w, 0.1, mean, std, 2)
	var sum [4]float64
	for k := 0; k < 3; k++ {
		g := LocalGradient(d, idx[k*1000:(k+1)*1000], w, 0.1, mean, std, 2)
		for j := range sum {
			sum[j] += g.GradW[j]
		}
	}
	for j := range sum {
		if math.Abs(sum[j]-full.GradW[j]) > 1e-9 {
			t.Fatalf("gradiente %d: %v vs %v", j, sum[j], full.GradW[j])
		}
	}
}

func TestParallelForCubreTodo(t *testing.T) {
	for _, w := range []int{1, 3, 8, 100} {
		seen := make([]int, 1001)
		ParallelFor(1001, w, func(lo, hi, _ int) {
			for i := lo; i < hi; i++ {
				seen[i]++ // rangos disjuntos: sin carrera
			}
		})
		for i, v := range seen {
			if v != 1 {
				t.Fatalf("workers=%d índice %d visitado %d veces", w, i, v)
			}
		}
	}
}

func BenchmarkForest1Worker(b *testing.B)  { benchForest(b, 1) }
func BenchmarkForest4Workers(b *testing.B) { benchForest(b, 4) }

func benchForest(b *testing.B, w int) {
	d := synthetic(50000, 5)
	train, _ := d.Split()
	Xb := FitBinner(d, train, 5000, 1).Transform(d, 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		TrainForest(Xb, d.Y, d.D, train, ForestConfig{Trees: 8, Workers: w, SampleSize: 5000, MaxDepth: 10, MinLeaf: 20, Mtry: 2, Seed: 1})
	}
}
