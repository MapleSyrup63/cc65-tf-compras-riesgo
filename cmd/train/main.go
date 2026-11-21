// train — Parte B: entrenamiento paralelo de Random Forest y Regresión Logística.
//
// Uso:
//
//	go run ./cmd/train -data data/processed/dataset.bin
//	go run ./cmd/train -data data/processed/dataset.bin -bench   # speedup 1,2,4,8,16 workers
package main

import (
	"encoding/json"
	"flag"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"time"

	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/ml"
)

func writeJSON(path string, v any) {
	b, _ := json.MarshalIndent(v, "", "  ")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		log.Fatal(err)
	}
}

func pick(a []uint8, idx []int) []uint8 {
	out := make([]uint8, len(idx))
	for i, r := range idx {
		out[i] = a[r]
	}
	return out
}

type impacto struct {
	OrdenesTest          int     `json:"ordenes_test"`
	OrdenesSenaladas     int     `json:"ordenes_senaladas"`
	PctSenaladas         float64 `json:"pct_senaladas"`
	MontoTestPEN         float64 `json:"monto_test_pen"`
	MontoSenaladoPEN     float64 `json:"monto_senalado_pen"`
	MontoRiesgoRealPEN   float64 `json:"monto_riesgo_real_pen"`
	MontoRiesgoDetectado float64 `json:"monto_riesgo_detectado_pen"`
	PctMontoDetectado    float64 `json:"pct_monto_riesgo_detectado"`
	// Revisión focalizada: auditar solo el top 10 % de órdenes con mayor score.
	Top10PctPrecision float64 `json:"top10_precision"`
	Top10PctRecall    float64 `json:"top10_recall"`
	Lift10            float64 `json:"lift_top10"`
}

func calcImpacto(d *ml.Dataset, test []int, scores []float64, thr float64) impacto {
	var im impacto
	im.OrdenesTest = len(test)
	var P int
	for i, r := range test {
		monto := math.Expm1(float64(d.X[r*d.D])) // feature 0 = log1p(monto_pen)
		im.MontoTestPEN += monto
		y := d.Y[r] == 1
		if y {
			P++
			im.MontoRiesgoRealPEN += monto
		}
		if scores[i] >= thr {
			im.OrdenesSenaladas++
			im.MontoSenaladoPEN += monto
			if y {
				im.MontoRiesgoDetectado += monto
			}
		}
	}
	im.PctSenaladas = float64(im.OrdenesSenaladas) / float64(len(test))
	im.PctMontoDetectado = im.MontoRiesgoDetectado / im.MontoRiesgoRealPEN
	ord := make([]int, len(test))
	for i := range ord {
		ord[i] = i
	}
	sort.Slice(ord, func(a, b int) bool { return scores[ord[a]] > scores[ord[b]] })
	k := len(ord) / 10
	hit := 0
	for _, i := range ord[:k] {
		if d.Y[test[i]] == 1 {
			hit++
		}
	}
	im.Top10PctPrecision = float64(hit) / float64(k)
	im.Top10PctRecall = float64(hit) / float64(P)
	im.Lift10 = im.Top10PctPrecision / (float64(P) / float64(len(test)))
	return im
}

func main() {
	path := flag.String("data", "data/processed/dataset.bin", "dataset binario generado por cmd/loader")
	workers := flag.Int("workers", runtime.NumCPU(), "workers (goroutines) de entrenamiento")
	trees := flag.Int("trees", 64, "árboles del Random Forest")
	depth := flag.Int("depth", 14, "profundidad máxima")
	sample := flag.Int("sample", 250000, "filas bootstrap por árbol")
	epochs := flag.Int("epochs", 4, "épocas de la regresión logística")
	bench := flag.Bool("bench", false, "benchmark de escalabilidad (1,2,4,8 workers)")
	flag.Parse()
	os.MkdirAll("reports", 0o755)
	os.MkdirAll("models", 0o755)

	t0 := time.Now()
	d, err := ml.LoadDataset(*path)
	if err != nil {
		log.Fatal(err)
	}
	trainAll, test := d.Split()
	// validación = 10 % del train (para elegir el umbral sin mirar el test)
	rng := rand.New(rand.NewSource(42))
	rng.Shuffle(len(trainAll), func(i, j int) { trainAll[i], trainAll[j] = trainAll[j], trainAll[i] })
	nVal := len(trainAll) / 10
	val, train := trainAll[:nVal], trainAll[nVal:]
	sort.Ints(train)
	log.Printf("dataset N=%d D=%d | train=%d val=%d test=%d | cargado en %.1fs | CPU=%d",
		d.N, d.D, len(train), len(val), len(test), time.Since(t0).Seconds(), runtime.NumCPU())

	tb := time.Now()
	binner := ml.FitBinner(d, train, 200000, 7)
	Xb := binner.Transform(d, *workers)
	log.Printf("binning (histogramas %d bins) en %.2fs", ml.MaxBins, time.Since(tb).Seconds())

	posRate := 0.0
	for _, r := range train {
		posRate += float64(d.Y[r])
	}
	posRate /= float64(len(train))
	mtry := int(math.Round(math.Sqrt(float64(d.D))))

	if *bench {
		runBench(d, Xb, train, mtry, posRate)
		return
	}

	// ---------------- Random Forest distribuido (bagging por shards)
	rfCfg := ml.ForestConfig{Trees: *trees, Workers: *workers, SampleSize: *sample, MaxDepth: *depth,
		MinLeaf: 50, Mtry: mtry, Seed: 2026}
	t := time.Now()
	forest, rfWorkers := ml.TrainForest(Xb, d.Y, d.D, train, rfCfg)
	rfTrain := time.Since(t).Seconds()
	log.Printf("[RF] %d árboles con %d workers en %.2fs", len(forest.Trees), *workers, rfTrain)
	t = time.Now()
	rfVal := forest.Predict(Xb, d.D, val, *workers)
	rfThr := ml.BestF1Threshold(rfVal, pick(d.Y, val))
	rfTest := forest.Predict(Xb, d.D, test, *workers)
	rfPredS := time.Since(t).Seconds()
	rfM := ml.Evaluate(rfTest, pick(d.Y, test), pick(d.Anulada, test), pick(d.Sobrecosto, test), rfThr)
	rfM.AUC, rfM.AP = ml.AUCAP(rfTest, pick(d.Y, test))
	log.Printf("[RF] test AUC=%.4f AP=%.4f F1=%.4f P=%.4f R=%.4f (umbral %.2f) recall_anul=%.3f recall_sobre=%.3f",
		rfM.AUC, rfM.AP, rfM.F1, rfM.Precision, rfM.Recall, rfThr, rfM.RecallAnuladas, rfM.RecallSobrecosto)

	// ---------------- Regresión logística (SGD sincrónico data-parallel)
	lrCfg := ml.LogRegConfig{Workers: *workers, Epochs: *epochs, BatchSize: 2048, LR: 0.01, L2: 1e-5,
		PosWeight: (1 - posRate) / posRate, Seed: 2026}
	t = time.Now()
	lr, lrWorkers := ml.TrainLogReg(d, train, lrCfg)
	lrTrain := time.Since(t).Seconds()
	log.Printf("[LR] %d épocas con %d workers en %.2fs, loss=%v", *epochs, *workers, lrTrain, lr.Loss)
	lrVal := lr.Predict(d, val, *workers)
	lrThr := ml.BestF1Threshold(lrVal, pick(d.Y, val))
	lrTest := lr.Predict(d, test, *workers)
	lrM := ml.Evaluate(lrTest, pick(d.Y, test), pick(d.Anulada, test), pick(d.Sobrecosto, test), lrThr)
	lrM.AUC, lrM.AP = ml.AUCAP(lrTest, pick(d.Y, test))
	log.Printf("[LR] test AUC=%.4f AP=%.4f F1=%.4f P=%.4f R=%.4f (umbral %.2f) recall_anul=%.3f recall_sobre=%.3f",
		lrM.AUC, lrM.AP, lrM.F1, lrM.Precision, lrM.Recall, lrThr, lrM.RecallAnuladas, lrM.RecallSobrecosto)

	// AUC solo para anulaciones (sub-problema más difícil: cancelación)
	anAUCrf, _ := ml.AUCAP(rfTest, pick(d.Anulada, test))
	anAUClr, _ := ml.AUCAP(lrTest, pick(d.Anulada, test))

	type imp struct {
		Feature    string  `json:"feature"`
		Importance float64 `json:"importancia"`
	}
	var imps []imp
	for j, v := range forest.Importance {
		imps = append(imps, imp{d.Names[j], v})
	}
	sort.Slice(imps, func(a, b int) bool { return imps[a].Importance > imps[b].Importance })
	coefs := map[string]float64{}
	for j, w := range lr.W {
		coefs[d.Names[j]] = w
	}

	writeJSON("reports/train_report.json", map[string]any{
		"cpu": runtime.NumCPU(), "workers": *workers,
		"split":                map[string]int{"train": len(train), "validacion": len(val), "test": len(test)},
		"tasa_positivos_train": posRate,
		"random_forest": map[string]any{"config": rfCfg, "segundos_entrenamiento": rfTrain,
			"segundos_prediccion_val_test": rfPredS, "workers": rfWorkers, "test": rfM,
			"auc_solo_anulaciones": anAUCrf, "importancia": imps,
			"impacto": calcImpacto(d, test, rfTest, rfThr)},
		"logistic_regression": map[string]any{"config": lrCfg, "segundos_entrenamiento": lrTrain,
			"workers": lrWorkers, "test": lrM, "auc_solo_anulaciones": anAUClr, "loss_por_epoca": lr.Loss,
			"coeficientes": coefs, "impacto": calcImpacto(d, test, lrTest, lrThr)},
	})
	writeJSON("models/random_forest.json", map[string]any{"binner": binner, "forest": forest, "umbral": rfThr, "features": d.Names})
	writeJSON("models/logistic_regression.json", map[string]any{"model": lr, "umbral": lrThr, "features": d.Names})
	log.Printf("listo en %.1fs → reports/train_report.json, models/", time.Since(t0).Seconds())
}

// runBench mide tiempo y speedup con distinto número de workers (misma carga total).
func runBench(d *ml.Dataset, Xb []uint8, train []int, mtry int, posRate float64) {
	type res struct {
		Modelo   string  `json:"modelo"`
		Workers  int     `json:"workers"`
		Segundos float64 `json:"segundos"`
		Speedup  float64 `json:"speedup"`
		Eficienc float64 `json:"eficiencia"`
	}
	var rs []res
	var base float64
	for _, w := range []int{1, 2, 4, 8, 16} {
		runtime.GC()
		cfg := ml.ForestConfig{Trees: 32, Workers: w, SampleSize: 100000, MaxDepth: 12, MinLeaf: 50, Mtry: mtry, Seed: 1}
		t := time.Now()
		ml.TrainForest(Xb, d.Y, d.D, train, cfg)
		s := time.Since(t).Seconds()
		if w == 1 {
			base = s
		}
		rs = append(rs, res{"random_forest", w, s, base / s, base / s / math.Min(float64(w), float64(runtime.NumCPU()))})
		log.Printf("[bench RF] workers=%d %.2fs speedup=%.2fx", w, s, base/s)
	}
	for _, w := range []int{1, 2, 4, 8, 16} {
		runtime.GC()
		cfg := ml.LogRegConfig{Workers: w, Epochs: 1, BatchSize: 8192 / w, LR: 0.01, L2: 1e-5, PosWeight: (1 - posRate) / posRate, Seed: 1}
		t := time.Now()
		ml.TrainLogReg(d, train, cfg)
		s := time.Since(t).Seconds()
		if w == 1 {
			base = s
		}
		rs = append(rs, res{"logistic_regression", w, s, base / s, base / s / math.Min(float64(w), float64(runtime.NumCPU()))})
		log.Printf("[bench LR] workers=%d %.2fs speedup=%.2fx", w, s, base/s)
	}
	writeJSON("reports/bench_train.json", map[string]any{"cpu": runtime.NumCPU(), "resultados": rs})
}
