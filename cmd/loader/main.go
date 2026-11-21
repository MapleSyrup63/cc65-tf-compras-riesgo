// loader — Parte A: carga, limpieza y procesamiento concurrente de >2M órdenes.
//
// Uso:
//
//	go run ./cmd/loader -in data/raw -out data/processed
//	go run ./cmd/loader -in data/raw -bench        # mide speedup con 1,2,4,8 workers
package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sort"
	"strconv"
	"time"

	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/eda"
	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/etl"
)

func writeJSON(path string, v any) {
	b, _ := json.MarshalIndent(v, "", "  ")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		log.Fatal(err)
	}
}

func cfgFor(files []string, w int) etl.Config {
	fw := w
	if fw > 3 {
		fw = 3
	}
	return etl.Config{Files: files, FileWorkers: fw, ParseWorkers: w, CleanWorkers: w, BatchSize: 5000}
}

func main() {
	in := flag.String("in", "data/raw", "carpeta con los .xlsx de OSCE")
	out := flag.String("out", "data/processed", "carpeta de salida")
	workers := flag.Int("workers", runtime.NumCPU()*2, "goroutines por etapa")
	bench := flag.Bool("bench", false, "ejecuta el benchmark de escalabilidad del cargador")
	sample := flag.Int("sample", 20000, "filas de muestra a exportar (datos de prueba)")
	memGB := flag.Float64("memgb", 4.5, "límite blando de memoria del runtime (GB)")
	flag.Parse()
	debug.SetMemoryLimit(int64(*memGB * (1 << 30)))

	files, _ := filepath.Glob(filepath.Join(*in, "*.xlsx"))
	if len(files) == 0 {
		log.Fatalf("no hay .xlsx en %s", *in)
	}
	sort.Strings(files)
	os.MkdirAll(*out, 0o755)
	os.MkdirAll("reports", 0o755)
	log.Printf("CPU lógicas=%d GOMAXPROCS=%d archivos=%d", runtime.NumCPU(), runtime.GOMAXPROCS(0), len(files))

	if *bench {
		type res struct {
			Workers  int     `json:"workers"`
			Segundos float64 `json:"segundos"`
			Filas    int     `json:"filas"`
			FilasSeg float64 `json:"filas_por_segundo"`
			Speedup  float64 `json:"speedup"`
		}
		var rs []res
		var base float64
		for _, w := range []int{1, 2, 4, 8, 16} {
			runtime.GC()
			_, st, err := etl.Load(cfgFor(files, w))
			if err != nil {
				log.Fatal(err)
			}
			s := st.TiempoTotal.Seconds()
			if w == 1 {
				base = s
			}
			r := res{w, s, st.FilasLeidas, float64(st.FilasLeidas) / s, base / s}
			log.Printf("workers=%d  %.2fs  %.0f filas/s  speedup=%.2fx", w, s, r.FilasSeg, r.Speedup)
			rs = append(rs, r)
		}
		writeJSON("reports/bench_loader.json", map[string]any{"cpu": runtime.NumCPU(), "resultados": rs})
		return
	}

	t0 := time.Now()
	data, st, err := etl.Load(cfgFor(files, *workers))
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("[A] leídas=%d válidas=%d duplicados=%d rechazos=%v en %.2fs",
		st.FilasLeidas, st.FilasValidas, st.Duplicados, st.Rechazos, st.TiempoTotalS)

	// Orden determinístico (los goroutines entregan lotes en orden arbitrario)
	sort.Slice(data, func(i, j int) bool {
		if !data[i].FechaEmision.Equal(data[j].FechaEmision) {
			return data[i].FechaEmision.Before(data[j].FechaEmision)
		}
		if data[i].RucEntidad != data[j].RucEntidad {
			return data[i].RucEntidad < data[j].RucEntidad
		}
		if data[i].Orden != data[j].Orden {
			return data[i].Orden < data[j].Orden
		}
		if data[i].MontoOriginal != data[j].MontoOriginal {
			return data[i].MontoOriginal < data[j].MontoOriginal
		}
		return data[i].DedupKey() < data[j].DedupKey()
	})

	ds, frep := etl.BuildFeatures(data, *workers)
	log.Printf("[A] features: N=%d D=%d positivos=%d (%.2f%%) anuladas=%d sobrecostoP95=%d tope8UIT=%d en %.2fs",
		ds.N, ds.D, frep.Positivos, 100*frep.TasaRiesgo, frep.Anuladas, frep.SobrecostoP95, frep.SobrecostoTope8UIT, frep.TiempoFeaturesS)

	te := time.Now()
	rep := eda.Compute(data, *workers)
	log.Printf("[A] EDA en %.2fs", time.Since(te).Seconds())

	if err := ds.Save(filepath.Join(*out, "dataset.bin")); err != nil {
		log.Fatal(err)
	}
	writeCSV(filepath.Join(*out, "ordenes_limpias.csv"), data, -1)
	writeCSV(filepath.Join("data", "sample", "ordenes_muestra.csv"), data, *sample)
	writeJSON("reports/etl_report.json", map[string]any{"carga": st, "features": frep,
		"feature_names": ds.Names, "workers": *workers, "cpu": runtime.NumCPU(),
		"tiempo_total_pipeline_s": time.Since(t0).Seconds()})
	writeJSON("reports/eda.json", rep)
	log.Printf("[A] pipeline completo en %.2fs", time.Since(t0).Seconds())
}

func writeCSV(path string, data []*etl.Record, limit int) {
	os.MkdirAll(filepath.Dir(path), 0o755)
	f, err := os.Create(path)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	bw := bufio.NewWriterSize(f, 1<<20)
	w := csv.NewWriter(bw)
	w.Write([]string{"entidad", "ruc_entidad", "departamento", "nivel_gobierno", "orden", "es_orden_compra",
		"descripcion", "rubro", "objeto", "tipo_contratacion", "monto_original", "moneda", "monto_pen",
		"ruc_contratista", "tipo_contratista", "contratista", "fecha_emision", "fecha_registro",
		"estado", "anulada", "sobrecosto", "riesgo", "split"})
	step := 1
	if limit > 0 && len(data) > limit {
		step = len(data) / limit
	}
	b := func(v bool) string {
		if v {
			return "1"
		}
		return "0"
	}
	for i := 0; i < len(data); i += step {
		r := data[i]
		split := "test"
		if r.EsTrain {
			split = "train"
		}
		w.Write([]string{r.Entidad, r.RucEntidad, r.Departamento, r.NivelGobierno, r.Orden, b(r.EsOrdenCompra),
			r.Descripcion, r.Rubro, r.Objeto, r.TipoContratacion, strconv.FormatFloat(r.MontoOriginal, 'f', 2, 64),
			r.Moneda, strconv.FormatFloat(r.MontoPEN, 'f', 2, 64), r.RucContratista, r.TipoContratista, r.Contratista,
			r.FechaEmision.Format("2006-01-02"), r.FechaRegistro.Format("2006-01-02"), r.Estado,
			b(r.Anulada), b(r.Sobrecosto), b(r.Riesgo), split})
	}
	w.Flush()
	bw.Flush()
	fmt.Println("escrito", path)
}
