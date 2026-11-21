// xlsx2csv: herramienta auxiliar que convierte cada XLSX a CSV crudo (sin limpieza),
// procesando varios archivos a la vez (fan-out por archivo, semáforo para acotar memoria).
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/xlsx"
)

func main() {
	in := flag.String("in", "data/raw", "carpeta con .xlsx")
	out := flag.String("out", "data/csv", "carpeta de salida")
	par := flag.Int("files", 2, "archivos en paralelo")
	flag.Parse()
	files, _ := filepath.Glob(filepath.Join(*in, "*.xlsx"))
	os.MkdirAll(*out, 0o755)
	sem := make(chan struct{}, *par)
	var wg sync.WaitGroup
	t0 := time.Now()
	for _, f := range files {
		wg.Add(1)
		go func(f string) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			t := time.Now()
			sh, err := xlsx.ReadFile(f, runtime.NumCPU()*2)
			if err != nil {
				log.Println(f, err)
				return
			}
			name := strings.TrimSuffix(filepath.Base(f), ".xlsx") + ".csv"
			fh, _ := os.Create(filepath.Join(*out, name))
			w := csv.NewWriter(fh)
			w.Write(sh.Header)
			n := len(sh.Header)
			for _, r := range sh.Rows {
				rec := make([]string, n)
				copy(rec, r)
				w.Write(rec)
			}
			w.Flush()
			fh.Close()
			fmt.Printf("%s: %d filas en %v\n", filepath.Base(f), len(sh.Rows), time.Since(t))
		}(f)
	}
	wg.Wait()
	fmt.Println("total", time.Since(t0))
}
