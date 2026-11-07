// Package ml implementa la Parte B del TF: modelos de clasificación binaria
// entrenados en paralelo (Random Forest por bagging distribuido entre workers y
// Regresión Logística con descenso de gradiente sincrónico data-parallel).
//
// Toda la lógica de "worker" (entrenar árboles sobre un shard, calcular el
// gradiente local de un mini-batch) está aislada en funciones puras para que en
// la PC4 se ejecute en nodos remotos vía TCP sin cambios.
package ml

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"io"
	"os"
	"sync"
)

// Dataset es una matriz densa N×D en formato plano (row-major, float32).
type Dataset struct {
	X          []float32
	Y          []uint8 // 1 = riesgo
	Anulada    []uint8 // componente "cancelación" de la etiqueta
	Sobrecosto []uint8 // componente "sobrecosto" de la etiqueta
	IsTrain    []bool
	N, D       int
	Names      []string
}

// Row devuelve la fila i (sin copiar).
func (d *Dataset) Row(i int) []float32 { return d.X[i*d.D : (i+1)*d.D] }

// Split devuelve los índices de train y test.
func (d *Dataset) Split() (train, test []int) {
	for i := 0; i < d.N; i++ {
		if d.IsTrain[i] {
			train = append(train, i)
		} else {
			test = append(test, i)
		}
	}
	return
}

// Save escribe el dataset en un formato binario simple por streaming:
// cabecera JSON (N, D, nombres) + arreglos little-endian. Evita copiar todo a memoria.
func (d *Dataset) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriterSize(f, 4<<20)
	hdr, _ := json.Marshal(struct {
		N, D  int
		Names []string
	}{d.N, d.D, d.Names})
	binary.Write(w, binary.LittleEndian, uint32(len(hdr)))
	w.Write(hdr)
	const chunk = 1 << 20
	for i := 0; i < len(d.X); i += chunk {
		j := i + chunk
		if j > len(d.X) {
			j = len(d.X)
		}
		if err := binary.Write(w, binary.LittleEndian, d.X[i:j]); err != nil {
			return err
		}
	}
	w.Write(d.Y)
	w.Write(d.Anulada)
	w.Write(d.Sobrecosto)
	tr := make([]byte, d.N)
	for i, b := range d.IsTrain {
		if b {
			tr[i] = 1
		}
	}
	w.Write(tr)
	return w.Flush()
}

// LoadDataset lee el formato escrito por Save.
func LoadDataset(path string) (*Dataset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 4<<20)
	var hl uint32
	if err := binary.Read(r, binary.LittleEndian, &hl); err != nil {
		return nil, err
	}
	hdr := make([]byte, hl)
	if _, err := io.ReadFull(r, hdr); err != nil {
		return nil, err
	}
	d := &Dataset{}
	var h struct {
		N, D  int
		Names []string
	}
	if err := json.Unmarshal(hdr, &h); err != nil {
		return nil, err
	}
	d.N, d.D, d.Names = h.N, h.D, h.Names
	d.X = make([]float32, d.N*d.D)
	const chunk = 1 << 20
	for i := 0; i < len(d.X); i += chunk {
		j := i + chunk
		if j > len(d.X) {
			j = len(d.X)
		}
		if err := binary.Read(r, binary.LittleEndian, d.X[i:j]); err != nil {
			return nil, err
		}
	}
	d.Y, d.Anulada, d.Sobrecosto = make([]uint8, d.N), make([]uint8, d.N), make([]uint8, d.N)
	tr := make([]byte, d.N)
	for _, b := range [][]byte{d.Y, d.Anulada, d.Sobrecosto, tr} {
		if _, err := io.ReadFull(r, b); err != nil {
			return nil, err
		}
	}
	d.IsTrain = make([]bool, d.N)
	for i, b := range tr {
		d.IsTrain[i] = b == 1
	}
	return d, nil
}

// ParallelFor ejecuta fn(lo, hi, w) sobre `workers` rangos contiguos de [0,n).
// Cada goroutine trabaja sobre un rango disjunto: no hay escrituras compartidas.
func ParallelFor(n, workers int, fn func(lo, hi, w int)) {
	if workers < 1 {
		workers = 1
	}
	step := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo, hi := w*step, (w+1)*step
		if hi > n {
			hi = n
		}
		if lo >= hi {
			continue
		}
		wg.Add(1)
		go func(lo, hi, w int) {
			defer wg.Done()
			fn(lo, hi, w)
		}(lo, hi, w)
	}
	wg.Wait()
}
