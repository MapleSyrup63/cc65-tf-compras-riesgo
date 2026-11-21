// Package xlsx implementa un lector XLSX en Go puro (solo librería estándar)
// pensado para archivos grandes (cientos de miles de filas).
//
// Patrón de concurrencia (Parte A del TF):
//   - El XML de la hoja se divide en K bloques cortando en límites "<row".
//   - Cada bloque se asigna a un worker (goroutine) que parsea sus filas.
//   - Los resultados vuelven por un channel etiquetados con su índice de bloque
//     para conservar el orden original del archivo.
//   - sharedStrings.xml se parsea de la misma forma (bloques de "<si>").
package xlsx

import (
	"archive/zip"
	"bytes"
	"fmt"
	"html"
	"io"
	"strconv"
	"sync"
)

// Row es una fila con las celdas ya resueltas a texto, indexadas por columna (0 = A).
type Row []string

// Sheet contiene las filas de la primera hoja del libro.
type Sheet struct {
	Header []string
	Rows   []Row
}

func readZipEntry(zr *zip.ReadCloser, name string) ([]byte, error) {
	for _, f := range zr.File {
		if f.Name == name {
			rc, err := f.Open()
			if err != nil {
				return nil, err
			}
			defer rc.Close()
			buf := bytes.NewBuffer(make([]byte, 0, f.UncompressedSize64))
			_, err = io.Copy(buf, rc)
			return buf.Bytes(), err
		}
	}
	return nil, fmt.Errorf("entrada %s no encontrada", name)
}

// splitAt divide data en hasta n bloques, cortando solo justo antes de "sep".
func splitAt(data []byte, sep []byte, n int) [][]byte {
	if n < 1 {
		n = 1
	}
	chunks := make([][]byte, 0, n)
	size := len(data) / n
	start := 0
	for i := 1; i < n; i++ {
		target := i * size
		if target <= start {
			continue
		}
		idx := bytes.Index(data[target:], sep)
		if idx < 0 {
			break
		}
		cut := target + idx
		chunks = append(chunks, data[start:cut])
		start = cut
	}
	chunks = append(chunks, data[start:])
	return chunks
}

type chunkResult[T any] struct {
	idx  int
	vals []T
}

// parallelParse aplica parse a cada bloque en una goroutine y concatena en orden.
func parallelParse[T any](chunks [][]byte, parse func([]byte) []T) []T {
	out := make(chan chunkResult[T], len(chunks))
	var wg sync.WaitGroup
	for i, c := range chunks {
		wg.Add(1)
		go func(i int, c []byte) {
			defer wg.Done()
			out <- chunkResult[T]{idx: i, vals: parse(c)}
		}(i, c)
	}
	go func() { wg.Wait(); close(out) }()

	parts := make([][]T, len(chunks))
	total := 0
	for r := range out { // fan-in
		parts[r.idx] = r.vals
		total += len(r.vals)
	}
	res := make([]T, 0, total)
	for _, p := range parts {
		res = append(res, p...)
	}
	return res
}

var (
	siOpen  = []byte("<si>")
	siClose = []byte("</si>")
	tOpen   = []byte("<t")
	tClose  = []byte("</t>")
	rowOpen = []byte("<row")
	rowEnd  = []byte("</row>")
	cOpen   = []byte("<c ")
	vOpen   = []byte("<v>")
	vClose  = []byte("</v>")
)

// parseSharedStrings extrae el texto de cada <si> (incluye rich text <r><t>).
func parseSharedStrings(b []byte) []string {
	var out []string
	for {
		i := bytes.Index(b, siOpen)
		if i < 0 {
			return out
		}
		b = b[i+len(siOpen):]
		j := bytes.Index(b, siClose)
		if j < 0 {
			return out
		}
		si := b[:j]
		b = b[j+len(siClose):]
		var sb []byte
		for {
			k := bytes.Index(si, tOpen)
			if k < 0 {
				break
			}
			si = si[k+len(tOpen):]
			gt := bytes.IndexByte(si, '>')
			if gt < 0 {
				break
			}
			if gt > 0 && si[gt-1] == '/' { // <t/> vacío
				si = si[gt+1:]
				continue
			}
			si = si[gt+1:]
			e := bytes.Index(si, tClose)
			if e < 0 {
				break
			}
			sb = append(sb, si[:e]...)
			si = si[e+len(tClose):]
		}
		s := string(sb)
		if bytes.IndexByte(sb, '&') >= 0 {
			s = html.UnescapeString(s)
		}
		out = append(out, s)
	}
}

// colIndex convierte la referencia "AB12" -> 27.
func colIndex(ref []byte) int {
	n := 0
	for _, ch := range ref {
		if ch < 'A' || ch > 'Z' {
			break
		}
		n = n*26 + int(ch-'A'+1)
	}
	return n - 1
}

func attr(tag []byte, name string) []byte {
	key := []byte(" " + name + `="`)
	i := bytes.Index(tag, key)
	if i < 0 {
		return nil
	}
	rest := tag[i+len(key):]
	j := bytes.IndexByte(rest, '"')
	if j < 0 {
		return nil
	}
	return rest[:j]
}

// rawCell es una celda antes de resolver shared strings.
type rawCell struct {
	col    int
	shared bool
	val    string
}

type rawRow []rawCell

func parseRows(b []byte) []rawRow {
	var out []rawRow
	for {
		i := bytes.Index(b, rowOpen)
		if i < 0 {
			return out
		}
		b = b[i:]
		j := bytes.Index(b, rowEnd)
		var rowXML []byte
		if j < 0 { // fila vacía autocerrada o fin
			gt := bytes.IndexByte(b, '>')
			if gt < 0 {
				return out
			}
			b = b[gt+1:]
			continue
		}
		rowXML = b[:j]
		b = b[j+len(rowEnd):]
		var row rawRow
		for {
			k := bytes.Index(rowXML, cOpen)
			if k < 0 {
				break
			}
			rowXML = rowXML[k:]
			gt := bytes.IndexByte(rowXML, '>')
			if gt < 0 {
				break
			}
			tag := rowXML[:gt]
			selfClosed := tag[len(tag)-1] == '/'
			rowXML = rowXML[gt+1:]
			if selfClosed {
				continue
			}
			end := bytes.Index(rowXML, []byte("</c>"))
			if end < 0 {
				break
			}
			body := rowXML[:end]
			rowXML = rowXML[end+4:]
			cell := rawCell{col: colIndex(attr(tag, "r"))}
			t := string(attr(tag, "t"))
			switch t {
			case "inlineStr":
				ti := bytes.Index(body, tOpen)
				if ti >= 0 {
					rest := body[ti:]
					g := bytes.IndexByte(rest, '>')
					e := bytes.Index(rest, tClose)
					if g >= 0 && e > g {
						cell.val = html.UnescapeString(string(rest[g+1 : e]))
					}
				}
			default:
				vi := bytes.Index(body, vOpen)
				if vi >= 0 {
					ve := bytes.Index(body[vi:], vClose)
					if ve >= 0 {
						cell.val = string(body[vi+len(vOpen) : vi+ve])
					}
				}
				cell.shared = t == "s"
				if t == "str" {
					cell.val = html.UnescapeString(cell.val)
				}
			}
			row = append(row, cell)
		}
		out = append(out, row)
	}
}

// ReadFile lee la primera hoja de un .xlsx usando `workers` goroutines.
func ReadFile(path string, workers int) (*Sheet, error) {
	zr, err := zip.OpenReader(path)
	if err != nil {
		return nil, err
	}
	defer zr.Close()

	var ss []string
	ssXML, err := readZipEntry(zr, "xl/sharedStrings.xml")
	if err == nil {
		ss = parallelParse(splitAt(ssXML, siOpen, workers), parseSharedStrings)
		ssXML = nil
	}
	sheetXML, err := readZipEntry(zr, "xl/worksheets/sheet1.xml")
	if err != nil {
		return nil, err
	}
	raws := parallelParse(splitAt(sheetXML, rowOpen, workers), parseRows)
	sheetXML = nil

	resolve := func(r rawRow) Row {
		maxCol := 0
		for _, c := range r {
			if c.col > maxCol {
				maxCol = c.col
			}
		}
		row := make(Row, maxCol+1)
		for _, c := range r {
			if c.col < 0 {
				continue
			}
			if c.shared {
				idx, err := strconv.Atoi(c.val)
				if err == nil && idx >= 0 && idx < len(ss) {
					row[c.col] = ss[idx]
				}
			} else {
				row[c.col] = c.val
			}
		}
		return row
	}
	if len(raws) == 0 {
		return &Sheet{}, nil
	}
	sh := &Sheet{Header: resolve(raws[0]), Rows: make([]Row, len(raws)-1)}
	// resolución de shared strings también en paralelo (por rangos)
	var wg sync.WaitGroup
	n := len(raws) - 1
	step := (n + workers - 1) / workers
	for w := 0; w < workers; w++ {
		lo, hi := w*step, (w+1)*step
		if hi > n {
			hi = n
		}
		if lo >= hi {
			continue
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			for i := lo; i < hi; i++ { // cada goroutine escribe en su propio rango: sin carrera
				sh.Rows[i] = resolve(raws[i+1])
			}
		}(lo, hi)
	}
	wg.Wait()
	return sh, nil
}
