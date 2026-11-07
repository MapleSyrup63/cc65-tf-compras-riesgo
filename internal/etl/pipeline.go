package etl

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/xlsx"
)

// Config del pipeline de carga.
type Config struct {
	Files        []string // rutas .xlsx
	FileWorkers  int      // archivos leídos en paralelo (acota la memoria)
	ParseWorkers int      // goroutines para parsear bloques de cada XLSX
	CleanWorkers int      // goroutines de limpieza/validación
	BatchSize    int      // filas por lote enviado por el channel
}

// Stats resume la ejecución (va al informe como evidencia).
type Stats struct {
	FilasLeidas     int            `json:"filas_leidas"`
	FilasValidas    int            `json:"filas_validas"`
	Duplicados      int            `json:"duplicados_eliminados"`
	Rechazos        map[string]int `json:"rechazos"`
	FilasPorArchivo map[string]int `json:"filas_por_archivo"`
	CorreccionesEnc int            `json:"correcciones_codificacion"`
	TiempoLectura   time.Duration  `json:"-"`
	TiempoTotal     time.Duration  `json:"-"`
	TiempoLecturaS  float64        `json:"tiempo_lectura_s"`
	TiempoTotalS    float64        `json:"tiempo_total_s"`
	Workers         Config         `json:"-"`
}

type rawBatch struct {
	archivo string
	rows    []xlsx.Row
}

type cleanBatch struct {
	recs     []*Record
	rechazos map[string]int
	fixes    int
	archivo  string
	leidas   int
}

var mesRe = regexp.MustCompile(`(?i)ORDENESCOMPRA([A-Z]+)\d{4}`)

func mesDeArchivo(path string) string {
	m := mesRe.FindStringSubmatch(filepath.Base(path))
	if len(m) == 2 {
		return strings.ToUpper(m[1])
	}
	return filepath.Base(path)
}

var ordenMeses = map[string]int{"ENERO": 1, "FEBRERO": 2, "MARZO": 3, "ABRIL": 4, "MAYO": 5, "JUNIO": 6,
	"JULIO": 7, "AGOSTO": 8, "SEPTIEMBRE": 9, "SETIEMBRE": 9, "OCTUBRE": 10, "NOVIEMBRE": 11, "DICIEMBRE": 12}

func mesRank(m string) int { return ordenMeses[m] }

var rankEstado = map[string]int{"EMITIDA": 1, "COMPROMETIDA": 2, "DEVENGADA": 3, "ANULADA": 4}

// newer decide de forma determinística qué versión de una orden duplicada conservar:
// primero el archivo más reciente; si empatan, el estado más avanzado del ciclo;
// si persiste el empate, la descripción mayor lexicográficamente.
func newer(a, b *Record) bool {
	if ma, mb := mesRank(a.ArchivoMes), mesRank(b.ArchivoMes); ma != mb {
		return ma > mb
	}
	if ea, eb := rankEstado[a.Estado], rankEstado[b.Estado]; ea != eb {
		return ea > eb
	}
	if a.Descripcion != b.Descripcion {
		return a.Descripcion > b.Descripcion
	}
	// Desempate total: copias con la misma descripción pueden diferir en
	// modalidad, objeto o razón social; sin esto el resultado depende del
	// orden de llegada de los goroutines.
	if a.TipoContratacion != b.TipoContratacion {
		return a.TipoContratacion > b.TipoContratacion
	}
	if a.Objeto != b.Objeto {
		return a.Objeto > b.Objeto
	}
	return a.Contratista > b.Contratista
}

// Load ejecuta el pipeline concurrente:
//
//	[productores: 1 goroutine por archivo, acotadas por semáforo]
//	        │ rawCh (lotes de filas)
//	        ▼
//	[CleanWorkers goroutines: CleanRow]   ← fan-out
//	        │ cleanCh (lotes limpios)
//	        ▼
//	[agregador único: deduplicación + estadísticas]  ← fan-in
//
// El agregador es el único dueño del mapa de deduplicación, por lo que no
// necesita mutex ("share memory by communicating").
func Load(cfg Config) ([]*Record, *Stats, error) {
	t0 := time.Now()
	rawCh := make(chan rawBatch, cfg.CleanWorkers*2)
	cleanCh := make(chan cleanBatch, cfg.CleanWorkers*2)
	errCh := make(chan error, len(cfg.Files))

	// --- Etapa 1: productores
	var prodWG sync.WaitGroup
	sem := make(chan struct{}, cfg.FileWorkers)
	var readDur time.Duration
	var readMu sync.Mutex
	for _, f := range cfg.Files {
		prodWG.Add(1)
		go func(path string) {
			defer prodWG.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			tr := time.Now()
			sh, err := xlsx.ReadFile(path, cfg.ParseWorkers)
			if err != nil {
				errCh <- fmt.Errorf("%s: %w", path, err)
				return
			}
			readMu.Lock()
			readDur += time.Since(tr)
			readMu.Unlock()
			if len(sh.Header) < len(Header) || normText(sh.Header[ColMonto]) != Header[ColMonto] {
				errCh <- fmt.Errorf("%s: esquema inesperado %v", path, sh.Header)
				return
			}
			mes := mesDeArchivo(path)
			for i := 0; i < len(sh.Rows); i += cfg.BatchSize {
				j := i + cfg.BatchSize
				if j > len(sh.Rows) {
					j = len(sh.Rows)
				}
				rawCh <- rawBatch{archivo: mes, rows: sh.Rows[i:j]}
			}
		}(f)
	}
	go func() { prodWG.Wait(); close(rawCh) }()

	// --- Etapa 2: workers de limpieza (fan-out)
	var cleanWG sync.WaitGroup
	for w := 0; w < cfg.CleanWorkers; w++ {
		cleanWG.Add(1)
		go func() {
			defer cleanWG.Done()
			for b := range rawCh {
				out := cleanBatch{rechazos: map[string]int{}, archivo: b.archivo, leidas: len(b.rows),
					recs: make([]*Record, 0, len(b.rows))}
				for _, row := range b.rows {
					if len(row) > ColNombreContratista && strings.ContainsRune(row[ColNombreContratista], '¿') {
						out.fixes++
					}
					motivo, rec := CleanRow(row, b.archivo)
					if rec == nil {
						out.rechazos[motivo]++
						continue
					}
					out.recs = append(out.recs, rec)
				}
				cleanCh <- out
			}
		}()
	}
	go func() { cleanWG.Wait(); close(cleanCh) }()

	// --- Etapa 3: agregador (fan-in)
	st := &Stats{Rechazos: map[string]int{}, FilasPorArchivo: map[string]int{}, Workers: cfg}
	// clave -> posición en data. Si una orden aparece en dos archivos mensuales,
	// se conserva la versión del archivo más reciente (estado más actualizado).
	seen := make(map[string]int, 2_500_000)
	var data []*Record
	for b := range cleanCh {
		st.FilasLeidas += b.leidas
		st.FilasPorArchivo[b.archivo] += b.leidas
		st.CorreccionesEnc += b.fixes
		for k, v := range b.rechazos {
			st.Rechazos[k] += v
		}
		for _, r := range b.recs {
			k := r.DedupKey()
			if pos, dup := seen[k]; dup {
				st.Duplicados++
				if newer(r, data[pos]) {
					data[pos] = r
				}
				continue
			}
			seen[k] = len(data)
			data = append(data, r)
		}
	}
	close(errCh)
	for err := range errCh {
		return nil, nil, err
	}
	st.FilasValidas = len(data)
	st.TiempoLectura = readDur
	st.TiempoTotal = time.Since(t0)
	st.TiempoLecturaS = readDur.Seconds()
	st.TiempoTotalS = st.TiempoTotal.Seconds()
	return data, st, nil
}
