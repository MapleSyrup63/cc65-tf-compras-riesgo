package etl

import (
	"hash/fnv"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/ml"
)

// FeatureNames: variables disponibles ANTES de la adjudicación / ejecución.
// Se excluyen a propósito ESTADOCONTRATACION y las fechas de compromiso y
// notificación porque son posteriores al evento a predecir (data leakage).
var FeatureNames = func() []string {
	n := []string{"log_monto_pen", "es_orden_compra",
		"obj_bienes", "obj_servicios", "obj_consultoria_obras",
		"tc_hasta_8uit", "tc_sin_proceso", "tc_proceso_seleccion", "tc_catalogo", "tc_servicio_publico", "tc_directa_otros",
		"moneda_extranjera", "contratista_pn", "contratista_empresa", "contratista_otro"}
	for _, r := range Rubros {
		n = append(n, "rubro_"+r)
	}
	n = append(n, "niv_muni_distrital", "niv_muni_provincial", "niv_gob_regional", "niv_gob_nacional", "niv_universidad", "niv_salud",
		"es_lima_callao", "te_departamento", "te_entidad", "log_n_ordenes_entidad",
		"te_contratista", "log_n_ordenes_contratista", "log_n_entidades_contratista",
		"concentracion_contratista_en_entidad", "dias_emision_a_registro", "mes_emision", "fin_de_semana",
		"cerca_tope_8uit", "log_len_descripcion", "contratista_es_entidad")
	return n
}()

var (
	tcIdx  = map[string]int{"HASTA_8UIT": 0, "SIN_PROCESO": 1, "PROCESO_SELECCION": 2, "CATALOGO": 3, "SERVICIO_PUBLICO": 4, "DIRECTA_OTROS": 5}
	objIdx = map[string]int{"BIENES": 0, "SERVICIOS": 1, "CONSULTORIAS OBRAS": 2}
	nivIdx = map[string]int{"MUNI_DISTRITAL": 0, "MUNI_PROVINCIAL": 1, "GOB_REGIONAL": 2, "GOB_NACIONAL": 3, "UNIVERSIDAD": 4, "SALUD": 5}
	rubIdx = func() map[string]int {
		m := map[string]int{}
		for i, r := range Rubros {
			m[r] = i
		}
		return m
	}()
)

// FeatureReport documenta la construcción de la etiqueta y los umbrales.
type FeatureReport struct {
	Train, Test        int
	Positivos          int
	Anuladas           int
	SobrecostoP95      int
	SobrecostoTope8UIT int
	GruposComparables  int
	TasaRiesgo         float64
	TasaAnulacionPrior float64
	TiempoFeaturesS    float64
}

// IsTrainRow asigna el split 80/20 de forma determinística (hash de la clave),
// independiente del orden en que llegaron los registros desde los goroutines.
func IsTrainRow(r *Record) bool {
	h := fnv.New32a()
	h.Write([]byte(r.DedupKey()))
	return h.Sum32()%100 < 80
}

func grupo(r *Record, nivel int) string {
	switch nivel {
	case 0:
		return r.Objeto + "|" + r.Rubro + "|" + r.TipoContratacion
	case 1:
		return r.Objeto + "|" + r.Rubro
	default:
		return r.Objeto
	}
}

type entStat struct{ n, nTrain, anulTrain int }

func mergeEnt(dst, src map[string]*entStat) {
	for k, v := range src {
		d, ok := dst[k]
		if !ok {
			dst[k] = v
			continue
		}
		d.n += v.n
		d.nTrain += v.nTrain
		d.anulTrain += v.anulTrain
	}
}

// percentil sobre un slice ordenado.
func percentil(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return math.Inf(1)
	}
	idx := int(math.Ceil(p*float64(len(sorted)))) - 1
	if idx < 0 {
		idx = 0
	}
	return sorted[idx]
}

// BuildFeatures etiqueta los registros y construye la matriz de features.
// Todas las etapas son map-reduce con `workers` goroutines.
func BuildFeatures(data []*Record, workers int) (*ml.Dataset, *FeatureReport) {
	t0 := time.Now()
	n := len(data)
	rep := &FeatureReport{}

	// 1) split determinístico (paralelo, cada goroutine escribe en su rango)
	ml.ParallelFor(n, workers, func(lo, hi, _ int) {
		for i := lo; i < hi; i++ {
			data[i].EsTrain = IsTrainRow(data[i])
		}
	})

	// 2) MAP: cada worker acumula montos por grupo comparable y conteos por
	//    entidad/contratista/departamento/par en mapas LOCALES (sin locks).
	type partial struct {
		montos [3]map[string][]float64
		ent    map[string]*entStat
		con    map[string]*entStat
		dep    map[string]*entStat
		par    map[string]int
	}
	parts := make([]*partial, workers)
	ml.ParallelFor(n, workers, func(lo, hi, w int) {
		p := &partial{ent: map[string]*entStat{}, con: map[string]*entStat{}, dep: map[string]*entStat{}, par: map[string]int{}}
		for l := 0; l < 3; l++ {
			p.montos[l] = map[string][]float64{}
		}
		inc := func(m map[string]*entStat, k string, r *Record) {
			s := m[k]
			if s == nil {
				s = &entStat{}
				m[k] = s
			}
			s.n++
			if r.EsTrain {
				s.nTrain++
				if r.Anulada {
					s.anulTrain++
				}
			}
		}
		for i := lo; i < hi; i++ {
			r := data[i]
			if r.EsTrain && !r.Anulada { // umbrales de sobrecosto solo con train (sin fuga)
				for l := 0; l < 3; l++ {
					g := grupo(r, l)
					p.montos[l][g] = append(p.montos[l][g], r.MontoPEN)
				}
			}
			inc(p.ent, r.RucEntidad, r)
			inc(p.con, r.RucContratista, r)
			inc(p.dep, r.Departamento, r)
			p.par[r.RucEntidad+"|"+r.RucContratista]++
		}
		parts[w] = p
	})

	// 3) REDUCE: fusión de los mapas parciales
	var montos [3]map[string][]float64
	for l := 0; l < 3; l++ {
		montos[l] = map[string][]float64{}
	}
	ent, con, dep, par := map[string]*entStat{}, map[string]*entStat{}, map[string]*entStat{}, map[string]int{}
	for _, p := range parts {
		if p == nil {
			continue
		}
		for l := 0; l < 3; l++ {
			for k, v := range p.montos[l] {
				montos[l][k] = append(montos[l][k], v...)
			}
		}
		mergeEnt(ent, p.ent)
		mergeEnt(con, p.con)
		mergeEnt(dep, p.dep)
		for k, v := range p.par {
			par[k] += v
		}
	}
	nEntCon := map[string]int{} // entidades distintas por contratista
	for k := range par {
		for j := len(k) - 1; j >= 0; j-- {
			if k[j] == '|' {
				nEntCon[k[j+1:]]++
				break
			}
		}
	}

	// 4) P95 por grupo: un pool de goroutines ordena cada grupo (jobs por channel)
	type job struct {
		l int
		g string
	}
	p95 := [3]map[string]float64{{}, {}, {}}
	var mu sync.Mutex
	jobs := make(chan job)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				v := montos[j.l][j.g]
				sort.Float64s(v)
				q := math.Inf(1)
				if len(v) >= 30 { // grupos pequeños usan el nivel superior
					q = percentil(v, 0.95)
				}
				mu.Lock()
				p95[j.l][j.g] = q
				mu.Unlock()
			}
		}()
	}
	for l := 0; l < 3; l++ {
		for g := range montos[l] {
			jobs <- job{l, g}
		}
	}
	close(jobs)
	wg.Wait()
	rep.GruposComparables = len(p95[0])

	// prior global de anulación (solo train)
	var nTr, anTr int
	for _, s := range dep {
		nTr += s.nTrain
		anTr += s.anulTrain
	}
	prior := float64(anTr) / float64(nTr)
	rep.TasaAnulacionPrior = prior
	const m = 20.0 // suavizado bayesiano del target encoding
	te := func(s *entStat, self *Record) float64 {
		nn, aa := float64(s.nTrain), float64(s.anulTrain)
		if self.EsTrain { // leave-one-out: la fila no ve su propia etiqueta
			nn--
			if self.Anulada {
				aa--
			}
		}
		return (aa + prior*m) / (nn + m)
	}

	// 5) etiqueta + vector de features (paralelo por rangos)
	D := len(FeatureNames)
	ds := &ml.Dataset{N: n, D: D, Names: FeatureNames, X: make([]float32, n*D),
		Y: make([]uint8, n), Anulada: make([]uint8, n), Sobrecosto: make([]uint8, n), IsTrain: make([]bool, n)}
	type cnt struct{ pos, an, p95, tope, train int }
	counts := make([]cnt, workers)
	ml.ParallelFor(n, workers, func(lo, hi, w int) {
		c := &counts[w]
		for i := lo; i < hi; i++ {
			r := data[i]
			// --- etiqueta
			umbral := math.Inf(1)
			for l := 0; l < 3; l++ {
				if q, ok := p95[l][grupo(r, l)]; ok && !math.IsInf(q, 1) {
					umbral = q
					break
				}
			}
			sobreP95 := r.MontoPEN > umbral
			sobreTope := r.TipoContratacion == "HASTA_8UIT" && r.MontoPEN > Tope8UIT
			r.Sobrecosto = sobreP95 || sobreTope
			r.Riesgo = r.Anulada || r.Sobrecosto
			if sobreP95 {
				c.p95++
			}
			if sobreTope {
				c.tope++
			}
			if r.Anulada {
				c.an++
			}
			if r.Riesgo {
				c.pos++
			}
			if r.EsTrain {
				c.train++
			}
			// --- features
			x := ds.X[i*D : (i+1)*D]
			k := 0
			set := func(v float64) { x[k] = float32(v); k++ }
			b2f := func(b bool) float64 {
				if b {
					return 1
				}
				return 0
			}
			set(math.Log1p(r.MontoPEN))
			set(b2f(r.EsOrdenCompra))
			oi := objIdx[r.Objeto]
			for j := 0; j < 3; j++ {
				set(b2f(oi == j))
			}
			ti := tcIdx[r.TipoContratacion]
			for j := 0; j < 6; j++ {
				set(b2f(ti == j))
			}
			set(b2f(r.Moneda != "NUEVOS SOLES"))
			set(b2f(r.TipoContratista == "PN"))
			set(b2f(r.TipoContratista == "EMPRESA"))
			set(b2f(r.TipoContratista == "OTRO"))
			ri := rubIdx[r.Rubro]
			for j := range Rubros {
				set(b2f(ri == j))
			}
			ni := nivIdx[r.NivelGobierno]
			for j := 0; j < 6; j++ {
				set(b2f(ni == j))
			}
			set(b2f(r.Departamento == "LIMA" || r.Departamento == "CALLAO"))
			set(te(dep[r.Departamento], r))
			es := ent[r.RucEntidad]
			set(te(es, r))
			set(math.Log1p(float64(es.n)))
			cs := con[r.RucContratista]
			set(te(cs, r))
			set(math.Log1p(float64(cs.n)))
			set(math.Log1p(float64(nEntCon[r.RucContratista])))
			set(float64(par[r.RucEntidad+"|"+r.RucContratista]) / float64(es.n))
			lag := r.FechaRegistro.Sub(r.FechaEmision).Hours() / 24
			set(math.Min(lag, 300) / 100)
			set(float64(r.FechaEmision.Month()) / 12)
			wd := r.FechaEmision.Weekday()
			set(b2f(wd == time.Saturday || wd == time.Sunday))
			set(b2f(r.TipoContratacion == "HASTA_8UIT" && r.MontoPEN >= 0.9*Tope8UIT && r.MontoPEN <= Tope8UIT))
			set(math.Log1p(float64(len(r.Descripcion))))
			set(b2f(r.RucContratista == r.RucEntidad))

			ds.Y[i] = uint8(b2f(r.Riesgo))
			ds.Anulada[i] = uint8(b2f(r.Anulada))
			ds.Sobrecosto[i] = uint8(b2f(r.Sobrecosto))
			ds.IsTrain[i] = r.EsTrain
		}
	})
	for _, c := range counts {
		rep.Positivos += c.pos
		rep.Anuladas += c.an
		rep.SobrecostoP95 += c.p95
		rep.SobrecostoTope8UIT += c.tope
		rep.Train += c.train
	}
	rep.Test = n - rep.Train
	rep.TasaRiesgo = float64(rep.Positivos) / float64(n)
	rep.TiempoFeaturesS = time.Since(t0).Seconds()
	return ds, rep
}
