// Package eda calcula estadísticas descriptivas del dataset limpio con un
// patrón map-reduce: cada goroutine agrega su rango y luego se fusiona.
package eda

import (
	"math"
	"sort"

	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/etl"
	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/ml"
)

// Agg acumula conteos, montos y casos de riesgo por categoría.
type Agg struct {
	Ordenes     int     `json:"ordenes"`
	MontoPEN    float64 `json:"monto_pen"`
	Anuladas    int     `json:"anuladas"`
	Sobrecosto  int     `json:"sobrecosto"`
	Riesgo      int     `json:"riesgo"`
	MontoRiesgo float64 `json:"monto_riesgo_pen"`
}

func (a *Agg) add(r *etl.Record) {
	a.Ordenes++
	a.MontoPEN += r.MontoPEN
	if r.Anulada {
		a.Anuladas++
	}
	if r.Sobrecosto {
		a.Sobrecosto++
	}
	if r.Riesgo {
		a.Riesgo++
		a.MontoRiesgo += r.MontoPEN
	}
}

func (a *Agg) merge(b *Agg) {
	a.Ordenes += b.Ordenes
	a.MontoPEN += b.MontoPEN
	a.Anuladas += b.Anuladas
	a.Sobrecosto += b.Sobrecosto
	a.Riesgo += b.Riesgo
	a.MontoRiesgo += b.MontoRiesgo
}

// Report es la salida JSON del EDA.
type Report struct {
	Total              Agg                `json:"total"`
	PorMesEmision      map[string]*Agg    `json:"por_mes_emision"`
	PorEstado          map[string]*Agg    `json:"por_estado"`
	PorDepartamento    map[string]*Agg    `json:"por_departamento"`
	PorRubro           map[string]*Agg    `json:"por_rubro"`
	PorTipoContrat     map[string]*Agg    `json:"por_tipo_contratacion"`
	PorNivelGobierno   map[string]*Agg    `json:"por_nivel_gobierno"`
	PorObjeto          map[string]*Agg    `json:"por_objeto"`
	PorTipoContratist  map[string]*Agg    `json:"por_tipo_contratista"`
	HistLogMonto       map[string]int     `json:"hist_log10_monto"`
	Percentiles        map[string]float64 `json:"percentiles_monto_pen"`
	TopEntidadesRiesgo []EntidadRiesgo    `json:"top_entidades_riesgo"`
	Cerca8UIT          map[string]int     `json:"monto_vs_tope_8uit"`
}

type EntidadRiesgo struct {
	Entidad      string  `json:"entidad"`
	Departamento string  `json:"departamento"`
	Ordenes      int     `json:"ordenes"`
	Riesgo       int     `json:"riesgo"`
	Tasa         float64 `json:"tasa"`
	MontoRiesgo  float64 `json:"monto_riesgo_pen"`
}

type partial struct {
	total                                       Agg
	mes, est, dep, rub, tc, niv, obj, tcon, ent map[string]*Agg
	entDep                                      map[string]string
	hist                                        map[string]int
	cerca                                       map[string]int
}

func newPartial() *partial {
	mk := func() map[string]*Agg { return map[string]*Agg{} }
	return &partial{mes: mk(), est: mk(), dep: mk(), rub: mk(), tc: mk(), niv: mk(), obj: mk(), tcon: mk(), ent: mk(),
		entDep: map[string]string{}, hist: map[string]int{}, cerca: map[string]int{}}
}

func addTo(m map[string]*Agg, k string, r *etl.Record) {
	a := m[k]
	if a == nil {
		a = &Agg{}
		m[k] = a
	}
	a.add(r)
}

func mergeMap(dst, src map[string]*Agg) {
	for k, v := range src {
		if d, ok := dst[k]; ok {
			d.merge(v)
		} else {
			dst[k] = v
		}
	}
}

var meses = []string{"", "01-ENE", "02-FEB", "03-MAR", "04-ABR", "05-MAY", "06-JUN", "07-JUL", "08-AGO", "09-SEP", "10-OCT", "11-NOV", "12-DIC"}

// Compute ejecuta el EDA con `workers` goroutines.
func Compute(data []*etl.Record, workers int) *Report {
	parts := make([]*partial, workers)
	ml.ParallelFor(len(data), workers, func(lo, hi, w int) {
		p := newPartial()
		for i := lo; i < hi; i++ {
			r := data[i]
			p.total.add(r)
			addTo(p.mes, meses[r.FechaEmision.Month()], r)
			addTo(p.est, r.Estado, r)
			addTo(p.dep, r.Departamento, r)
			addTo(p.rub, r.Rubro, r)
			addTo(p.tc, r.TipoContratacion, r)
			addTo(p.niv, r.NivelGobierno, r)
			addTo(p.obj, r.Objeto, r)
			addTo(p.tcon, r.TipoContratista, r)
			addTo(p.ent, r.Entidad, r)
			p.entDep[r.Entidad] = r.Departamento
			b := int(math.Floor(math.Log10(r.MontoPEN)))
			if b > 7 {
				b = 7
			}
			p.hist[[]string{"10^0", "10^1", "10^2", "10^3", "10^4", "10^5", "10^6", ">=10^7"}[b]]++
			if r.TipoContratacion == "HASTA_8UIT" {
				switch f := r.MontoPEN / etl.Tope8UIT; {
				case f > 1:
					p.cerca["supera_tope"]++
				case f >= 0.9:
					p.cerca["90-100%"]++
				case f >= 0.75:
					p.cerca["75-90%"]++
				default:
					p.cerca["<75%"]++
				}
			}
		}
		parts[w] = p
	})
	rep := &Report{PorMesEmision: map[string]*Agg{}, PorEstado: map[string]*Agg{}, PorDepartamento: map[string]*Agg{},
		PorRubro: map[string]*Agg{}, PorTipoContrat: map[string]*Agg{}, PorNivelGobierno: map[string]*Agg{},
		PorObjeto: map[string]*Agg{}, PorTipoContratist: map[string]*Agg{}, HistLogMonto: map[string]int{}, Cerca8UIT: map[string]int{}}
	ent := map[string]*Agg{}
	entDep := map[string]string{}
	for _, p := range parts {
		if p == nil {
			continue
		}
		rep.Total.merge(&p.total)
		mergeMap(rep.PorMesEmision, p.mes)
		mergeMap(rep.PorEstado, p.est)
		mergeMap(rep.PorDepartamento, p.dep)
		mergeMap(rep.PorRubro, p.rub)
		mergeMap(rep.PorTipoContrat, p.tc)
		mergeMap(rep.PorNivelGobierno, p.niv)
		mergeMap(rep.PorObjeto, p.obj)
		mergeMap(rep.PorTipoContratist, p.tcon)
		mergeMap(ent, p.ent)
		for k, v := range p.entDep {
			entDep[k] = v
		}
		for k, v := range p.hist {
			rep.HistLogMonto[k] += v
		}
		for k, v := range p.cerca {
			rep.Cerca8UIT[k] += v
		}
	}
	// top 15 entidades por monto en riesgo (con al menos 200 órdenes)
	var list []EntidadRiesgo
	for k, a := range ent {
		if a.Ordenes >= 200 {
			list = append(list, EntidadRiesgo{k, entDep[k], a.Ordenes, a.Riesgo, float64(a.Riesgo) / float64(a.Ordenes), a.MontoRiesgo})
		}
	}
	sort.Slice(list, func(i, j int) bool { return list[i].MontoRiesgo > list[j].MontoRiesgo })
	if len(list) > 15 {
		list = list[:15]
	}
	rep.TopEntidadesRiesgo = list
	// percentiles de monto (muestra ordenada)
	montos := make([]float64, len(data))
	ml.ParallelFor(len(data), workers, func(lo, hi, _ int) {
		for i := lo; i < hi; i++ {
			montos[i] = data[i].MontoPEN
		}
	})
	sort.Float64s(montos)
	rep.Percentiles = map[string]float64{}
	for _, q := range []float64{0.01, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99} {
		rep.Percentiles[ftoa(q)] = montos[int(q*float64(len(montos)-1))]
	}
	return rep
}

func ftoa(q float64) string {
	return "p" + itoa(int(math.Round(q*100)))
}

func itoa(i int) string {
	if i < 10 {
		return "0" + string(rune('0'+i))
	}
	return string(rune('0'+i/10)) + string(rune('0'+i%10))
}
