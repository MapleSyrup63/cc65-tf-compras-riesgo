package api

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/ml"
)

// Models agrupa los modelos servidos por /predict. Es INMUTABLE una vez cargado:
// se reemplaza completo con un atomic.Pointer (tras un reentrenamiento), por lo que las
// peticiones en vuelo nunca ven un modelo a medias y no hay carreras.
type Models struct {
	RF       *ml.Forest
	Binner   *ml.Binner
	RFThr    float64
	LR       *ml.LogReg
	LRThr    float64
	Features []string
	Loaded   time.Time
	Version  string

	idx map[string]int
}

type rfFile struct {
	Binner   *ml.Binner `json:"binner"`
	Forest   *ml.Forest `json:"forest"`
	Umbral   float64    `json:"umbral"`
	Features []string   `json:"features"`
}

type lrFile struct {
	Model    *ml.LogReg `json:"model"`
	Umbral   float64    `json:"umbral"`
	Features []string   `json:"features"`
}

// LoadModels lee models/random_forest.json y models/logistic_regression.json.
func LoadModels(dir string) (*Models, error) {
	var rf rfFile
	var lr lrFile
	if err := readJSON(filepath.Join(dir, "random_forest.json"), &rf); err != nil {
		return nil, err
	}
	if err := readJSON(filepath.Join(dir, "logistic_regression.json"), &lr); err != nil {
		return nil, err
	}
	if rf.Forest == nil || rf.Binner == nil || len(rf.Forest.Trees) == 0 || lr.Model == nil {
		return nil, fmt.Errorf("modelos incompletos en %s", dir)
	}
	d := rf.Forest.D
	if len(rf.Features) != d || len(lr.Model.W) != d || len(lr.Model.Mean) != d || len(rf.Binner.Edges) != d {
		return nil, fmt.Errorf("dimensiones inconsistentes entre modelos (D=%d)", d)
	}
	m := &Models{RF: rf.Forest, Binner: rf.Binner, RFThr: rf.Umbral, LR: lr.Model, LRThr: lr.Umbral,
		Features: rf.Features, Loaded: time.Now(), idx: map[string]int{}}
	for i, n := range m.Features {
		m.idx[n] = i
	}
	m.Version = fmt.Sprintf("rf%d-%s", len(rf.Forest.Trees), m.Loaded.Format("20060102T150405"))
	return m, nil
}

func readJSON(path string, v any) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return json.NewDecoder(f).Decode(v)
}

// Vector arma el vector de D variables. Las variables no enviadas toman la MEDIA del
// entrenamiento (valor neutro). "monto_pen" es un atajo que calcula log_monto_pen.
func (m *Models) Vector(in map[string]float64, montoPEN *float64) ([]float32, error) {
	x := make([]float32, len(m.Features))
	for j := range x {
		x[j] = float32(m.LR.Mean[j])
	}
	var unknown []string
	for k, v := range in {
		j, ok := m.idx[k]
		if !ok {
			unknown = append(unknown, k)
			continue
		}
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("valor no finito en %q", k)
		}
		x[j] = float32(v)
	}
	if len(unknown) > 0 {
		sort.Strings(unknown)
		return nil, fmt.Errorf("variables desconocidas: %s", strings.Join(unknown, ", "))
	}
	if montoPEN != nil {
		if *montoPEN < 0 || math.IsNaN(*montoPEN) || math.IsInf(*montoPEN, 0) {
			return nil, fmt.Errorf("monto_pen inválido")
		}
		if j, ok := m.idx["log_monto_pen"]; ok {
			x[j] = float32(math.Log1p(*montoPEN))
		}
	}
	return x, nil
}

// Score devuelve las probabilidades de riesgo del Random Forest y de la regresión logística.
func (m *Models) Score(x []float32) (rf, lr float64) {
	xb := make([]uint8, len(x))
	for j := range x {
		xb[j] = m.Binner.BinValue(j, x[j])
	}
	return m.RF.PredictRow(xb), m.LR.PredictRow(x)
}

// Nivel clasifica el riesgo según el umbral del Random Forest.
func (m *Models) Nivel(rf float64) string {
	switch {
	case rf >= m.RFThr:
		return "alto"
	case rf >= m.RFThr/2:
		return "medio"
	default:
		return "bajo"
	}
}
