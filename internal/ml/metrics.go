package ml

import "sort"

// Metrics de clasificación binaria.
type Metrics struct {
	AUC       float64 `json:"roc_auc"`
	AP        float64 `json:"pr_auc_average_precision"`
	Threshold float64 `json:"umbral"`
	Accuracy  float64 `json:"accuracy"`
	Precision float64 `json:"precision"`
	Recall    float64 `json:"recall"`
	F1        float64 `json:"f1"`
	TP        int     `json:"tp"`
	FP        int     `json:"fp"`
	TN        int     `json:"tn"`
	FN        int     `json:"fn"`
	// Recall por componente de la etiqueta (para verificar que el modelo no solo
	// aprende la regla de sobrecosto sino también las anulaciones).
	RecallAnuladas   float64 `json:"recall_anuladas"`
	RecallSobrecosto float64 `json:"recall_sobrecosto"`
}

// AUCAP calcula ROC-AUC (Mann-Whitney) y Average Precision.
func AUCAP(scores []float64, y []uint8) (auc, ap float64) {
	idx := make([]int, len(scores))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return scores[idx[a]] > scores[idx[b]] })
	var P, N float64
	for _, v := range y {
		if v == 1 {
			P++
		} else {
			N++
		}
	}
	var tp, fp, prevTP, prevFP float64
	for k := 0; k < len(idx); {
		s := scores[idx[k]]
		for k < len(idx) && scores[idx[k]] == s { // empates
			if y[idx[k]] == 1 {
				tp++
			} else {
				fp++
			}
			k++
		}
		auc += (fp - prevFP) * (tp + prevTP) / 2
		if tp > prevTP {
			ap += (tp - prevTP) / P * (tp / (tp + fp))
		}
		prevTP, prevFP = tp, fp
	}
	auc /= P * N
	return
}

// BestF1Threshold busca el umbral que maximiza F1 (se usa sobre validación).
func BestF1Threshold(scores []float64, y []uint8) float64 {
	best, bestT := -1.0, 0.5
	for t := 0.02; t < 0.98; t += 0.01 {
		m := Evaluate(scores, y, nil, nil, t)
		if m.F1 > best {
			best, bestT = m.F1, t
		}
	}
	return bestT
}

// Evaluate calcula la matriz de confusión y métricas en un umbral.
func Evaluate(scores []float64, y, anul, sobre []uint8, thr float64) Metrics {
	var m Metrics
	m.Threshold = thr
	var an, anHit, so, soHit int
	for i, s := range scores {
		pred := s >= thr
		switch {
		case pred && y[i] == 1:
			m.TP++
		case pred && y[i] == 0:
			m.FP++
		case !pred && y[i] == 0:
			m.TN++
		default:
			m.FN++
		}
		if anul != nil && anul[i] == 1 {
			an++
			if pred {
				anHit++
			}
		}
		if sobre != nil && sobre[i] == 1 {
			so++
			if pred {
				soHit++
			}
		}
	}
	n := float64(len(scores))
	m.Accuracy = float64(m.TP+m.TN) / n
	if m.TP+m.FP > 0 {
		m.Precision = float64(m.TP) / float64(m.TP+m.FP)
	}
	if m.TP+m.FN > 0 {
		m.Recall = float64(m.TP) / float64(m.TP+m.FN)
	}
	if m.Precision+m.Recall > 0 {
		m.F1 = 2 * m.Precision * m.Recall / (m.Precision + m.Recall)
	}
	if an > 0 {
		m.RecallAnuladas = float64(anHit) / float64(an)
	}
	if so > 0 {
		m.RecallSobrecosto = float64(soHit) / float64(so)
	}
	return m
}
