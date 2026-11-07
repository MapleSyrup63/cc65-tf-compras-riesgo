package etl

import (
	"strconv"
	"strings"
	"time"
	"unicode"
)

// Motivos de rechazo (se contabilizan para el informe de calidad de datos).
const (
	RejColumnas    = "columnas_incompletas"
	RejRucEntidad  = "ruc_entidad_invalido"
	RejMonto       = "monto_no_numerico_o_<=1"
	RejFecha       = "fecha_invalida"
	RejOrdenFechas = "registro_antes_de_emision"
	RejEstado      = "estado_desconocido"
)

// Tipos de cambio referenciales (promedio aprox. 2026) para normalizar a soles.
// Las órdenes en moneda extranjera son <0.1 % del total.
var tipoCambio = map[string]float64{
	"NUEVOS SOLES":         1.0,
	"DOLAR NORTEAMERICANO": 3.70,
	"DOLAR CANADIENSE":     2.70,
	"EURO":                 4.10,
	"LIBRA ESTERLINA":      4.80,
	"REAL BRASILENO":       0.68,
}

var estadosValidos = map[string]bool{"EMITIDA": true, "COMPROMETIDA": true, "DEVENGADA": true, "ANULADA": true}

var accentRepl = strings.NewReplacer(
	"Á", "A", "É", "E", "Í", "I", "Ó", "O", "Ú", "U", "Ü", "U",
	"á", "A", "é", "E", "í", "I", "ó", "O", "ú", "U", "ñ", "Ñ",
)

// fixEncoding corrige el artefacto "¿" que aparece en el origen en lugar de Ñ
// (p.ej. "PE¿A" -> "PEÑA"). Si el "¿" no está entre letras se reemplaza por espacio.
func fixEncoding(s string) string {
	if !strings.ContainsRune(s, '¿') {
		return s
	}
	rs := []rune(s)
	for i, r := range rs {
		if r != '¿' {
			continue
		}
		prevL := i > 0 && unicode.IsLetter(rs[i-1])
		nextL := i+1 < len(rs) && unicode.IsLetter(rs[i+1])
		if prevL && nextL {
			rs[i] = 'Ñ'
		} else {
			rs[i] = ' '
		}
	}
	return string(rs)
}

// normText: trim, mayúsculas, corrige codificación y colapsa espacios.
func normText(s string) string {
	s = fixEncoding(strings.ToUpper(strings.TrimSpace(s)))
	return strings.Join(strings.Fields(s), " ")
}

func noAccents(s string) string { return accentRepl.Replace(s) }

func isDigits(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
}

func normDepartamento(d string) string {
	d = noAccents(normText(d))
	if strings.Contains(d, "CALLAO") {
		return "CALLAO"
	}
	return d
}

func nivelGobierno(entidad string) string {
	e := noAccents(entidad)
	switch {
	case strings.Contains(e, "MUNICIPALIDAD DISTRITAL"), strings.Contains(e, "MUNICIPALIDAD DEL CENTRO POBLADO"):
		return "MUNI_DISTRITAL"
	case strings.Contains(e, "MUNICIPALIDAD"):
		return "MUNI_PROVINCIAL"
	case strings.Contains(e, "GOBIERNO REGIONAL"), strings.Contains(e, "GOB. REG"):
		return "GOB_REGIONAL"
	case strings.Contains(e, "UNIVERSIDAD"):
		return "UNIVERSIDAD"
	case strings.Contains(e, "HOSPITAL"), strings.Contains(e, "SALUD"), strings.Contains(e, "ESSALUD"), strings.Contains(e, "RED DE "):
		return "SALUD"
	default:
		return "GOB_NACIONAL"
	}
}

func grupoTipoContratacion(t string) string {
	t = noAccents(normText(t))
	switch {
	case strings.HasPrefix(t, "CONTRATACIONES HASTA 8 UIT"):
		return "HASTA_8UIT"
	case strings.HasPrefix(t, "OTRAS CONTRATACIONES SIN PROCESO"):
		return "SIN_PROCESO"
	case strings.Contains(t, "PROCESOS DE SELECCION"):
		return "PROCESO_SELECCION"
	case strings.Contains(t, "CATALOGO"):
		return "CATALOGO"
	case strings.HasPrefix(t, "SERVICIO PUBLICO"):
		return "SERVICIO_PUBLICO"
	default: // exoneraciones, regímenes especiales, convenios, decretos de urgencia
		return "DIRECTA_OTROS"
	}
}

func tipoContratista(ruc string) string {
	if len(ruc) == 11 && isDigits(ruc) {
		switch ruc[:2] {
		case "10":
			return "PN"
		case "20":
			return "EMPRESA"
		}
	}
	return "OTRO"
}

const dateLayout = "2006-01-02"

// CleanRow valida y normaliza una fila cruda. Devuelve ("", rec) si es válida,
// o (motivo, nil) si debe descartarse. Es una función pura: segura para goroutines.
func CleanRow(row []string, archivo string) (string, *Record) {
	if len(row) < ColFechaEmision+1 {
		return RejColumnas, nil
	}
	get := func(i int) string {
		if i < len(row) {
			return row[i]
		}
		return ""
	}
	rucEnt := strings.TrimSpace(get(ColRucEntidad))
	if len(rucEnt) != 11 || !isDigits(rucEnt) {
		return RejRucEntidad, nil
	}
	monto, err := strconv.ParseFloat(strings.TrimSpace(get(ColMonto)), 64)
	if err != nil || monto < 1 {
		return RejMonto, nil
	}
	fReg, err1 := time.Parse(dateLayout, strings.TrimSpace(get(ColFechaRegistro)))
	fEmi, err2 := time.Parse(dateLayout, strings.TrimSpace(get(ColFechaEmision)))
	if err1 != nil || err2 != nil {
		return RejFecha, nil
	}
	if fReg.Before(fEmi) {
		return RejOrdenFechas, nil
	}
	estado := noAccents(normText(get(ColEstado)))
	if !estadosValidos[estado] {
		return RejEstado, nil
	}
	moneda := noAccents(normText(get(ColMoneda)))
	tc, ok := tipoCambio[moneda]
	if !ok {
		tc = 1
	}
	nro, _ := strconv.Atoi(strings.TrimSpace(get(ColNroOrden)))
	entidad := normText(get(ColEntidad))
	desc := normText(get(ColDescripcion))
	rucCon := strings.TrimSpace(get(ColRucContratista))
	return "", &Record{
		Entidad:          entidad,
		RucEntidad:       rucEnt,
		Departamento:     normDepartamento(get(ColDepartamento)),
		NivelGobierno:    nivelGobierno(entidad),
		EsOrdenCompra:    strings.Contains(strings.ToUpper(get(ColTipoOrden)), "COMPRA"),
		NroOrden:         nro,
		Orden:            normText(get(ColOrden)),
		Descripcion:      desc,
		Rubro:            ClasificarRubro(desc),
		Objeto:           noAccents(normText(get(ColObjeto))),
		Estado:           estado,
		TipoContratacion: grupoTipoContratacion(get(ColTipoContratacion)),
		MontoOriginal:    monto,
		Moneda:           moneda,
		MontoPEN:         monto * tc,
		RucContratista:   rucCon,
		TipoContratista:  tipoContratista(rucCon),
		Contratista:      normText(get(ColNombreContratista)),
		FechaRegistro:    fReg,
		FechaEmision:     fEmi,
		ArchivoMes:       archivo,
		Anulada:          estado == "ANULADA",
	}
}

// DedupKey identifica duplicados exactos de una misma orden (la misma orden
// puede publicarse en dos archivos mensuales).
func (r *Record) DedupKey() string {
	return r.RucEntidad + "|" + r.Orden + "|" + r.FechaEmision.Format(dateLayout) + "|" +
		r.RucContratista + "|" + strconv.FormatFloat(r.MontoOriginal, 'f', 2, 64)
}
